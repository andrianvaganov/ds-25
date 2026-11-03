"""Обучение и оценка моделей на процессоре. Запускаются отдельными командами"""
from __future__ import annotations

import argparse
import copy
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
# Сначала ищем библиотеки рядом с проектом, потом в обычном окружении Python
sys.path.insert(0, str(ROOT / 'python_deps'))
sys.path.insert(0, str(ROOT.parent / 'data_audit' / 'python_deps'))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# В протоколе лежат настройки запуска. M — метрики L — логи, T — трассировки
P = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
S2 = (ROOT / P['stage2_directory']).resolve()
MODEL_DIR = ROOT / 'models'
RESULT_DIR = ROOT / 'results'


def configure(stage2_directory=None):
    # Читаем схему данных после разбора команды чтобы справка работала без данных
    global S2, SCHEMA, SOURCES, SERVICES, COLS, SOURCE_INDEX, GROUPS
    if stage2_directory is not None:
        S2 = Path(stage2_directory).resolve()
    SCHEMA = json.loads((S2 / 'feature_schema.json').read_text(encoding='utf-8'))
    SOURCES = SCHEMA['sources']
    SERVICES = SCHEMA['candidates']
    COLS = SCHEMA['source_feature_indices']
    # Для каждого столбца запоминаем номер его источника
    SOURCE_INDEX = np.array([SOURCES.index(n.split('.')[0]) for n in SCHEMA['numeric_features']])
    GROUPS = feature_groups()


def save(path, obj):
    # Создаём папку, если её ещё нет и записываем данные в JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    # Читаем JSON в обычные списки и словари Python
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    # по хешу потом можно проверить, изменился ли файл
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(kind, **details):
    # Событие показываем в консоли и добавляем отдельной строкой в журнал
    obj = dict(event=kind, **details)
    print(json.dumps(obj, ensure_ascii=False), flush=True)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def resources():
    """Смотрим память и время работы нашего процесса в Windows"""
    # Структура нужна для вызова функции Windows. Память всего компьютера тут не считаем
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
             'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
             'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    # Переводим байты в МиБ. Ещё возвращаем время потраченное процессором
    return dict(peak_rss_mib=counters.PeakWorkingSetSize / 2**20,
                current_rss_mib=counters.WorkingSetSize / 2**20,
                process_cpu_seconds=time.process_time())


def setup():
    # Задаём число потоков и включаем повторяемые вычисления
    torch.set_num_threads(P['network']['torch_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)


def feature_groups():
    # признаки делим на четыре группы. При обучении одну из них будем скрывать
    groups = []
    for name in SCHEMA['numeric_features']:
        src, attr, *_ = name.split('.')
        if src == 'M':
            g = P['feature_groups']['M'][attr]
        elif src == 'L':
            # Число сообщений и его части держим в одной группе
            key = ('count_and_all_hash_bins' if attr == 'count' or attr.startswith('hash_')
                   else 'mean_message_chars' if attr == 'mean_message_chars'
                   else 'error_warning_timeout')
            g = P['feature_groups']['L'][key]
        else:
            assert src == 'T', src
            # Время выполнения и число трассировок относятся к разным группам
            key = 'both_duration_features' if 'duration' in attr else 'all_count_features'
            g = P['feature_groups']['T'][key]
        assert g in range(4), (name, g)
        groups.append(g)
    return np.array(groups)


def indices(component):
    # берём только столбцы источников, которые нужны этой модели
    sources = P['components'][component]
    cols = np.concatenate([COLS[s] for s in sources])
    # Номера источников здесь уже внутри выбранной модели начиная с нуля
    local_src = np.concatenate([np.full(len(COLS[s]), i) for i, s in enumerate(sources)])
    return sources, cols, local_src


def load_archive(regime):
    # Из архива забираем все массивы для выбранного режима истории
    with np.load(S2 / 'archives' / f"{P['archive_seed']}_{regime}.npz") as z:
        return {k: z[k] for k in z.files}


def project(x, observed, access, service_indices, component):
    """Выбираем нужные источники и убираем недоступные значения"""
    sources, cols, local_src = indices(component)
    # access относится к источнику целиком, observed — к отдельному признаку
    a = access[:, [SOURCES.index(s) for s in sources]].astype(bool)
    o = observed[:, cols].astype(bool) & a[:, local_src]
    # ставим нули до преобразования чтобы скрытые значения не попали в расчёт
    clean = np.where(o, x[:, cols], 0.0)
    assert np.isfinite(clean).all()
    # asinh уменьшает очень большие значения, но сохраняет их знак
    v = np.arcsinh(clean).astype('float32')
    # переводим массивы в тензоры с которыми работает PyTorch
    return {'x': torch.from_numpy(v), 'observed': torch.from_numpy(o),
            'access': torch.from_numpy(a), 'services': torch.from_numpy(service_indices.astype('int64')),
            'groups': torch.from_numpy(GROUPS[cols]), 'local_src': torch.from_numpy(local_src),
            'source_names': sources}


def subset(d, selection):
    # Выбираем строки. Общие сведения о столбцах оставляем без изменений
    ids = torch.as_tensor(np.flatnonzero(selection), dtype=torch.long)
    return {k: v[ids] if k in ['x', 'observed', 'access', 'services'] else v for k, v in d.items()}


def digest_data(d):
    # считаем общий хеш входов, масок и номеров групп
    h = hashlib.sha256()
    for k in ['x', 'observed', 'access', 'services', 'groups', 'local_src']:
        h.update(k.encode())
        h.update(d[k].numpy().tobytes())
    return h.hexdigest()


class Predictor(nn.Module):
    def __init__(self, dimensions, sources):
        super().__init__()
        h1, h2 = P['network']['hidden_widths']
        # На входе значения маски признаков, доступность источников и номер сервиса
        # два скрытых слоя на выходе предсказания числовых признаков
        self.layers = nn.Sequential(nn.Linear(2 * dimensions + sources + len(SERVICES), h1),
                                    nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, dimensions))

    def forward(self, values, observed, access, services):
        # Скрытые значения заменяем нулями, а номер сервиса кодируем набором нулей и единицы
        z = torch.cat([torch.where(observed, values, 0), observed.float(), access.float(),
                       F.one_hot(services, len(SERVICES)).float()], dim=1)
        return self.layers(z)


def inputs_and_targets(d, row_ids, chosen_groups):
    # цель — известные значения из выбранной группы. Их нельзя оставлять на входе
    original = d['observed'][row_ids]
    target = original & (d['groups'][None, :] == chosen_groups[:, None])
    input_seen = original & ~target
    values = torch.where(input_seen, d['x'][row_ids], 0)
    # Остальные значения цели обнуляем чтобы они не мешали считать ошибку
    targets = torch.where(target, d['x'][row_ids], 0)
    assert not torch.any(input_seen & target)
    return values, input_seen, targets, target


def source_probabilities(d):
    # Считаем, в какой доле строк есть хотя бы один признак каждого источника
    return torch.stack([d['observed'][:, d['local_src'] == s].any(1).float().mean()
                        for s in range(len(d['source_names']))])


def training_loss(model, d, row_ids, chosen_groups, probabilities):
    # Делаем предсказание и оставляем ошибки только для скрытых целей
    values, seen, targets, target = inputs_and_targets(d, row_ids, chosen_groups)
    pred = model(values, seen, d['access'][row_ids], d['services'][row_ids])
    errors = F.huber_loss(pred, targets, reduction='none', delta=1.0)
    errors = torch.where(target, errors, 0)
    losses = []
    for s in range(len(d['source_names'])):
        cols = d['local_src'] == s
        # Делим на число всех известных признаков источника а не только скрытых
        # множитель 4 учитывает, что случайно выбрана одна группа из четырёх
        denom = d['observed'][row_ids][:, cols].sum(1).clamp_min(1)
        # Редкий источник получает больший вес. Потом усредняем по источникам
        losses.append((4 * errors[:, cols].sum(1) / denom).mean() / probabilities[s])
    return torch.stack(losses).mean()


@torch.no_grad()
def predict_errors(model, d, zero=False):
    """Считаем ошибки. Нужная группа признаков каждый раз скрыта"""
    # Градиенты здесь не нужны: веса не меняем
    n, dim = d['x'].shape
    errors = torch.zeros((n, dim), dtype=torch.float32)
    batch = P['optimizer']['batch_size']
    if model is not None:
        model.eval()
    for begin in range(0, n, batch):
        ids = torch.arange(begin, min(begin + batch, n))
        for g in range(4):
            # проходим все четыре группы чтобы каждый известный признак проверить один раз
            chosen = torch.full((len(ids),), g)
            values, seen, targets, target = inputs_and_targets(d, ids, chosen)
            # Нулевой прогноз нужен как простой вариант для сравнения с моделью
            pred = torch.zeros_like(targets) if zero else model(values, seen, d['access'][ids], d['services'][ids])
            errors[ids] += torch.where(target, F.huber_loss(pred, targets, reduction='none'), 0)
    # Если источник не виден, его ошибка останется NaN
    residual = np.full((n, len(d['source_names'])), np.nan)
    for s in range(len(d['source_names'])):
        cols = d['local_src'] == s
        counts = d['observed'][:, cols].sum(1).numpy()
        ok = counts > 0
        # Получаем среднюю ошибку признаков отдельно для каждого источника
        residual[ok, s] = errors[:, cols].sum(1).numpy()[ok] / counts[ok]
    assert np.isfinite(errors.numpy()).all()
    return residual, errors.numpy()


def objective(residual):
    # Сначала среднее по строкам каждого источника потом по источникам. NaN пропускаем
    return float(np.nanmean(np.nanmean(residual, axis=0)))


def fit_calibrator(residual, services):
    # Для каждого сервиса и источника запоминаем обычный уровень ошибки и её разброс
    center = np.zeros((len(SERVICES), residual.shape[1]))
    scale = np.ones_like(center)
    counts = np.zeros_like(center, dtype=int)
    for s in range(len(SERVICES)):
        for j in range(residual.shape[1]):
            vals = residual[services == s, j]
            vals = vals[np.isfinite(vals)]
            assert len(vals) > 0
            center[s, j] = np.median(vals)
            q = np.quantile(vals, [.25, .75])
            # Разность квартилей — наш масштаб. Не даём ему стать слишком маленьким
            scale[s, j] = max(float(q[1] - q[0]), 0.01)
            counts[s, j] = len(vals)
    return center, scale, counts


def calibrate(residual, services, calibrator):
    # Смотрим, насколько ошибка выше обычной. Отрицательные оценки заменяем нулём
    center, scale, _ = calibrator
    return np.maximum(0, (residual - center[services]) / scale[services])


def verify_stage2():
    # проверяем размеры и хеши входных файлов перед расчётами
    manifest = read(S2 / 'completion_manifest.json')
    checked = []
    for f in manifest['files']:
        path = S2 / f['path']
        assert path.stat().st_size == f['bytes'] and sha(path) == f['sha256'], f['path']
        checked.append(f)
    # сохраняем какие именно входы и исходники использовались в запуске
    save(ROOT / 'input_provenance.json', {'stage2_manifest_sha256': sha(S2 / 'completion_manifest.json'),
         'protocol_sha256': sha(ROOT / 'protocol.json'), 'source_sha256': sha(Path(__file__)),
         'files': checked, 'note': 'Hashes read bytes; no diagnostic labels are parsed during training.'})


def preflight():
    # До обучения проверяем данные и основные правила маскирования
    verify_stage2()
    archive_info = {}
    hashes = {}
    for regime in P['regimes']:
        a = load_archive(regime)
        archive_info[regime] = {}
        for component in P['components']:
            d = project(a['x'], a['observed'], a['source_access'], a['service_indices'], component)
            hashes[regime, component] = digest_data(d)
            archive_info[regime][component] = {
                'projected_data_sha256': hashes[regime, component],
                'train_rows': int(a['train_rows'].sum()), 'validation_rows': int(a['validation_rows'].sum()),
                'train_source_probabilities': source_probabilities(subset(d, a['train_rows'])).tolist(),
                'max_abs_model_value': float(d['x'].abs().max())}
    for c in ['M', 'L', 'ML']:
        # Эти входы одинаковы между режимами, поэтому модели можно будет переиспользовать
        assert len({hashes[q, c] for q in P['regimes']}) == 1, c
    checks = technical_checks()
    save(ROOT / 'preflight.json', {'archives': archive_info, 'checks': checks,
         'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
         'torch_threads': torch.get_num_threads(), 'logical_cpus': os.cpu_count(), 'resources': resources(),
         'feature_group_by_name': dict(zip(SCHEMA['numeric_features'], GROUPS.tolist()))})
    event('preflight_passed', checks=checks, torch=torch.__version__)


def technical_checks():
    # Берём реальные данные и специально портим только то что модель видеть не должна
    a = load_archive('q0')
    d = project(a['x'], a['observed'], a['source_access'], a['service_indices'], 'MLT')
    bad_x = a['x'].copy()
    bad_o = a['observed'].copy()
    hidden = ~a['source_access'][:, SOURCE_INDEX]
    bad_x[hidden] = np.inf
    bad_o[hidden] = True
    bad_x[~a['observed'] & ~hidden] = np.nan
    poisoned = project(bad_x, bad_o, a['source_access'], a['service_indices'], 'MLT')
    assert digest_data(d) == digest_data(poisoned)
    # Ошибка и градиенты должны совпасть с вариантом без испорченных значений
    torch.manual_seed(P['model_seed'])
    dimensions = len(SCHEMA['numeric_features'])
    model = Predictor(dimensions, len(SOURCES))
    ids = torch.arange(128)
    chosen = torch.arange(len(ids)) % 4
    probs = source_probabilities(d)
    loss = training_loss(model, d, ids, chosen, probs)
    model.zero_grad()
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    loss2 = training_loss(model, poisoned, ids, chosen, probs)
    loss2.backward()
    assert torch.equal(loss, loss2)
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters()))
    # Меняем скрытые цели. Входы и предсказания от этого меняться не должны
    v, seen, _, target = inputs_and_targets(d, ids, chosen)
    changed = dict(d)
    changed['x'] = d['x'].clone()
    part = changed['x'][ids].clone()
    part[target] += 999
    changed['x'][ids] = part
    v2, seen2, _, _ = inputs_and_targets(changed, ids, chosen)
    assert torch.equal(v, v2) and torch.equal(seen, seen2)
    with torch.no_grad():
        out = model(v, seen, d['access'][ids], d['services'][ids])
        out2 = model(v2, seen2, d['access'][ids], d['services'][ids])
    assert torch.equal(out, out2)
    # проверяем, что недоступные значения не влияют на ошибку через градиенты
    grad_data = dict(d)
    grad_data['x'] = d['x'].clone().requires_grad_()
    l = training_loss(model, grad_data, ids, chosen, probs)
    l.backward()
    assert torch.count_nonzero(grad_data['x'].grad[~d['observed']]) == 0
    assert torch.isfinite(grad_data['x'].grad).all()
    # Средняя ошибка по четырём вариантам маски должна совпасть с полным расчётом
    tiny = subset(d, np.arange(len(d['x'])) < 128)
    full, _ = predict_errors(model, tiny)
    ps = source_probabilities(tiny)
    exact = sum(training_loss(model, tiny, torch.arange(128), torch.full((128,), g), ps).item() for g in range(4)) / 4
    assert abs(exact - objective(full)) < 1e-6
    # Проверяем отдельно влияние логов на трассировки и трассировок на логи
    # Признак который предсказываем, в этот момент скрыт
    for input_source, target_name in [('L', 'T.mean_duration_ms'), ('T', 'L.mean_message_chars')]:
        target_index = SCHEMA['numeric_features'].index(target_name)
        synthetic = project(np.zeros((2, dimensions), dtype='float32'),
                            np.ones((2, dimensions), bool), np.ones((2, len(SOURCES)), bool),
                            np.array([0, 0]), 'MLT')
        synthetic['x'].requires_grad_()
        chosen = torch.full((2,), int(GROUPS[target_index]))
        vv, ss, _, _ = inputs_and_targets(synthetic, torch.arange(2), chosen)
        pp = model(vv, ss, synthetic['access'], synthetic['services'])
        derivative = torch.autograd.grad(pp[:, target_index].sum(), synthetic['x'])[0]
        assert torch.count_nonzero(derivative[:, COLS[input_source]]) > 0, (input_source, target_name)
    return {'structural_and_natural_poison_invariance': True, 'hidden_target_zero_gradient': True,
            'no_heldout_target_copy': True, 'source_weighted_loss_exact_expectation': True,
            'cross_source_gradient_path': True, 'unchanged_M_L_ML_projections_across_q': True}


def train_one(regime, component, a):
    # Для каждой отдельной модели делаем свою папку
    run_id = regime + '_' + component
    folder = MODEL_DIR / run_id
    folder.mkdir(exist_ok=False)
    # seed задаёт случайную последовательность. Для одной модели он одинаковый между режимами
    seed = P['model_seed'] + list(P['components']).index(component) * 100
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    d = project(a['x'], a['observed'], a['source_access'], a['service_indices'], component)
    train = subset(d, a['train_rows'])
    val = subset(d, a['validation_rows'])
    # Сохраняем начальные веса и считаем ошибку до обучения
    model = Predictor(d['x'].shape[1], len(d['source_names']))
    torch.save(model.state_dict(), folder / 'initial.pt')
    initial, _ = predict_errors(model, val)
    zero, _ = predict_errors(None, val, zero=True)
    train_zero, _ = predict_errors(None, train, zero=True)
    # Начальные веса тоже могут оказаться лучшими поэтому сразу запоминаем их
    best_loss = objective(initial)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    optim = torch.optim.AdamW(model.parameters(), lr=P['optimizer']['learning_rate'],
                             weight_decay=P['optimizer']['weight_decay'])
    probs = source_probabilities(train)
    history = [{'epoch': 0, 'validation_loss': best_loss, 'training_sample_loss': None,
                'seconds': 0.0}]
    n = len(train['x'])
    start = time.perf_counter()
    cpu_start = time.process_time()
    updates = 0
    for epoch in range(1, P['optimizer']['epochs'] + 1):
        model.train()
        # На каждой эпохе перемешиваем строки, а внутри пакета случайно скрываем группы
        perm = torch.randperm(n, generator=generator)
        total = 0.0
        for begin in range(0, n, P['optimizer']['batch_size']):
            ids = perm[begin:begin + P['optimizer']['batch_size']]
            chosen = torch.randint(0, 4, (len(ids),), generator=generator)
            optim.zero_grad(set_to_none=True)
            loss = training_loss(model, train, ids, chosen, probs)
            assert torch.isfinite(loss), (run_id, epoch)
            loss.backward()
            # Ограничиваем слишком большие градиенты перед обновлением весов
            grad = nn.utils.clip_grad_norm_(model.parameters(), P['optimizer']['gradient_norm_clip'])
            assert torch.isfinite(grad)
            optim.step()
            total += loss.item() * len(ids)
            updates += 1
        if epoch % 5 == 0:
            # Раз в пять эпох проверяем модель на нормальной валидации
            vv, _ = predict_errors(model, val)
            loss_v = objective(vv)
            history.append({'epoch': epoch, 'validation_loss': loss_v,
                            'training_sample_loss': total / n, 'seconds': time.perf_counter() - start})
            if loss_v < best_loss:
                best_loss, best_epoch = loss_v, epoch
                # нужна копия весов иначе дальнейшее обучение изменит и сохранённый вариант
                best_state = copy.deepcopy(model.state_dict())
            event('training_checkpoint', run=run_id, epoch=epoch, normal_validation_loss=loss_v)
    # Возвращаем лучшие веса, а не обязательно веса последней эпохи
    model.load_state_dict(best_state)
    torch.save(best_state, folder / 'model.pt')
    vr, ve = predict_errors(model, val)
    tr, _ = predict_errors(model, train)
    # Обычную ошибку и её разброс считаем только по нормальной валидации
    calibrator = fit_calibrator(vr, val['services'].numpy())
    np.savez_compressed(folder / 'calibrator.npz', center=calibrator[0], scale=calibrator[1], counts=calibrator[2])
    np.savez_compressed(folder / 'normal_residuals.npz', train=tr, validation=vr,
                        validation_feature_errors=ve, constant_validation=zero)
    save(folder / 'history.json', history)
    # Собираем результаты обучения время работы и хеши сохранённых файлов
    info = {'run': run_id, 'regime': regime, 'component': component, 'seed': seed,
            'parameter_count': sum(p.numel() for p in model.parameters()), 'updates': updates,
            'chosen_epoch': best_epoch, 'initial_validation_loss': objective(initial),
            'constant_validation_loss': objective(zero), 'chosen_validation_loss': objective(vr),
            'constant_training_loss': objective(train_zero), 'chosen_training_loss': objective(tr),
            'validation_loss_by_source': dict(zip(d['source_names'], np.nanmean(vr, axis=0).tolist())),
            'constant_validation_by_source': dict(zip(d['source_names'], np.nanmean(zero, axis=0).tolist())),
            'elapsed_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu_start,
            'resources': resources(), 'projected_data_sha256': digest_data(d),
            'initial_state_sha256': sha(folder / 'initial.pt'), 'model_sha256': sha(folder / 'model.pt'),
            'calibrator_sha256': sha(folder / 'calibrator.npz')}
    save(folder / 'summary.json', info)
    event('model_complete', **info)
    return info


def train_all():
    # Готовый запуск не перезаписываем. Для нового обучения нужна новая папка
    assert not (ROOT / 'training_complete.json').exists(), 'Use a new run directory; never overwrite an evaluated pilot.'
    preflight()
    started = time.perf_counter()
    records = []
    # В routing записываем, какую сохранённую модель брать для каждого режима
    routing = {}
    for regime in P['regimes']:
        a = load_archive(regime)
        routing[regime] = {}
        for component in P['components']:
            if regime != 'q0' and component in ['M', 'L', 'ML']:
                # Эти данные не менялись поэтому повторное обучение не нужно
                routing[regime][component] = 'q0_' + component
            else:
                records.append(train_one(regime, component, a))
                routing[regime][component] = regime + '_' + component
    # проверяем одинаковые начальные веса модели между режимами истории
    for c in ['T', 'MT', 'MLT']:
        states = [torch.load(MODEL_DIR / routing[q][c] / 'initial.pt', weights_only=True) for q in P['regimes']]
        assert all(all(torch.equal(states[0][k], st[k]) for k in states[0]) for st in states[1:])
    save(ROOT / 'training_complete.json', {'models': records, 'routing': routing,
         'elapsed_seconds': time.perf_counter() - started, 'resources': resources(),
         'protocol_sha256': sha(ROOT / 'protocol.json'), 'source_sha256': sha(Path(__file__)),
         'fault_labels_parsed': False, 'identical_initial_states_across_q_verified': True})
    event('training_complete', unique_models=len(records), seconds=time.perf_counter() - started)


def transform_test(raw, access, service_indices, normalizer):
    # сначала убираем недоступные источники, только потом преобразуем числа
    permitted = access[:, SOURCE_INDEX]
    clean = np.where(permitted, raw, np.nan)
    logged = np.sign(clean) * np.log1p(np.abs(clean))
    observed = np.isfinite(logged)
    # Используем готовые параметры из обучения. По тесту ничего не подбираем
    x = (logged - normalizer['center'][service_indices]) / normalizer['scale'][service_indices]
    x[~observed] = 0
    assert np.isfinite(x).all()
    return x.astype('float32'), observed


def load_component(run_id):
    # перед загрузкой весов и калибровки проверяем что файлы не изменились
    folder = MODEL_DIR / run_id
    info = read(folder / 'summary.json')
    assert sha(folder / 'model.pt') == info['model_sha256']
    assert sha(folder / 'calibrator.npz') == info['calibrator_sha256']
    sources, cols, _ = indices(info['component'])
    model = Predictor(len(cols), len(sources))
    model.load_state_dict(torch.load(folder / 'model.pt', weights_only=True))
    with np.load(folder / 'calibrator.npz') as z:
        cal = tuple(z[k] for k in ['center', 'scale', 'counts'])
    return model, cal


def read_rows(path):
    # Читаем индекс строк из Parquet. Соединение с базой работает в памяти
    import duckdb
    con = duckdb.connect()
    result = con.execute('SELECT * FROM read_parquet(?)', [str(path)]).fetchdf().to_dict('records')
    con.close()
    return result


def model_scores(x, observed, access, services, component, model, cal):
    # Подготавливаем входы, считаем ошибки и переводим их в оценки аномальности
    d = project(x, observed, access, services, component)
    r, e = predict_errors(model, d)
    return calibrate(r, services, cal), r, e


def fuse(scores, method, view):
    # объединяем оценки только тех источников которые есть в этом представлении
    used = [s for s in SOURCES if s in view]
    if method == 'metrics':
        # Самый простой вариант использует только метрики
        return scores['M'][:, 0]
    if method == 'separate':
        return np.mean(np.stack([scores[s][:, 0] for s in used]), axis=0)
    if method == 'pairs':
        # Метрики есть в обеих парах. Сначала усредняем их, чтобы не считать дважды
        entries = {'M': (scores['ML'][:, 0] + scores['MT'][:, 0]) / 2,
                   'L': scores['ML'][:, 1], 'T': scores['MT'][:, 1]}
        return np.mean(np.stack([entries[s] for s in used]), axis=0)
    # для совместной модели берём среднее её доступных источников
    return np.mean(scores['MLT'][:, [SOURCES.index(s) for s in used]], axis=1)


def evaluate():
    # Проверяем связь с обучением и защищаем готовые результаты от перезаписи
    train = read(ROOT / 'training_complete.json')
    assert train['protocol_sha256'] == sha(ROOT / 'protocol.json'), 'Protocol changed after training.'
    assert train['source_sha256'] == sha(Path(__file__)), 'Source changed after training.'
    assert not (RESULT_DIR / 'evaluation.json').exists(), 'Do not overwrite evaluated results.'
    started = time.perf_counter()
    rows = read_rows(S2 / 'test_row_index.parquet')
    with np.load(S2 / 'test_unscaled.npz') as z:
        test = {k: z[k] for k in z.files}
    assert [r['service'] for r in rows] == [SERVICES[i] for i in test['service_indices']]
    # собираем строки сервисов для каждого случая сбоя
    cases = list(dict.fromkeys(r['case'] for r in rows))
    assert len(rows) == 385 and len(cases) == 35
    case_rows = {c: np.array([i for i, r in enumerate(rows) if r['case'] == c]) for c in cases}
    for c, ids in case_rows.items():
        assert sorted(rows[i]['service'] for i in ids) == SERVICES
    models = {r['run']: load_component(r['run']) for r in train['models']}
    all_rankings = []
    raw_outputs = {}
    for regime in P['regimes']:
        normalizer = dict(np.load(S2 / 'archives' / f"{P['archive_seed']}_{regime}_normalizer.npz"))
        for view in P['test_views']:
            # В вариантах ML и MT закрываем источник которого нет в названии
            access = test['source_access'].copy()
            for s in SOURCES:
                if s not in view:
                    access[:, SOURCES.index(s)] = False
            x, o = transform_test(test['raw'], access, test['service_indices'], normalizer)
            score_by_component = {}
            for component in P['components']:
                model, cal = models[train['routing'][regime][component]]
                scores, residual, errors = model_scores(x, o, access, test['service_indices'], component, model, cal)
                score_by_component[component] = scores
                raw_outputs[f'{regime}_{view}_{component}_scores'] = scores
                raw_outputs[f'{regime}_{view}_{component}_residual'] = residual
                raw_outputs[f'{regime}_{view}_{component}_feature_errors'] = errors
            for method in P['method_families']:
                combined = fuse(score_by_component, method, view)
                assert np.isfinite(combined).all()
                for case in cases:
                    ids = case_rows[case]
                    # Больше оценка — выше место. При равенстве сортируем по имени сервиса
                    order = sorted(ids, key=lambda i: (-combined[i], rows[i]['service']))
                    for rank, i in enumerate(order, 1):
                        all_rankings.append({'regime': regime, 'view': view, 'method': method,
                                             'case': case, 'service': rows[i]['service'], 'rank': rank,
                                             'score': float(combined[i])})
            # портим скрытый источник до нормализации. Подготовленные входы должны совпасть
            if view != 'MLT':
                poison = test['raw'].copy()
                poison[~access[:, SOURCE_INDEX]] = 1e100
                xx, oo = transform_test(poison, access, test['service_indices'], normalizer)
                assert np.array_equal(x, xx) and np.array_equal(o, oo)
    # сначала сохраняем все предсказания, потом читаем правильные причины сбоев
    save(RESULT_DIR / 'service_rankings.json', all_rankings)
    np.savez_compressed(RESULT_DIR / 'component_outputs.npz', **raw_outputs)
    prediction_hash = sha(RESULT_DIR / 'service_rankings.json')
    labels_raw = read(S2 / 'test_labels.json')
    labels = {r['case']: r for r in labels_raw}
    assert set(labels) == set(cases)
    per_case = []
    aggregate = []
    for regime in P['regimes']:
        for view in P['test_views']:
            for method in P['method_families']:
                ranks = []
                ties = []
                for case in cases:
                    group = [r for r in all_rankings if (r['regime'], r['view'], r['method'], r['case']) == (regime, view, method, case)]
                    truth = labels[case]['root_cause_service']
                    root = next(r for r in group if r['service'] == truth)
                    rank = root['rank']
                    ranks.append(rank)
                    # Считаем лишние места с повторяющейся оценкой
                    tie = len(group) - len(set(r['score'] for r in group))
                    ties.append(tie)
                    per_case.append({'regime': regime, 'view': view, 'method': method, 'case': case,
                                     'root_service': truth, 'fault': labels[case]['fault'], 'root_rank': rank,
                                     'reciprocal_rank': 1 / rank, 'top_service': group[0]['service'],
                                     'extra_tied_candidates': tie})
                # MRR — среднее 1/место. Top1 и Top3 — доля попаданий в первые 1 и 3 места
                aggregate.append({'regime': regime, 'view': view, 'method': method, 'incidents': len(ranks),
                                  'MRR': float(np.mean(1 / np.array(ranks))),
                                  'Top1': float(np.mean(np.array(ranks) <= 1)),
                                  'Top3': float(np.mean(np.array(ranks) <= 3)),
                                  'top1_count': sum(r == 1 for r in ranks),
                                  'top3_count': sum(r <= 3 for r in ranks),
                                  'incidents_with_ties': sum(t > 0 for t in ties)})
    save(RESULT_DIR / 'incident_results.json', per_case)
    # вариант только с метриками должен давать те же ранги при любых q и представлениях
    metric_ranks = [[r['root_rank'] for r in per_case if r['method'] == 'metrics' and r['regime'] == q and r['view'] == v]
                    for q in P['regimes'] for v in P['test_views']]
    assert all(metric_ranks[0] == x for x in metric_ranks[1:])
    # Для сравнения добавляем ожидаемые метрики случайной перестановки сервисов
    result = {'aggregate': aggregate, 'uniform_random_expectation': {'MRR': sum(1 / k for k in range(1, 12)) / 11,
              'Top1': 1 / 11, 'Top3': 3 / 11}, 'elapsed_seconds': time.perf_counter() - started,
              'resources': resources(), 'predictions_written_before_label_read_sha256': prediction_hash,
              'metric_reference_identical_across_q_and_views': True,
              'test_hidden_source_poison_invariance': True,
              'interpretation': 'One archive seed and one initialization. No uncertainty assessment or novelty claim.'}
    save(RESULT_DIR / 'evaluation.json', result)
    event('evaluation_complete', aggregate=[r for r in aggregate if r['view'] == 'MLT'],
          seconds=result['elapsed_seconds'])


def main():
    # Из команды выбираем проверку входов обучение или оценку
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'train', 'evaluate'])
    parser.add_argument('--stage2-dir', type=Path, help='Override the input directory from protocol.json.')
    args = parser.parse_args()
    configure(args.stage2_dir)
    setup()
    {'preflight': preflight, 'train': train_all, 'evaluate': evaluate}[args.command]()


if __name__ == '__main__':
    main()
