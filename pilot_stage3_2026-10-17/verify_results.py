"""Проверяем сохранённые ошибки калибровку, места сервисов и метрики через NumPy"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Все пути к результатам считаем от папки этого скрипта
ROOT = Path(__file__).resolve().parent


def read(path):
    # читаем JSON и получаем обычный список или словарь
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    # по этой строке можно проверить, менялся ли файл
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    # Сначала создаём папку потом записываем результат в JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main(argv=None):
    # Пути можно передать при запуске, чтобы не менять сам код
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage2-dir', type=Path, help='Override the input directory from the recorded protocol.')
    parser.add_argument('--recorded-source-dir', type=Path,
                        help='Original source files for auditing a run made before source edits.')
    parser.add_argument('--output-dir', type=Path,
                        help='Write audit reports here instead of updating the project reports.')
    args = parser.parse_args(argv)
    # recorded хранит исходники того запуска который сейчас проверяем
    recorded = args.recorded_source_dir.resolve() if args.recorded_source_dir else ROOT
    output = args.output_dir.resolve() if args.output_dir else ROOT
    # Старые результаты нельзя затереть отчётом новой проверки
    if args.recorded_source_dir and (output == ROOT or ROOT in output.parents):
        parser.error('An archived audit requires --output-dir outside the recorded run directory.')
    p = read(recorded / 'protocol.json')
    stage2 = args.stage2_dir.resolve() if args.stage2_dir else (ROOT / p['stage2_directory']).resolve()
    training = read(ROOT / 'training_complete.json')
    # сравниваем файлы с теми, которые были при обучении
    assert training['protocol_sha256'] == sha(recorded / 'protocol.json'), 'Protocol differs from the recorded run; use --recorded-source-dir.'
    assert training['source_sha256'] == sha(recorded / 'run_stage3.py'), 'Source differs from the recorded run; use --recorded-source-dir.'
    if args.recorded_source_dir:
        # четыре старых исходника берём из архива остальные файлы из проекта
        source_names = {'run_stage3.py', 'verify_results.py', 'protocol.json', 'README.md'}
        for entry in read(ROOT / 'completion_manifest.json')['files']:
            base = recorded if entry['path'] in source_names else ROOT
            path = base / entry['path']
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['path']
    schema = read(stage2 / 'feature_schema.json')
    evaluation = read(ROOT / 'results' / 'evaluation.json')
    rankings = read(ROOT / 'results' / 'service_rankings.json')
    incidents = read(ROOT / 'results' / 'incident_results.json')
    # Для каждого случая сразу сохраняем его правильный ответ по имени случая
    labels = {r['case']: r for r in read(stage2 / 'test_labels.json')}
    # файл с оценками должен совпадать с тем, что был записан при оценке модели
    assert evaluation['predictions_written_before_label_read_sha256'] == sha(ROOT / 'results' / 'service_rankings.json')
    checked_inputs = 0
    # Проверяем что входные файлы второго этапа не изменились
    for f in read(ROOT / 'input_provenance.json')['files']:
        assert sha(stage2 / f['path']) == f['sha256'], f['path']
        checked_inputs += 1
    models = []
    for rec in training['models']:
        folder = ROOT / 'models' / rec['run']
        # Веса и параметры приведения ошибок к общему масштабу тоже проверяем по хешам
        assert sha(folder / 'model.pt') == rec['model_sha256']
        assert sha(folder / 'calibrator.npz') == rec['calibrator_sha256']
        assert sha(folder / 'initial.pt') == rec['initial_state_sha256']
        curve = read(folder / 'history.json')
        # Должны быть выбраны веса с самой маленькой ошибкой на проверочных данных
        best = min(curve, key=lambda r: r['validation_loss'])
        assert rec['chosen_epoch'] == best['epoch']
        assert rec['chosen_validation_loss'] == best['validation_loss']
        assert rec['updates'] == 840
        with np.load(stage2 / 'archives' / f"{p['archive_seed']}_{rec['regime']}.npz") as a:
            # берём только строки проверки и признаки нужных источников
            val = a['validation_rows']
            service_ids = a['service_indices'][val]
            sources = p['components'][rec['component']]
            cols = sum([schema['source_feature_indices'][s] for s in sources], [])
            valid_targets = a['observed'][val][:, cols]
        with np.load(folder / 'normal_residuals.npz') as n, np.load(folder / 'calibrator.npz') as cal:
            errors = n['validation_feature_errors']
            # Ошибки не могут быть отрицательными. Для отсутствующих признаков должен быть ноль
            assert np.isfinite(errors).all() and np.all(errors >= 0)
            assert np.all(errors[~valid_targets] == 0)
            # offset показывает начало признаков очередного источника
            offset = 0
            for j, source in enumerate(sources):
                count = len(schema['source_feature_indices'][source])
                # считаем среднюю ошибку только по признакам, для которых есть данные
                active = valid_targets[:, offset:offset + count].sum(1)
                # если данных по признакам нет среднюю ошибку не считаем
                expected = np.full(len(active), np.nan)
                ok = active > 0
                expected[ok] = errors[:, offset:offset + count].sum(1)[ok] / active[ok]
                assert np.allclose(expected, n['validation'][:, j], equal_nan=True, atol=1e-6)
                for si in range(len(schema['candidates'])):
                    # Для каждого сервиса заново считаем медиану и разброс ошибок
                    values = expected[(service_ids == si) & np.isfinite(expected)]
                    # При сложении чисел может быть небольшая разница из-за округления
                    q = np.quantile(values, [.25, .75])
                    assert np.isclose(cal['center'][si, j], np.median(values), atol=1e-6)
                    assert np.isclose(cal['scale'][si, j], max(q[1] - q[0], .01), atol=1e-6)
                    assert cal['counts'][si, j] == len(values)
                offset += count
            # Средняя ошибка по источникам должна совпасть с итогом обучения
            assert np.isclose(np.nanmean(np.nanmean(n['validation'], axis=0)), rec['chosen_validation_loss'])
            # Эти числа нужны для короткого отчёта по каждой модели
            models.append({'run': rec['run'], 'validation_improvement_vs_constant_percent':
                           100 * (1 - rec['chosen_validation_loss'] / rec['constant_validation_loss']),
                           'calibration_scale_floor_cells': int((cal['scale'] == .01).sum()),
                           'calibration_cells': int(cal['scale'].size),
                           'calibration_count_range': [int(cal['counts'].min()), int(cal['counts'].max())]})
    # собираем сервисы одного случая и одного варианта оценки в один список
    groups = defaultdict(list)
    for r in rankings:
        groups[r['regime'], r['view'], r['method'], r['case']].append(r)
    # Здесь три режима истории, три набора источников четыре метода и 35 случаев
    assert len(groups) == 3 * 3 * 4 * 35
    assert len(rankings) == len(groups) * 11
    incident_lookup = {}
    for row in incidents:
        key = row['regime'], row['view'], row['method'], row['case']
        # Один и тот же результат случая не должен встречаться два раза
        assert key not in incident_lookup, ('Duplicate incident record', key)
        incident_lookup[key] = row
    assert set(incident_lookup) == set(groups)
    tie_details = []
    for key, group in groups.items():
        assert sorted(r['service'] for r in group) == schema['candidates']
        # Больше оценка — выше место. Равные оценки сортируем по имени сервиса
        ordered = sorted(group, key=lambda r: (-r['score'], r['service']))
        assert [r['rank'] for r in ordered] == list(range(1, 12))
        assert all(np.isfinite(r['score']) and r['score'] >= 0 for r in group)
        # Находим сервис, который указан в разметке как причина сбоя
        root = next(r for r in group if r['service'] == labels[key[3]]['root_cause_service'])
        incident = incident_lookup[key]
        # Краткая запись случая должна совпадать с полным списком сервисов
        assert incident['root_service'] == root['service']
        assert incident['root_rank'] == root['rank']
        assert np.isclose(incident['reciprocal_rank'], 1 / root['rank'])
        assert incident['top_service'] == ordered[0]['service']
        assert incident['extra_tied_candidates'] == len(group) - len({r['score'] for r in group})
        # При равных оценках правильный сервис мог бы занять место от lo до hi
        lo = 1 + sum(r['score'] > root['score'] for r in group)
        hi = sum(r['score'] >= root['score'] for r in group)
        # Считаем средние метрики если равные оценки упорядочить случайно
        tie_details.append(dict(regime=key[0], view=key[1], method=key[2], case=key[3],
                                deterministic_rank=root['rank'], best_tie_rank=lo, worst_tie_rank=hi,
                                expected_reciprocal_rank=float(np.mean([1 / k for k in range(lo, hi + 1)])),
                                expected_top1=max(0, min(hi, 1) - lo + 1) / (hi - lo + 1),
                                expected_top3=max(0, min(hi, 3) - lo + 1) / (hi - lo + 1),
                                top_score_tie_count=sum(r['score'] == ordered[0]['score'] for r in group)))
    tie_summary = []
    for agg in evaluation['aggregate']:
        key = agg['regime'], agg['view'], agg['method']
        # проверяем итоговые метрики по отдельным случаям, а не берём их на веру
        rows = [r for r in incidents if (r['regime'], r['view'], r['method']) == key]
        assert len(rows) == len(labels) == 35
        assert {r['case'] for r in rows} == set(labels)
        assert agg['incidents'] == len(rows)
        # MRR — это среднее от 1 / место правильного сервиса
        assert np.isclose(agg['MRR'], np.mean([1 / r['root_rank'] for r in rows]))
        # Top1 и Top3 показывают как часто правильный сервис попал в первые места
        assert agg['top1_count'] == sum(r['root_rank'] == 1 for r in rows)
        assert agg['top3_count'] == sum(r['root_rank'] <= 3 for r in rows)
        assert np.isclose(agg['Top1'], agg['top1_count'] / len(rows))
        assert np.isclose(agg['Top3'], agg['top3_count'] / len(rows))
        assert agg['incidents_with_ties'] == sum(r['extra_tied_candidates'] > 0 for r in rows)
        # Отдельно собираем итог с учётом случайного порядка равных оценок
        tied = [r for r in tie_details if (r['regime'], r['view'], r['method']) == key]
        tie_summary.append(dict(regime=key[0], view=key[1], method=key[2],
                                cases_with_root_tie=sum(r['worst_tie_rank'] > r['best_tie_rank'] for r in tied),
                                cases_with_top_tie=sum(r['top_score_tie_count'] > 1 for r in tied),
                                expected_tiebreak_MRR=float(np.mean([r['expected_reciprocal_rank'] for r in tied])),
                                expected_tiebreak_Top1=float(np.mean([r['expected_top1'] for r in tied])),
                                expected_tiebreak_Top3=float(np.mean([r['expected_top3'] for r in tied]))))
    # Для метода только с M должны совпасть сами оценки во всех режимах и наборах
    ref = {(r['case'], r['service']): r['score'] for r in rankings if r['method'] == 'metrics' and r['regime'] == 'q0' and r['view'] == 'MLT'}
    assert all(r['score'] == ref[r['case'], r['service']] for r in rankings if r['method'] == 'metrics')
    delta = []
    # Сравниваем совместную модель в q25 и q50 с тем же методом в q0
    joint0 = {r['case']: r for r in incidents if r['method'] == 'joint' and r['regime'] == 'q0' and r['view'] == 'MLT'}
    for q in ['q25', 'q50']:
        other = {r['case']: r for r in incidents if r['method'] == 'joint' and r['regime'] == q and r['view'] == 'MLT'}
        changed = [{'case': c, 'root_service': labels[c]['root_cause_service'], 'q0_rank': a['root_rank'],
                    'other_rank': other[c]['root_rank'], 'delta_reciprocal_rank':
                    other[c]['reciprocal_rank'] - a['reciprocal_rank']} for c, a in joint0.items()]
        # положительная разница значит, что правильный сервис стал ближе к началу
        delta.append({'contrast': q + '-q0', 'improved': sum(r['delta_reciprocal_rank'] > 0 for r in changed),
                      'worsened': sum(r['delta_reciprocal_rank'] < 0 for r in changed),
                      'unchanged': sum(r['delta_reciprocal_rank'] == 0 for r in changed),
                      'delta_MRR': float(np.mean([r['delta_reciprocal_rank'] for r in changed])),
                      'changed_cases': [r for r in changed if r['delta_reciprocal_rank'] != 0]})
    # Пишем отчёты только после того как все проверки выше прошли
    save(output / 'results' / 'tie_sensitivity.json', {'note': 'Post-run audit, no retraining or replacement of the prespecified alphabetic tie rule. Uniform permutation within each exactly equal-score group.', 'aggregate': tie_summary, 'per_case': tie_details})
    save(output / 'results' / 'joint_rank_changes.json', delta)
    # В итоговый файл добавляем хеши, чтобы было понятно какие исходники проверяли
    result = {'passed': True, 'stage2_file_hashes_unchanged': checked_inputs,
              'frozen_protocol_and_source': True, 'checkpoint_hashes_and_selection': True,
              'masked_validation_target_errors': True, 'calibrations_recomputed': True,
              'candidate_scores_and_ranks': True, 'metrics_independently_recomputed': True,
              'exact_metric_control_across_q_and_views': True,
              'model_diagnostics': models, 'unique_models': len(models),
              'incident_method_view_regime_rankings': len(groups), 'service_ranking_rows': len(rankings),
              'source_files': {n: sha(recorded / n) for n in ['run_stage3.py', 'verify_results.py', 'protocol.json']},
              'auditor_sha256': sha(Path(__file__)),
              'audit_scope': 'Saved target errors, calibration, service ranks and aggregate metrics; no model inference or component-score fusion replay.',
              'recorded_source_directory': str(recorded)}
    save(output / 'verification.json', result)
    # В консоль выводим основные итоги, полные данные уже записаны в файлы
    print(json.dumps({'passed': True, 'models': len(models), 'ranking_rows': len(rankings),
                      'tie_audit_MLT': [r for r in tie_summary if r['view'] == 'MLT'],
                      'joint_deltas': delta}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    # При импорте файла проверка сама не запускается
    main()
