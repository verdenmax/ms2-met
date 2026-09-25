"""Frozen neg20 experiment: reliability R x removal of legacy nonempty counts.

Every arm uses the same real q01 rows, neg20 fitting pool and member splits.
Only two feature columns and the three declared drops differ between arms.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import yaml

from tools import fragment_structure_neg20 as neg20
from tools import fragment_structure_validation as reference
from tools.counterfactual_effectiveness import (
    _assign_sample_ids, _atomic_csv, _atomic_json, _load_pooled_predictions,
    _model_metrics, _prepare_real, _provenance, _sha256, verify_effectiveness_bundle,
)
from tools.spec_trainer.src.feature_groups import resolve_experiment_arm
from tools.spec_trainer.src.cv_train import validate_fit_augmentation
from workflows.fragment_reliability import FEATURE_NAMES, STATUS_COLUMNS, VERSION


SCHEMA = 'fragment_reliability_ablation_v1'
SAMPLE, GROUP, OUTER = reference.SAMPLE, reference.GROUP, reference.OUTER
SEMANTICS = reference.SEMANTICS
ARMS = {'a0': 'ms1_ms2_qds', 'a1': 'ms1_ms2_qds_r',
        'a2': 'ms1_ms2_qds', 'a3': 'ms1_ms2_qds_r'}
COUNT_DROPS = ('b_count', 'y_count', 'all_count')
CONTRASTS = {
    'reliability_full': {'a1': 1, 'a0': -1},
    'drop_counts': {'a2': 1, 'a0': -1},
    'reliability_pruned': {'a3': 1, 'a2': -1},
    'combined': {'a3': 1, 'a0': -1},
    'interaction': {'a3': 1, 'a2': -1, 'a1': -1, 'a0': 1},
}
R_COLUMNS = [*FEATURE_NAMES, *STATUS_COLUMNS]
CODE_FILES = sorted(set(neg20.CODE_FILES) | {
    'tools/fragment_reliability_ablation.py', 'tools/extract_fragment_structure.py',
    'workflows/fragment_reliability.py', 'workflows/q1a_helpers.py',
    'workflows/single_work.py', 'constant/keys.py',
})


def _reference_protocol(root):
    """Imported reference bundles retain server paths and historical code hashes."""
    verified = verify_effectiveness_bundle(root)
    protocol = json.loads((root/'protocol.json').read_text())
    if (protocol.get('schema') != neg20.SCHEMA or protocol.get('arms') != neg20.ARMS
            or any(protocol.get(k) != v for k, v in SEMANTICS.items())):
        raise ValueError('R ablation requires a frozen neg20 experiment as reference')
    listed = json.loads((root/'artifact_checksums.json').read_text())['artifacts']
    required = {'cohort.csv', 'neg20_pool.csv', 'source_outer_fold_manifest.csv',
                'outer_fold_manifest.csv', 'protocol.json', 'augmentation_audit.json'}
    for fold in range(5):
        required.update(f'folds/fold_{fold}/{name}.csv'
                        for name in ('train_real_q01', 'test_real_q01', 'fit_neg20'))
        required.add(f'configs/fold_{fold}/b_qds_neg20.yaml')
    if required-set(listed):
        raise ValueError('reference checksums do not cover all frozen neg20 artifacts')
    return verified, protocol


def _identified(extracted):
    current, _ = _prepare_real(extracted.reset_index(drop=True), .20)
    for name in ('parent_id', 'query_id'):
        current[name] = current[name].astype(object).where(current[name].notna(), np.nan)
    _assign_sample_ids(current)
    return current.set_index(SAMPLE, drop=False)


def _validate_reliability(frame):
    missing = set(R_COLUMNS)-set(frame)
    if missing:
        raise ValueError(f'extraction missing R columns: {sorted(missing)}')
    values = frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors='raise')
    valid = reference._integer(frame.fragment_reliability_valid, 'fragment_reliability_valid')
    if (not valid.isin([0, 1]).all()
            or not frame.fragment_reliability_version.eq(VERSION).all()
            or not frame.fragment_reliability_status.eq('ok').equals(valid.eq(1))
            or not valid.equals(frame.fragment_structure_valid.astype('int64'))
            or not frame.fragment_reliability_status.equals(frame.fragment_structure_status)
            or values.loc[valid.eq(0)].notna().any().any()
            or not np.isfinite(values.loc[valid.eq(1)].to_numpy()).all()
            or ((values < 0) | (values > 1)).any().any()):
        raise ValueError('invalid R availability, version or values')
    lengths = frame.sequence.str.len()
    if (lengths[valid.eq(1)].lt(2).any()
            or (values[FEATURE_NAMES[0]] > frame.ms2_structure_main_cut_fraction+1e-9).any()
            or (values[FEATURE_NAMES[1]] < frame.ms2_structure_main_longest_gap_fraction-1e-9).any()
            or (values[FEATURE_NAMES[1]] < 1/lengths-1e-9).any()):
        raise ValueError('R cannot add cuts or reduce the original main-group gap')
    n = lengths[valid.eq(1)].to_numpy()
    cuts = values.loc[valid.eq(1), FEATURE_NAMES[0]].to_numpy() * (n-1)
    gap = values.loc[valid.eq(1), FEATURE_NAMES[1]].to_numpy() * n
    if (not np.allclose(cuts, np.rint(cuts), rtol=0, atol=1e-8)
            or not np.allclose(gap, np.rint(gap), rtol=0, atol=1e-8)
            or (np.rint(gap) < np.ceil(n/(np.rint(cuts)+1))).any()
            or (np.rint(gap) > n-np.rint(cuts)).any()):
        raise ValueError('R values do not represent feasible sequence cuts/gaps')
    frame[list(FEATURE_NAMES)] = values


def _check_parity(frozen, current, columns, name):
    if set(frozen[SAMPLE])-set(current.index):
        raise ValueError(f'{name} samples missing from new extraction')
    paired = current.loc[frozen[SAMPLE]].reset_index(drop=True)
    drift = [col for col in [*columns, 'label', 'q_value']
             if not np.allclose(frozen[col].to_numpy(dtype=float), paired[col].to_numpy(dtype=float),
                                rtol=1e-9, atol=1e-9, equal_nan=True)]
    for col in reference.STATUS_COLUMNS:
        if not np.array_equal(frozen[col], paired[col]): drift.append(col)
    # These graph tokens are not all part of sample identity. Changing them
    # could silently invalidate frozen q01 groups even without adding rows.
    for col in ('peptide_group_id', 'parent_id', 'query_id', 'group_id', 'candidate_family_id'):
        left = frozen[col].fillna('').astype(str) if col in frozen else pd.Series('', index=frozen.index)
        right = paired[col].fillna('').astype(str) if col in paired else pd.Series('', index=paired.index)
        if not np.array_equal(left, right): drift.append(col)
    if drift:
        raise ValueError(f'{name} changed frozen features/relationships: {drift}')


def _append_reliability(frame, current):
    if set(R_COLUMNS) & set(frame):
        raise ValueError('reference already contains R; choose the original neg20 reference')
    result = frame.copy()
    result[R_COLUMNS] = current.loc[frame[SAMPLE], R_COLUMNS].to_numpy()
    for col in FEATURE_NAMES:
        result[col] = pd.to_numeric(result[col])
    result['fragment_reliability_valid'] = result.fragment_reliability_valid.astype('int64')
    return result


def build_bundle(reference_root, features, output_root, *, bootstrap_reps=1000):
    source, features, root = map(lambda p: Path(p).resolve(), (reference_root, features, output_root))
    if root.exists():
        raise FileExistsError(f'refusing to overwrite R experiment: {root}')
    if root.is_relative_to(source):
        raise ValueError('R output must be separate from its reference')
    if isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int) or bootstrap_reps < 1:
        raise ValueError('bootstrap_reps must be a positive integer')
    verified, old_protocol = _reference_protocol(source)
    sources = {'reliability_features': _provenance(features),
               'reference_checksums': _provenance(source/'artifact_checksums.json')}
    frozen = pd.read_csv(source/'cohort.csv')
    manifest = pd.read_csv(source/'outer_fold_manifest.csv')
    pool = pd.read_csv(source/'neg20_pool.csv')
    identity = [SAMPLE, GROUP, OUTER, 'label', 'sequence']
    if (not frozen[identity].equals(manifest[identity]) or frozen[SAMPLE].duplicated().any()
            or pool[SAMPLE].duplicated().any() or set(frozen[OUTER]) != set(range(5))
            or frozen.groupby(GROUP)[OUTER].nunique().gt(1).any()):
        raise ValueError('reference frozen population/manifest is inconsistent')
    base_columns = old_protocol['feature_columns']['b_qds_neg20']
    extracted = pd.read_csv(features)
    # Reuse the full relationship graph audit before availability filtering.
    recomputed, rejected, audit = neg20.prepare_neg20(
        frozen, pd.read_csv(source/'source_outer_fold_manifest.csv'), extracted, base_columns)
    if set(recomputed[SAMPLE]) != set(pool[SAMPLE]):
        raise ValueError('new extraction changed the frozen neg20 fitting pool')
    aligned = recomputed.set_index(SAMPLE).loc[pool[SAMPLE]]
    if not np.array_equal(aligned[GROUP], pool[GROUP]):
        raise ValueError('new extraction changed frozen augmentation groups')
    current = _identified(extracted)
    _validate_reliability(current)
    _check_parity(frozen, current, base_columns, 'q01')
    _check_parity(pool, current, base_columns, 'neg20 pool')
    cohort, new_pool = _append_reliability(frozen, current), _append_reliability(pool, current)
    if not cohort.fragment_reliability_valid.eq(1).any():
        raise ValueError('no evaluable R in frozen q01 population')
    columns = {arm: [col for col in resolve_experiment_arm(feature_arm, list(cohort), strict=True)
                     if not (arm in ('a2', 'a3') and col in COUNT_DROPS)]
               for arm, feature_arm in ARMS.items()}
    if columns['a0'] != base_columns or not set(COUNT_DROPS) <= set(base_columns):
        raise ValueError('current baseline registry differs from the frozen reference')
    analysis = {**old_protocol['analysis'], 'bootstrap_reps': bootstrap_reps,
                'bootstrap_seed': 20260919, 'primary_comparison': 'reliability_full'}
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.staging.', dir=root.parent))
    try:
        cohort.to_csv(staging/'cohort.csv', index=False)
        new_pool.to_csv(staging/'neg20_pool.csv', index=False)
        for name in ('outer_fold_manifest.csv', 'source_outer_fold_manifest.csv'):
            shutil.copyfile(source/name, staging/name)
        rejected.to_csv(staging/'rejected_bridges.csv', index=False)
        jobs, membership = [], {}
        old_audits = json.loads((source/'augmentation_audit.json').read_text())['members']
        for fold in range(5):
            folder = Path('folds')/f'fold_{fold}'
            (staging/folder).mkdir(parents=True)
            frames = {}
            for role in ('train', 'test'):
                original = pd.read_csv(source/folder/f'{role}_real_q01.csv')
                expected = frozen.loc[frozen[OUTER].ne(fold) if role == 'train' else frozen[OUTER].eq(fold)]
                if not original[identity].reset_index(drop=True).equals(expected[identity].reset_index(drop=True)):
                    raise ValueError('reference fold membership differs from its manifest')
                _check_parity(original, current, base_columns, f'fold {fold} {role}')
                frames[role] = _append_reliability(original, current)
                frames[role].to_csv(staging/folder/f'{role}_real_q01.csv', index=False)
            fit = pd.read_csv(source/folder/'fit_neg20.csv')
            expected_fit = pool.loc[~pool[GROUP].isin(frames['test'][GROUP])]
            if not fit[identity].reset_index(drop=True).equals(expected_fit[identity].reset_index(drop=True)):
                raise ValueError('reference fold fitting pool differs from the frozen pool')
            _check_parity(fit, current, base_columns, f'fold {fold} neg20 pool')
            fit = _append_reliability(fit, current)
            fit.to_csv(staging/folder/'fit_neg20.csv', index=False)
            template = yaml.safe_load((source/f'configs/fold_{fold}/b_qds_neg20.yaml').read_text())
            if (template['data'].get('feature_arm') != 'ms1_ms2_qds'
                    or template['data'].get('drop_features')
                    or template['data'].get('feature_cols')
                    or not template['data'].get('fit_augmentation_files')):
                raise ValueError('reference baseline is not the complete QDS + neg20 arm')
            reference._validate_inner(frames['train'], template)
            validate_fit_augmentation(frames['train'], fit, frames['test'], columns['a1'])
            membership[str(fold)] = neg20.member_augmentation_audits(frames['train'], fit, template)
            if membership[str(fold)] != old_audits[str(fold)]:
                raise ValueError('R build changed member fitting rows/counts')
            for arm, feature_arm in ARMS.items():
                cfg = copy.deepcopy(template)
                cfg['data'].update(train_files=[str(root/folder/'train_real_q01.csv')],
                    test_files=[str(root/folder/'test_real_q01.csv')],
                    fit_augmentation_files=[str(root/folder/'fit_neg20.csv')],
                    feature_arm=feature_arm, feature_cols=[], require_complete_arm=True,
                    drop_features=list(COUNT_DROPS) if arm in ('a2', 'a3') else [])
                directory = Path('training')/f'fold_{fold}'/arm
                cfg['output'] = {'model_path': str(root/directory/'models/cv.txt'),
                                 'result_path': str(root/directory/'training.cv.json')}
                cfg['structure_validation_contract'] = {'schema': SCHEMA, 'arm': arm,
                    'outer_folds_preserved': True, 'shared_inner_protocol': True,
                    'is_independent_confirmation': False, 'fit_augmentation_only': True}
                relative = Path('configs')/f'fold_{fold}'/f'{arm}.yaml'
                (staging/relative).parent.mkdir(parents=True, exist_ok=True)
                (staging/relative).write_text(yaml.safe_dump(cfg, sort_keys=False))
                jobs.append({'fold': fold, 'arm': arm, 'config': str(relative), 'result_dir': str(directory)})
        for item in sources.values():
            if _sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('input changed during R build')
        _reference_protocol(source)
        _atomic_json(staging/'augmentation_audit.json', {**audit, 'members': membership,
                     'frozen_neg20_pool_preserved': True, 'r_quality_filter_applied': False})
        _atomic_json(staging/'reliability_audit.json', {**SEMANTICS,
            'n_frozen_q01': len(cohort), 'n_frozen_neg20': len(new_pool),
            'q01_status_counts': cohort.fragment_reliability_status.value_counts().to_dict(),
            'neg20_status_counts': new_pool.fragment_reliability_status.value_counts().to_dict(),
            'original_columns_preserved': base_columns, 'appended_columns': R_COLUMNS})
        _atomic_json(staging/'protocol.json', {**SEMANTICS, 'schema': SCHEMA,
            'reference_root': str(source), 'reference_verification': verified, 'sources': sources,
            'arms': ARMS, 'count_drops': COUNT_DROPS, 'feature_columns': columns, 'jobs': jobs,
            'analysis': analysis, 'contrasts': CONTRASTS, 'is_independent_confirmation': False,
            'reliability_version': VERSION,
            'code_sha256': {p: _sha256(reference.PROJECT/p) for p in CODE_FILES}})
        _atomic_json(staging/'artifact_checksums.json', {'algorithm': 'sha256', 'artifacts': {
            str(p.relative_to(staging)): _sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}})
        _atomic_json(staging/'bundle_status.json', {**SEMANTICS, 'schema': SCHEMA, 'status': 'prepared'})
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def verify_bundle(root):
    root = Path(root).resolve()
    verified = verify_effectiveness_bundle(root)
    protocol = json.loads((root/'protocol.json').read_text())
    if (protocol.get('schema') != SCHEMA or protocol.get('arms') != ARMS
            or protocol.get('contrasts') != CONTRASTS
            or any(protocol.get(k) != v for k, v in SEMANTICS.items())):
        raise ValueError('incompatible R ablation protocol')
    for path, digest in protocol['code_sha256'].items():
        if _sha256(reference.PROJECT/path) != digest:
            raise ValueError(f'implementation changed since freeze: {path}; choose a new output root')
    return verified, protocol


def validate_job(root, job):
    reference._validate_completed_job(root, job)
    result = json.loads((root/job['result_dir']/'training.cv.json').read_text())
    expected = json.loads((root/'augmentation_audit.json').read_text())['members'][str(job['fold'])]
    for member, audit in zip(result['train_fold_metrics'], expected, strict=True):
        if member.get('fit_augmentation') != {k: v for k, v in audit.items() if k != 'member'}:
            raise ValueError('completed R job did not fit the frozen augmentation rows')


def summarize_bundle(root):
    root = Path(root).resolve()
    verified, protocol = verify_bundle(root)
    for job in protocol['jobs']:
        validate_job(root, job)
    pooled = _load_pooled_predictions(root, 5, models=ARMS)
    manifest = pd.read_csv(root/'outer_fold_manifest.csv')
    keys = [SAMPLE, GROUP, OUTER, 'label', 'sequence']
    if not pooled[keys].sort_values(SAMPLE).reset_index(drop=True).equals(
            manifest[keys].sort_values(SAMPLE).reset_index(drop=True)):
        raise ValueError('R predictions changed the frozen q01 population')
    models = {arm: _model_metrics(pooled, arm) for arm in ARMS}
    comparisons, changes = neg20.paired_analysis(
        pooled, protocol['analysis'], arms=ARMS, contrasts=CONTRASTS)
    rows = []
    for arm, model in models.items():
        point = model['operating_points']['fpr_5']['external_ensemble']['test_metrics']
        rows.append({**SEMANTICS, 'arm': arm, **{k: v for k, v in model.items() if k != 'operating_points'},
            **{k: point[k] for k in ('tp', 'fp', 'fn', 'tn', 'n_actual_correct', 'n_actual_error')}})
    for name, frame in [('pooled_test_predictions.csv', pooled), ('summary.csv', pd.DataFrame(rows)),
                        ('paired_comparisons.csv', comparisons), ('transitions.csv', changes)]:
        for key, value in SEMANTICS.items(): frame[key] = value
        _atomic_csv(root/name, frame)
    summary = {**SEMANTICS, 'schema': SCHEMA, 'models': models,
        'comparisons': comparisons.to_dict(orient='records'), 'analysis': protocol['analysis'],
        'bundle_verification': verified, 'is_independent_confirmation': False,
        'bootstrap_note': 'paired class-stratified frozen-group resampling; no retraining; exploratory intervals unadjusted',
        'interpretation': 'Identical neg20 fitting pool and real q01 early-stop/OOF/test rows. '
            'Only R columns and declared count drops differ. FPR names are calibration targets; '
            'external decisions are per-member calibrated majority votes. Development analysis, not independent confirmation.'}
    lines = ['# 可靠切割 R × 非空计数消融', '',
        '四臂均使用相同 neg20 拟合池及冻结 q01 测试。FPR5 是训练校准目标，实际测试 FPR 单独列出。', '',
        '| 臂 | 设置 | ROC-AUC | FP | FN | 实际 FPR | FNR |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    descriptions = {'a0': 'B+QDS', 'a1': 'B+QDS+R', 'a2': 'B+QDS 去除旧非空计数', 'a3': '去除旧非空计数 + R'}
    for arm, model in models.items():
        point = model['operating_points']['fpr_5']['external_ensemble']['test_metrics']
        lines.append(f'| {arm} | {descriptions[arm]} | {model["roc_auc"]:.4f} | {point["fp"]} | {point["fn"]} | {point["fpr"]:.3%} | {point["fnr"]:.3%} |')
    lines += ['', '| 比较 | 错误召回变化及 95% 区间 | 实际 FPR 变化及 95% 区间 |', '| --- | --- | --- |']
    for row in comparisons.to_dict(orient='records'):
        lines.append(f'| {row["comparison"]} | {row["error_recall_delta_at_fpr5"]:.3%} '
            f'[{row["error_recall_delta_ci95_low"]:.3%}, {row["error_recall_delta_ci95_high"]:.3%}] | '
            f'{row["observed_fpr_delta_at_fpr5"]:.3%} [{row["observed_fpr_delta_ci95_low"]:.3%}, {row["observed_fpr_delta_ci95_high"]:.3%}] |')
    supported = comparisons.set_index('comparison').loc['reliability_full', 'development_support']
    lines += ['', '主比较：reliability_full（a1−a0）。',
        '主比较达到预声明开发条件。' if supported else '主比较未达到预声明开发条件。',
        '去除的仅为 b_count/y_count/all_count，其它计数代理仍可能存在。',
        '该数据已用于假设发现，不构成独立确认；需同时检查新增 FN 与 FP。',
        'metric_semantics=error_identification_positive_v1；positive_class=incorrect_identification。', '']
    (root/'report.md').write_text('\n'.join(lines))
    _atomic_json(root/'summary.json', summary)
    _atomic_json(root/'bundle_status.json', {**SEMANTICS, 'schema': SCHEMA, 'status': 'complete'})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check-reference', 'build', 'verify', 'train', 'summarize', 'run'])
    parser.add_argument('--reference-root', type=Path, default=Path('runs/spec_trainer/single-peptide-structure-neg20'))
    parser.add_argument('--features', type=Path, default=Path('runs/baseline_2da_neg20_reliability/features.csv'))
    parser.add_argument('--output-root', type=Path, default=Path('runs/spec_trainer/single-peptide-reliability-ablation'))
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == 'check-reference':
        print(_reference_protocol(args.reference_root.resolve())[0], flush=True)
        return
    root = args.output_root.resolve()
    if args.command in ('build', 'run'):
        if args.command == 'run' and root.exists():
            _, protocol = verify_bundle(root)
            for key, path in [('reliability_features', args.features), ('reference_checksums', args.reference_root/'artifact_checksums.json')]:
                if _sha256(path) != protocol['sources'][key]['sha256']:
                    raise ValueError(f'resume input changed: {key}; choose a new output root')
            if args.bootstrap_reps != protocol['analysis']['bootstrap_reps']:
                raise ValueError('resume bootstrap settings differ from the frozen experiment')
        else:
            build_bundle(args.reference_root, args.features, root, bootstrap_reps=args.bootstrap_reps)
    if args.command == 'verify':
        print(verify_bundle(root)[0], flush=True)
    if args.command in ('run', 'train'):
        reference.train_bundle(root, verifier=verify_bundle, job_validator=validate_job)
    if args.command in ('run', 'summarize'):
        summarize_bundle(root)
    print(f'{args.command} complete: {root}', flush=True)


if __name__ == '__main__':
    main()
