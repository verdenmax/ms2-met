"""Frozen-group, real-only Q/D/S ablation using the production CV trainer.

Build joins new features by the original experiment identity (never CSV row
position), keeps the original outer folds, and freezes one inner protocol for
all five arms. Run trains/resumes complete verified members of the experiment;
summarize compares locked member-majority decisions on identical test rows.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from tools.counterfactual_effectiveness import (
    _assign_sample_ids, _attach_inner_protocol, _atomic_csv, _atomic_json,
    _cv_config, _load_pooled_predictions, _locked_vote_metrics, _model_metrics,
    _prepare_real, _provenance, _sha256, _validate_training_result,
    verify_effectiveness_bundle,
)
from tools.spec_trainer.src.cohort import apply_training_cohort
from tools.spec_trainer.src.cv_train import (
    _configured_predefined_protocol, _predefined_cv_splits,
    _predefined_inner_split, _validate_frame,
)
from tools.spec_trainer.src.cv_core import METRIC_SEMANTICS_VERSION, threshold_at_fpr
from tools.spec_trainer.src.feature_groups import experiment_arm_features, resolve_experiment_arm
from tools.spec_trainer.src.sample_groups import prepare_cv_groups
from workflows.fragment_structure import FEATURE_NAMES, VERSION


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = 'fragment_structure_validation_v1'
SEMANTICS = {'metric_semantics': METRIC_SEMANTICS_VERSION,
             'positive_class': 'incorrect_identification'}
ARMS = {
    'b': 'ms1_ms2_no_prediction',
    'b_q': 'ms1_ms2_charge',
    'b_d': 'ms1_ms2_dedup',
    'b_s': 'ms1_ms2_structure',
    'b_qds': 'ms1_ms2_qds',
}
SAMPLE = 'experiment_sample_id'
GROUP = 'leakage_group_id'
OUTER = 'experiment_outer_fold'
SOURCE_ROW = 'experiment_source_row'
STATUS_COLUMNS = ['fragment_structure_valid', 'fragment_structure_version',
                  'fragment_structure_status']
CODE_FILES = [
    'tools/fragment_structure_validation.py',
    'tools/counterfactual_effectiveness.py',
    'tools/spec_trainer/src/cv_train.py', 'tools/spec_trainer/src/cv_core.py',
    'tools/spec_trainer/src/feature_groups.py',
    'tools/spec_trainer/src/feature_cols.py',
    'tools/spec_trainer/src/sample_groups.py', 'tools/spec_trainer/src/cohort.py',
    'tools/spec_trainer/src/models/model_manager.py',
    'tools/spec_trainer/src/models/lgb_model.py',
    'workflows/fragment_structure.py',
]


def _integer(series, name, *, minimum=0):
    values = pd.to_numeric(series, errors='coerce')
    if (not np.isfinite(values).all() or (values < minimum).any()
            or (values != np.floor(values)).any()):
        raise ValueError(f'{name} must contain finite integers >= {minimum}')
    return values.astype('int64')


def _real_with_identity(frame):
    # Match CSV serialization in the original concatenated effectiveness table.
    work, _ = _prepare_real(frame.reset_index(drop=True), .01)
    for column in ('parent_id', 'query_id'):
        work[column] = work[column].astype(object).where(work[column].notna(), np.nan)
    _assign_sample_ids(work)
    return work


def join_frozen_features(source, manifest, extracted):
    """Return the identical frozen real population with only Q/D/S appended."""
    required = {SAMPLE, GROUP, OUTER, SOURCE_ROW, 'experiment_origin', 'label', 'sequence'}
    if required - set(manifest):
        raise ValueError(f'manifest missing columns: {sorted(required-set(manifest))}')
    frozen = manifest.loc[manifest.experiment_origin.eq('real_q01')].copy()
    if frozen.empty or frozen[SAMPLE].isna().any() or frozen[SAMPLE].duplicated().any():
        raise ValueError('frozen real manifest must have nonempty unique sample IDs')
    for column in (GROUP, SAMPLE):
        if frozen[column].astype('string').str.strip().fillna('').eq('').any():
            raise ValueError(f'manifest has empty {column}')
    frozen[SOURCE_ROW] = _integer(frozen[SOURCE_ROW], SOURCE_ROW)
    frozen[OUTER] = _integer(frozen[OUTER], OUTER)
    if frozen[SOURCE_ROW].duplicated().any() or frozen[SOURCE_ROW].max() >= len(source):
        raise ValueError('manifest source rows are duplicated or outside source snapshot')
    if sorted(frozen[OUTER].unique()) != list(range(5)):
        raise ValueError('the frozen manifest must contain exactly outer folds 0..4')
    if frozen.groupby(GROUP)[OUTER].nunique().gt(1).any():
        raise ValueError('frozen outer folds split a leakage group')
    old = _real_with_identity(source).iloc[frozen[SOURCE_ROW]].reset_index(drop=True)
    frozen = frozen.reset_index(drop=True)
    for column in (SAMPLE, 'sequence', 'label'):
        if not old[column].equals(frozen[column]):
            raise ValueError(f'original source/manifest identity mismatch: {column}')
    available = _real_with_identity(extracted).set_index(SAMPLE, drop=False)
    missing = set(frozen[SAMPLE]) - set(available.index)
    if missing:
        raise ValueError(f'{len(missing)} frozen samples missing in new extraction; examples: {sorted(missing)[:5]}')
    current = available.loc[frozen[SAMPLE]].reset_index(drop=True)
    needed = set(FEATURE_NAMES) | set(STATUS_COLUMNS) | experiment_arm_features(ARMS['b'])
    if needed - set(current):
        raise ValueError(f'extraction missing feature columns: {sorted(needed-set(current))}')
    baseline_columns = sorted(experiment_arm_features(ARMS['b']))
    if set(baseline_columns) - set(old):
        raise ValueError('original feature snapshot lacks the complete baseline arm')
    changed = []
    for column in baseline_columns + ['q_value']:
        a = pd.to_numeric(old[column], errors='raise').to_numpy(dtype='f8')
        b = pd.to_numeric(current[column], errors='raise').to_numpy(dtype='f8')
        if not np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
            changed.append(column)
    if changed:
        raise ValueError(f'legacy observed feature drift against frozen source: {changed}')
    if not current.fragment_structure_version.eq(VERSION).all():
        raise ValueError(f'extraction must declare fragment_structure_version={VERSION}')
    valid = _integer(current.fragment_structure_valid, 'fragment_structure_valid')
    if not valid.isin([0, 1]).all() or not valid.eq(1).any():
        raise ValueError('new extraction has no usable Q/D/S rows or invalid availability flags')
    status = current.fragment_structure_status.astype('string').str.strip()
    if status.isna().any() or status.eq('').any() or not status.eq('ok').equals(valid.eq(1).astype('boolean')):
        raise ValueError('Q/D/S availability and status disagree')
    numeric = current[list(FEATURE_NAMES)].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy(dtype='f8')).any():
        raise ValueError('Q/D/S contains infinite values')
    if numeric.loc[valid.eq(0)].notna().any().any():
        raise ValueError('unavailable Q/D/S rows must keep all numeric features missing')
    for column in frozen:
        old[column] = frozen[column].to_numpy()
    old[list(FEATURE_NAMES)] = numeric.to_numpy()
    old[STATUS_COLUMNS] = current[STATUS_COLUMNS].to_numpy()
    old['fragment_structure_valid'] = valid.to_numpy()
    # Validate the existing cohort without reselecting it or filtering Q/D/S NA.
    eligible, cohort_audit = apply_training_cohort(old, 'evidence_observed')
    if len(eligible) != len(old):
        raise ValueError('frozen real manifest no longer matches evidence_observed cohort')
    _, grouping_audit = prepare_cv_groups(old, GROUP, frozen_group_graph=True)
    if old.groupby(GROUP).label.nunique().gt(1).any():
        raise ValueError('expected class-pure connected groups from the frozen effectiveness experiment')
    for fold in range(5):
        if set(old.loc[old[OUTER].eq(fold), 'label']) != {0, 1}:
            raise ValueError(f'outer test fold {fold} lacks one class')
    columns = {arm: resolve_experiment_arm(name, list(old.columns)) for arm, name in ARMS.items()}
    for names in columns.values():
        _validate_frame(old, names, 'label', GROUP)
    audit = {
        **SEMANTICS, 'n_rows': len(old), 'n_actual_correct': int(old.label.eq(1).sum()),
        'n_actual_error': int(old.label.eq(0).sum()), 'n_groups': int(old[GROUP].nunique()),
        'n_source_rows': len(source), 'n_extracted_rows': len(extracted),
        'n_extracted_rows_outside_frozen_cohort': len(extracted)-len(old),
        'n_structure_available': int(valid.sum()),
        'structure_status_counts': status.value_counts().to_dict(),
        'new_feature_missingness': numeric.isna().mean().to_dict(),
        'legacy_columns_checked': baseline_columns,
        'legacy_comparison_tolerance': {'rtol': 1e-9, 'atol': 1e-9},
        'cohort': cohort_audit, 'groups': grouping_audit,
        'join': 'original experiment_sample_id, checked against source snapshot and manifest',
        'qds_quality_filter_applied': False,
    }
    return old, columns, audit


def _load_design(path):
    with Path(path).open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or cfg.get('schema') != SCHEMA:
        raise ValueError(f'expected experiment schema {SCHEMA}')
    analysis = cfg.get('analysis', {})
    for key, default in (('minimum_recall_gain', .03), ('max_fpr_increase', .005)):
        value = float(analysis.get(key, default))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'invalid analysis.{key}')
        analysis[key] = value
    for key, default in (('bootstrap_reps', 1000), ('bootstrap_seed', 20260912)):
        value = analysis.get(key, default)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f'analysis.{key} must be a positive integer')
        analysis[key] = int(value)
    cfg['analysis'] = analysis
    template = Path(cfg['training_template']).expanduser()
    if not template.is_absolute():
        template = Path(path).resolve().parent/template
    return cfg, template.resolve()


def _validate_inner(train, cfg):
    folds, masks, _ = _configured_predefined_protocol(train, cfg['data'], 5)
    groups = train[GROUP]
    for member, (fit_valid, calibration) in enumerate(_predefined_cv_splits(folds, len(train), 5, groups)):
        fit, valid = _predefined_inner_split(fit_valid, calibration, masks[member], len(train), groups)
        for role, positions in [('fit', fit), ('early_stop', valid), ('calibration', calibration)]:
            if set(train.iloc[positions].label) != {0, 1}:
                raise ValueError(f'inner member {member} {role} lacks one class')
            counts = train.iloc[positions].groupby('label')[GROUP].nunique()
            if counts.min() < int(cfg['training'].get('min_class_groups_per_split', 1)):
                raise ValueError(f'inner member {member} {role} has too few independent class groups')


def build_bundle(features, source_features, manifest, config, output_root):
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f'refusing to overwrite validation bundle: {root}')
    design, template_path = _load_design(config)
    template = yaml.safe_load(template_path.read_text())
    if int(template['training'].get('cv_folds', 5)) != 5:
        raise ValueError('Q/D/S validation requires five inner members')
    # Freeze source fingerprints before reading to catch concurrent replacement.
    sources = {k: _provenance(p) for k, p in {
        'features': features, 'source_features': source_features,
        'manifest': manifest, 'config': config, 'training_template': template_path,
    }.items()}
    frame, columns, audit = join_frozen_features(
        pd.read_csv(source_features), pd.read_csv(manifest), pd.read_csv(features))
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.staging.', dir=root.parent))
    try:
        frame.to_csv(staging/'cohort.csv', index=False)
        pd.read_csv(manifest).to_csv(staging/'source_outer_fold_manifest.csv', index=False)
        frame[[SAMPLE, GROUP, OUTER, SOURCE_ROW, 'label', 'sequence']].to_csv(
            staging/'outer_fold_manifest.csv', index=False)
        jobs = []
        inner_audits = {}
        for fold in range(5):
            train = frame.loc[frame[OUTER].ne(fold)].copy()
            test = frame.loc[frame[OUTER].eq(fold)].copy()
            train, _, inner_audits[str(fold)] = _attach_inner_protocol(
                train, train.iloc[:0].copy(), template['training'])
            _validate_inner(train, _cv_config(template, root, fold, 'm_real'))
            fold_dir = staging/'folds'/f'fold_{fold}'
            fold_dir.mkdir(parents=True)
            train.to_csv(fold_dir/'train_real_q01.csv', index=False)
            test.to_csv(fold_dir/'test_real_q01.csv', index=False)
            for arm, feature_arm in ARMS.items():
                cfg = _cv_config(template, root, fold, 'm_real')
                cfg['data'].update(feature_arm=feature_arm, drop_features=[],
                                   feature_cols=[])
                result_dir = root/'training'/f'fold_{fold}'/arm
                cfg['output'] = {'model_path': str(result_dir/'models'/'cv.txt'),
                                 'result_path': str(result_dir/'training.cv.json')}
                cfg['structure_validation_contract'] = {
                    'schema': SCHEMA, 'arm': arm, 'outer_folds_preserved': True,
                    'shared_inner_protocol': True, 'is_independent_confirmation': False,
                }
                path = Path('configs')/f'fold_{fold}'/f'{arm}.yaml'
                (staging/path).parent.mkdir(parents=True, exist_ok=True)
                (staging/path).write_text(yaml.safe_dump(cfg, sort_keys=False))
                jobs.append({'fold': fold, 'arm': arm, 'config': str(path),
                             'result_dir': str(result_dir.relative_to(root))})
        for item in sources.values():
            if _sha256(Path(item['path'])) != item['sha256']:
                raise ValueError(f'input changed during build: {item["path"]}')
        _atomic_json(staging/'join_audit.json', audit)
        _atomic_json(staging/'protocol.json', {
            **SEMANTICS, 'schema': SCHEMA, 'sources': sources, 'arms': ARMS,
            'feature_columns': columns, 'jobs': jobs, 'analysis': design['analysis'],
            'primary_comparison': ['b', 'b_qds'], 'inner_protocols': inner_audits,
            'is_independent_confirmation': False,
            'code_sha256': {p: _sha256(PROJECT/p) for p in CODE_FILES},
        })
        _atomic_json(staging/'artifact_checksums.json', {
            'algorithm': 'sha256', 'artifacts': {
                str(p.relative_to(staging)): _sha256(p)
                for p in sorted(staging.rglob('*')) if p.is_file()},
        })
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
    if protocol.get('schema') != SCHEMA or protocol.get('arms') != ARMS:
        raise ValueError('incompatible Q/D/S validation protocol')
    for path, digest in protocol['code_sha256'].items():
        if _sha256(PROJECT/path) != digest:
            raise ValueError(f'implementation changed since bundle freeze: {path}; use a new output root')
    return verified, protocol


def _validate_completed_job(root, job):
    path = root/job['result_dir']
    scores = pd.read_csv(path/'training.cv.test_scores.csv', float_precision='round_trip')
    _validate_training_result(root, job['fold'], job['arm'], scores)
    cols = [SAMPLE, GROUP, OUTER, 'label']
    expected = pd.read_csv(root/'folds'/f'fold_{job["fold"]}'/'test_real_q01.csv', usecols=cols)
    if scores[SAMPLE].duplicated().any() or not expected[cols].sort_values(SAMPLE).reset_index(drop=True).equals(
            scores[cols].sort_values(SAMPLE).reset_index(drop=True)):
        raise ValueError('completed training predictions do not match frozen test membership')
    result = json.loads((path/'training.cv.json').read_text())
    if result['experiment']['feature_cols'] != json.loads((root/'protocol.json').read_text())['feature_columns'][job['arm']]:
        raise ValueError('completed model used different feature columns')
    if len(result['model_paths']) != 5 or len(result['train_fold_metrics']) != 5:
        raise ValueError('completed training requires exactly five calibrated members')
    if [m['fold'] for m in result['train_fold_metrics']] != list(range(5)):
        raise ValueError('member calibration order differs from the five-member protocol')
    oof = pd.read_csv(path/'training.cv.oof.csv', float_precision='round_trip')
    train = pd.read_csv(root/'folds'/f'fold_{job["fold"]}'/'train_real_q01.csv',
                        usecols=[GROUP, 'label', 'sequence', 'experiment_inner_fold'])
    if (len(oof) != len(train) or not np.array_equal(oof['__source_row'], np.arange(len(train)))
            or not np.array_equal(oof.oof_fold, train.experiment_inner_fold)
            or not oof[[GROUP, 'label', 'sequence']].equals(train[[GROUP, 'label', 'sequence']])):
        raise ValueError('OOF calibration rows differ from the frozen inner protocol')
    member_scores = scores[[f'member_{k}_trust_score' for k in range(5)]].to_numpy(dtype='f8').T
    if (not np.isfinite(member_scores).all() or (member_scores < 0).any()
            or (member_scores > 1).any()):
        raise ValueError('invalid member trust scores')
    if not np.allclose(member_scores.mean(axis=0), scores.ensemble_trust_score, rtol=0, atol=1e-12):
        raise ValueError('ensemble trust does not equal member average')
    for target in (1, 5, 10):
        key = f'fpr_{target}'
        external = result['operating_points'][key]['external_ensemble']
        thresholds = np.asarray(external['member_error_thresholds'], dtype='f8')
        expected_thresholds = [m['calibration_operating_points'][key]['error_threshold']
                               for m in result['train_fold_metrics']]
        if (thresholds.shape != (5,) or not np.isfinite(thresholds).all()
                or not np.array_equal(thresholds, expected_thresholds)):
            raise ValueError('external thresholds differ from member OOF calibration')
        calibrated = [threshold_at_fpr(part.label, part.trust_score, target/100)
                      for _, part in oof.groupby('oof_fold', sort=True)]
        if not np.array_equal(thresholds, calibrated):
            raise ValueError('member thresholds do not reproduce from the frozen OOF scores')
        votes = ((1-member_scores) >= thresholds[:, None]).mean(axis=0)
        if not np.allclose(votes, scores[f'{key}_error_vote_fraction'], rtol=0, atol=1e-12):
            raise ValueError('saved votes disagree with individual member thresholds')
    for model in result['model_paths']:
        if not Path(model).is_file():
            raise FileNotFoundError(f'completed job model missing: {model}')


def train_bundle(root):
    root = Path(root).resolve()
    _, protocol = verify_bundle(root)
    # One experiment writer at a time; interrupted partial jobs are retried in
    # place, while verified completed jobs are never retrained or overwritten.
    lock = root/'.train.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'training lock exists: {lock}; inspect its PID before removing a stale lock') from exc
    os.close(descriptor)
    lock.write_text(str(os.getpid())+'\n')
    try:
        for job in protocol['jobs']:
            directory = root/job['result_dir']
            if (directory/'training.cv.json').exists():
                _validate_completed_job(root, job)
                print(f'Skip verified fold {job["fold"]} / {job["arm"]}', flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            print(f'Train fold {job["fold"]} / {job["arm"]}', flush=True)
            subprocess.run([
                sys.executable, str(PROJECT/'tools/spec_trainer/src/cv_train.py'),
                '--config', str(root/job['config']),
                '--name', f'structure_fold_{job["fold"]}_{job["arm"]}',
                '--logpath', str(directory/'train.log'), '--overwrite',
            ], cwd=PROJECT, check=True)
            _validate_completed_job(root, job)
        _atomic_json(root/'bundle_status.json', {**SEMANTICS, 'schema': SCHEMA, 'status': 'trained'})
    finally:
        lock.unlink()


def _transitions(pooled):
    """Keep direction explicit; counts here are paired changes, not new metrics."""
    rows = []
    correct = pooled.label.eq(1).to_numpy()
    baseline = pooled['b_fpr_5_vote'].to_numpy() >= .5
    identity = [SAMPLE, GROUP, OUTER, 'label', 'sequence']
    for arm in ARMS:
        if arm == 'b':
            continue
        flag = pooled[f'{arm}_fpr_5_vote'].to_numpy() >= .5
        outcome = np.select([
            ~correct & ~baseline & flag, ~correct & baseline & ~flag,
            correct & ~baseline & flag, correct & baseline & ~flag,
            ~correct & ~flag, ~correct & flag, correct & flag,
        ], ['recovered_fn', 'new_fn', 'new_fp', 'recovered_fp',
            'persistent_fn', 'persistent_detected_error', 'persistent_fp'],
            default='persistent_correct_accept')
        part = pooled[identity].copy()
        part['arm'] = arm
        part['transition'] = outcome
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def paired_comparisons(pooled, analysis):
    """Paired, class-stratified group bootstrap; only B+QDS is primary."""
    codes, groups = pd.factorize(pooled[GROUP], sort=True)
    group_labels = pd.DataFrame({'code': codes, 'label': pooled.label})
    if group_labels.groupby('code').label.nunique().gt(1).any():
        raise ValueError('bootstrap requires the frozen class-pure groups')
    labels_by_group = group_labels.drop_duplicates('code').set_index('code').label
    by_class = [labels_by_group[labels_by_group.eq(label)].index.to_numpy() for label in (0, 1)]
    if any(len(part) == 0 for part in by_class):
        raise ValueError('bootstrap requires both identification classes')
    def points(weights=None):
        return {arm: _locked_vote_metrics(pooled.label, pooled[f'{arm}_fpr_5_vote'], weights)
                for arm in ARMS}
    observed = points()
    samples = {arm: [] for arm in ARMS if arm != 'b'}
    rng = np.random.default_rng(analysis['bootstrap_seed'])
    for _ in range(analysis['bootstrap_reps']):
        counts = np.zeros(len(groups))
        for part in by_class:
            counts += np.bincount(rng.choice(part, len(part), replace=True), minlength=len(groups))
        metrics = points(counts[codes])
        for arm in samples:
            samples[arm].append([metrics[arm]['error_recall']-metrics['b']['error_recall'],
                                 metrics[arm]['fpr']-metrics['b']['fpr']])
    transitions = _transitions(pooled)
    results = []
    for arm, draws in samples.items():
        bounds = np.quantile(draws, [.025, .975], axis=0)
        recall = observed[arm]['error_recall']-observed['b']['error_recall']
        fpr = observed[arm]['fpr']-observed['b']['fpr']
        counts = transitions.loc[transitions.arm.eq(arm), 'transition'].value_counts()
        results.append({
            **SEMANTICS, 'baseline_arm': 'b', 'arm': arm,
            'comparison_role': 'primary' if arm == 'b_qds' else 'exploratory',
            'error_recall_delta_at_fpr5': recall, 'observed_fpr_delta_at_fpr5': fpr,
            'error_recall_delta_ci95_low': float(bounds[0, 0]),
            'error_recall_delta_ci95_high': float(bounds[1, 0]),
            'observed_fpr_delta_ci95_low': float(bounds[0, 1]),
            'observed_fpr_delta_ci95_high': float(bounds[1, 1]),
            'recovered_fn': int(counts.get('recovered_fn', 0)),
            'new_fn': int(counts.get('new_fn', 0)),
            'net_recovered_fn': int(counts.get('recovered_fn', 0)-counts.get('new_fn', 0)),
            'new_fp': int(counts.get('new_fp', 0)), 'recovered_fp': int(counts.get('recovered_fp', 0)),
            'development_support': (bool(recall >= analysis['minimum_recall_gain']
                and fpr <= analysis['max_fpr_increase']
                and bounds[0, 0] > 0 and bounds[1, 1] <= analysis['max_fpr_increase'])
                if arm == 'b_qds' else None),
            'resampling_unit': GROUP, 'bootstrap_reps': analysis['bootstrap_reps'],
            'ci_interpretation': 'paired frozen-prediction group uncertainty; no retraining; exploratory arms unadjusted',
        })
    return pd.DataFrame(results), transitions


def summarize_bundle(root):
    root = Path(root).resolve()
    verified, protocol = verify_bundle(root)
    for job in protocol['jobs']:
        _validate_completed_job(root, job)
    pooled = _load_pooled_predictions(root, 5, models=ARMS)
    expected = pd.read_csv(root/'outer_fold_manifest.csv')
    identity = [SAMPLE, GROUP, OUTER, 'label', 'sequence']
    if not expected[identity].sort_values(SAMPLE).reset_index(drop=True).equals(
            pooled[identity].sort_values(SAMPLE).reset_index(drop=True)):
        raise ValueError('pooled test membership differs from frozen cohort')
    metrics = {arm: _model_metrics(pooled, arm) for arm in ARMS}
    comparisons, transitions = paired_comparisons(pooled, protocol['analysis'])
    summary_rows = []
    fold_rows = []
    for arm, result in metrics.items():
        point = result['operating_points']['fpr_5']['external_ensemble']['test_metrics']
        summary_rows.append({**SEMANTICS, 'arm': arm, 'feature_arm': ARMS[arm],
                             **{k: v for k, v in result.items() if k != 'operating_points'},
                             **{k: point[k] for k in ('n_actual_correct', 'n_actual_error', 'tp', 'fp', 'fn', 'tn')}})
        for fold, part in pooled.groupby(OUTER):
            observed = _locked_vote_metrics(part.label, part[f'{arm}_fpr_5_vote'])
            fold_rows.append({**SEMANTICS, 'arm': arm, OUTER: int(fold), **observed})
    group_changes = transitions.groupby(['arm', OUTER, GROUP, 'label', 'transition']).size().rename('n_rows').reset_index()
    summary = {
        **SEMANTICS, 'schema': SCHEMA, 'models': metrics,
        'comparisons': comparisons.to_dict(orient='records'), 'analysis': protocol['analysis'],
        'bundle_verification': verified, 'is_independent_confirmation': False,
        'primary_comparison': ['b', 'b_qds'],
        'interpretation': ('Development validation on previously inspected data. Working-point names refer to '
            'training-side FPR calibration; report observed external FPR. FN transitions use the newly trained B '
            'baseline, not automatically the historical 130 FNs. No test-label thresholds select the winning arm.'),
    }
    for name, frame in [('pooled_test_predictions.csv', pooled), ('summary.csv', pd.DataFrame(summary_rows)),
                        ('fold_metrics.csv', pd.DataFrame(fold_rows)), ('paired_comparisons.csv', comparisons),
                        ('transitions.csv', transitions), ('group_transitions.csv', group_changes)]:
        for key, value in SEMANTICS.items():
            frame[key] = value
        _atomic_csv(root/name, frame)
    lines = ['# Q/D/S 冻结分组开发验证', '',
             '以下 FPR/FNR 使用训练侧成员 OOF 校准后的外层多数投票；测试 FPR 不保证等于 5%。', '',
             '| 臂 | FP | FN | 实际 FPR | 实际 FNR |', '| --- | ---: | ---: | ---: | ---: |']
    for arm, result in metrics.items():
        p = result['operating_points']['fpr_5']['external_ensemble']['test_metrics']
        lines.append(f'| {arm} | {p["fp"]} | {p["fn"]} | {p["fpr"]:.3%} | {p["fnr"]:.3%} |')
    lines += ['', '| 比较臂 | 找回 FN | 新增 FN | 净找回 | 新增 FP | 召回变化及 95% 区间 | FPR 变化及 95% 区间 |',
              '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for row in comparisons.to_dict(orient='records'):
        lines.append(f'| {row["arm"]} | {row["recovered_fn"]} | {row["new_fn"]} | '
                     f'{row["net_recovered_fn"]} | {row["new_fp"]} | '
                     f'{row["error_recall_delta_at_fpr5"]:.3%} '
                     f'[{row["error_recall_delta_ci95_low"]:.3%}, {row["error_recall_delta_ci95_high"]:.3%}] | '
                     f'{row["observed_fpr_delta_at_fpr5"]:.3%} '
                     f'[{row["observed_fpr_delta_ci95_low"]:.3%}, {row["observed_fpr_delta_ci95_high"]:.3%}] |')
    supported = comparisons.set_index('arm').loc['b_qds', 'development_support']
    lines += ['', ('B+QDS 达到本轮预声明的继续验证条件。' if supported else 'B+QDS 未达到本轮预声明的继续验证条件。'),
              '主比较为 B+QDS 对 B，其余单组为探索性比较。完整区间与条件见 paired_comparisons.csv / summary.json。',
              'FN 转移以本次重新训练的 B 为基准，不自动等同此前人工审阅的 130 条 FN。',
              '该数据已用于发现特征假设，本报告不能作为独立确认；折间标准差也不是置信区间。', '']
    (root/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    _atomic_json(root/'summary.json', summary)
    _atomic_json(root/'bundle_status.json', {**SEMANTICS, 'schema': SCHEMA, 'status': 'complete'})
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['build', 'verify', 'train', 'summarize', 'run'])
    p.add_argument('--features', type=Path, default=Path('runs/baseline_2da_structure/features.csv'))
    p.add_argument('--source-features', type=Path, default=Path('runs/baseline_2da_clean/features.csv'))
    p.add_argument('--manifest', type=Path, default=Path('runs/spec_trainer/counterfactual-2da-real-q01-effectiveness/outer_fold_manifest.csv'))
    p.add_argument('--config', type=Path, default=Path('config/fragment_structure_validation.yaml'))
    p.add_argument('--output-root', type=Path, default=Path('runs/spec_trainer/single-peptide-structure-validation'))
    args = p.parse_args(argv)
    root = args.output_root.resolve()
    if args.command in ('build', 'run'):
        if args.command == 'run' and root.exists():
            _, protocol = verify_bundle(root)
            _, template = _load_design(args.config)
            for key, path in [('features', args.features), ('source_features', args.source_features),
                              ('manifest', args.manifest), ('config', args.config),
                              ('training_template', template)]:
                if _sha256(path) != protocol['sources'][key]['sha256']:
                    raise ValueError(f'resume input changed: {key}; use a new output root')
        else:
            build_bundle(args.features, args.source_features, args.manifest, args.config, root)
    if args.command == 'verify':
        print(verify_bundle(root)[0])
    if args.command in ('train', 'run'):
        train_bundle(root)
    if args.command in ('summarize', 'run'):
        summarize_bundle(root)
    print(f'{args.command} complete: {root}', flush=True)


if __name__ == '__main__':
    main()
