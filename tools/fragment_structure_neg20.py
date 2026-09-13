"""Add real neg20 errors only to fitting; preserve the existing q01 experiment.

Four paired arms use the exact reference outer/inner rows and configurations.
New candidates never enter early stopping, threshold calibration, or testing.
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

from tools import fragment_structure_validation as reference
from tools.counterfactual_effectiveness import (
    _assign_sample_ids, _atomic_csv, _atomic_json, _load_pooled_predictions,
    _locked_vote_metrics, _model_metrics, _prepare_real, _provenance, _sha256,
    verify_effectiveness_bundle,
)
from tools.spec_trainer.src.cohort import apply_training_cohort
from tools.spec_trainer.src.cv_train import (
    _configured_predefined_protocol, _predefined_cv_splits,
    _predefined_inner_split, select_fit_augmentation, validate_fit_augmentation,
)
from tools.spec_trainer.src.sample_groups import assign_leakage_groups
from workflows.fragment_structure import FEATURE_NAMES, VERSION

SCHEMA = 'fragment_structure_neg20_v1'
SAMPLE, GROUP, OUTER = reference.SAMPLE, reference.GROUP, reference.OUTER
SEMANTICS = reference.SEMANTICS
ARMS = {'b': 'ms1_ms2_no_prediction', 'b_qds': 'ms1_ms2_qds',
        'b_neg20': 'ms1_ms2_no_prediction', 'b_qds_neg20': 'ms1_ms2_qds'}
CONTRASTS = {
    'augmentation_base': {'b_neg20': 1, 'b': -1},
    'augmentation_qds': {'b_qds_neg20': 1, 'b_qds': -1},
    'structure_q01': {'b_qds': 1, 'b': -1},
    'structure_neg20': {'b_qds_neg20': 1, 'b_neg20': -1},
    'interaction': {'b_qds_neg20': 1, 'b_neg20': -1, 'b_qds': -1, 'b': 1},
}
CODE_FILES = [*reference.CODE_FILES, 'tools/fragment_structure_neg20.py']


def _reference_protocol(root):
    # Imported server bundles retain their original absolute paths. Validate
    # immutable file contents, not those historical path names or current code.
    verified = verify_effectiveness_bundle(root)
    protocol = json.loads((root/'protocol.json').read_text())
    if (protocol.get('schema') != reference.SCHEMA or protocol.get('arms') != reference.ARMS
            or any(protocol.get(k) != v for k, v in SEMANTICS.items())):
        raise ValueError('reference must be a frozen Q/D/S validation bundle')
    listed = json.loads((root/'artifact_checksums.json').read_text())['artifacts']
    required = {'cohort.csv', 'outer_fold_manifest.csv', 'source_outer_fold_manifest.csv', 'protocol.json'}
    for fold in range(5):
        required.update({f'folds/fold_{fold}/{name}_real_q01.csv' for name in ('train', 'test')})
        required.update({f'configs/fold_{fold}/{arm}.yaml' for arm in ('b', 'b_qds')})
    if required-set(listed):
        raise ValueError('reference checksum manifest does not cover required frozen artifacts')
    return verified, protocol


def prepare_neg20(frozen, graph, extracted, columns):
    """Validate feature parity and attach new rows to the full frozen graph."""
    current, _ = _prepare_real(extracted.reset_index(drop=True), .20)
    if current.loc[current.label.eq(1), 'q_value'].gt(.01).any():
        raise ValueError('neg20 must keep correct identifications at q<=0.01')
    for name in ('parent_id', 'query_id'):
        current[name] = current[name].astype(object).where(current[name].notna(), np.nan)
    _assign_sample_ids(current)
    if current.query_id.notna().any() or current.parent_id.notna().any():
        raise ValueError('neg20 input must contain real identifications, not synthetic candidates')
    required = set(columns) | set(reference.STATUS_COLUMNS)
    if required-set(current):
        raise ValueError(f'neg20 needs newly extracted complete Q/D/S: {sorted(required-set(current))}')
    numeric = current[list(FEATURE_NAMES)].apply(pd.to_numeric, errors='raise')
    valid = reference._integer(current.fragment_structure_valid, 'fragment_structure_valid')
    if (not valid.isin([0, 1]).all() or not current.fragment_structure_version.eq(VERSION).all()
            or not current.fragment_structure_status.eq('ok').equals(valid.eq(1))
            or current.fragment_structure_status.fillna('').astype(str).str.strip().eq('').any()
            or np.isinf(numeric.to_numpy()).any()
            or numeric.loc[valid.eq(0)].notna().any().any()):
        raise ValueError('neg20 has invalid Q/D/S availability, version or values')
    current[list(FEATURE_NAMES)] = numeric
    eligible, cohort_audit = apply_training_cohort(current, 'evidence_observed')
    indexed = eligible.set_index(SAMPLE)
    if set(frozen[SAMPLE])-set(indexed.index):
        raise ValueError('neg20 extraction is missing frozen q01 samples')
    paired = indexed.loc[frozen[SAMPLE]]
    drift = [c for c in [*columns, 'q_value', 'label']
             if not np.allclose(frozen[c].to_numpy(dtype=float), paired[c].to_numpy(dtype=float),
                                rtol=1e-9, atol=1e-9, equal_nan=True)]
    if drift:
        raise ValueError(f'neg20 extraction changed frozen observed/QDS features: {drift}')
    for c in reference.STATUS_COLUMNS:
        if not np.array_equal(frozen[c], paired[c]):
            raise ValueError(f'neg20 extraction changed frozen {c}')

    # Retain all original graph bridges (including non-model synthetic rows),
    # and all new input rows before availability/cohort filtering.
    anchors = pd.concat([graph, frozen], ignore_index=True, sort=False)
    if anchors[GROUP].isna().any():
        raise ValueError('reference full graph contains missing frozen groups')
    anchors['_anchor_group'] = anchors[GROUP].astype(str)
    anchors['_graph_base'] = anchors[GROUP]
    original = anchors.copy()
    assign_leakage_groups(original, '_graph_base')
    if original.groupby(GROUP)._anchor_group.nunique().gt(1).any():
        raise ValueError('reference full relationship graph breaks its frozen groups')
    new_graph = current.copy()
    # Do not trust an upstream leakage_group_id from a different experiment.
    new_graph = new_graph.drop(columns=[GROUP], errors='ignore')
    new_graph['_graph_base'] = new_graph.peptide_group_id
    joined = pd.concat([anchors, new_graph], ignore_index=True, sort=False)
    assign_leakage_groups(joined, '_graph_base')
    anchor_rows = joined.iloc[:len(anchors)]
    counts = anchor_rows.groupby(GROUP)._anchor_group.nunique()
    bridges = set(counts[counts.gt(1)].index)
    mapping = anchor_rows.groupby(GROUP)._anchor_group.first().to_dict()
    components = joined.iloc[len(anchors):].set_index(SAMPLE)[GROUP]
    extra = eligible.loc[eligible.label.eq(0) & eligible.q_value.gt(.01)].copy()
    extra['_component'] = extra[SAMPLE].map(components)
    rejected = extra.loc[extra._component.isin(bridges), [SAMPLE, 'sequence', '_component']].copy()
    rejected['reason'] = 'connects_multiple_frozen_groups'
    extra = extra.loc[~extra._component.isin(bridges)].copy()
    extra[GROUP] = extra._component.map(lambda g: mapping.get(g, 'neg20_new_'+g))
    extra = extra.drop(columns='_component')
    if extra.empty or not extra.fragment_structure_valid.eq(1).any():
        raise ValueError('no eligible new neg20 errors with usable Q/D/S remain')
    if set(extra[SAMPLE]) & set(frozen[SAMPLE]):
        raise ValueError('new neg20 rows duplicate frozen sample identities')
    extra['negative_source'] = 'real_entrapment_neg20'
    extra['experiment_origin'] = 'real_neg20_augmentation'
    extra[OUTER] = extra[GROUP].map(frozen.drop_duplicates(GROUP).set_index(GROUP)[OUTER]).fillna(-1).astype(int)
    audit = {**SEMANTICS, 'n_input': len(current), 'cohort': cohort_audit,
             'n_additional_error_before_bridge_filter': len(extra)+len(rejected),
             'n_rejected_bridge_rows': len(rejected), 'n_actual_error': len(extra),
             'n_groups': int(extra[GROUP].nunique()),
             'n_new_groups': int(extra.loc[extra[OUTER].eq(-1), GROUP].nunique()),
             'n_frozen_rows_matched': len(frozen), 'feature_parity_checked': list(columns),
             'qds_quality_filter_applied': False,
             'full_graph_before_cohort_filter': True}
    return extra.reset_index(drop=True), rejected, audit


def member_augmentation_audits(train, extra, cfg):
    folds, masks, _ = _configured_predefined_protocol(train, cfg['data'], 5)
    records = []
    for member, (fit_valid, calibration) in enumerate(_predefined_cv_splits(folds, len(train), 5, train[GROUP])):
        fit, valid = _predefined_inner_split(fit_valid, calibration, masks[member], len(train), train[GROUP])
        selected, audit = select_fit_augmentation(extra, set(train.iloc[np.r_[valid, calibration]][GROUP]))
        n_correct = int(train.iloc[fit].label.eq(1).sum())
        n_error = int(train.iloc[fit].label.eq(0).sum())+len(selected)
        records.append({'member': member, **audit, 'n_total_fit_correct': n_correct,
                        'n_total_fit_error': n_error, 'error_fraction': n_error/(n_correct+n_error)})
    return records


def build_bundle(reference_root, features, output_root, *, bootstrap_reps=1000):
    source, root, features = map(lambda p: Path(p).resolve(), (reference_root, output_root, features))
    if root.exists():
        raise FileExistsError(f'refusing to overwrite neg20 experiment: {root}')
    if root.is_relative_to(source):
        raise ValueError('neg20 output must be separate from the reference bundle')
    if isinstance(bootstrap_reps, bool) or not isinstance(bootstrap_reps, int) or bootstrap_reps < 1:
        raise ValueError('bootstrap_reps must be a positive integer')
    verified, old_protocol = _reference_protocol(source)
    sources = {'neg20_features': _provenance(features),
               'reference_checksums': _provenance(source/'artifact_checksums.json')}
    frozen = pd.read_csv(source/'cohort.csv')
    manifest = pd.read_csv(source/'outer_fold_manifest.csv')
    identity = [SAMPLE, GROUP, OUTER, 'label', 'sequence']
    if (not frozen[identity].equals(manifest[identity]) or frozen[SAMPLE].duplicated().any()
            or set(frozen[OUTER]) != set(range(5)) or frozen.groupby(GROUP)[OUTER].nunique().gt(1).any()):
        raise ValueError('reference cohort and frozen outer membership disagree')
    columns = {arm: old_protocol['feature_columns']['b_qds' if 'qds' in arm else 'b'] for arm in ARMS}
    extra, rejected, audit = prepare_neg20(frozen, pd.read_csv(source/'source_outer_fold_manifest.csv'),
                                         pd.read_csv(features), columns['b_qds'])
    analysis = {**old_protocol['analysis'], 'bootstrap_reps': bootstrap_reps,
                'bootstrap_seed': 20260913, 'primary_comparison': 'structure_neg20'}
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{root.name}.staging.', dir=root.parent))
    try:
        for name in ('cohort.csv', 'outer_fold_manifest.csv', 'source_outer_fold_manifest.csv'):
            shutil.copyfile(source/name, staging/name)
        extra.to_csv(staging/'neg20_pool.csv', index=False)
        rejected.to_csv(staging/'rejected_bridges.csv', index=False)
        jobs, membership = [], {}
        for fold in range(5):
            folder = Path('folds')/f'fold_{fold}'
            (staging/folder).mkdir(parents=True)
            train = pd.read_csv(source/folder/'train_real_q01.csv')
            test = pd.read_csv(source/folder/'test_real_q01.csv')
            for role, frame in (('train', train), ('test', test)):
                expected = frozen.loc[frozen[OUTER].ne(fold) if role == 'train' else frozen[OUTER].eq(fold)]
                if not frame[identity].reset_index(drop=True).equals(expected[identity].reset_index(drop=True)):
                    raise ValueError('reference fold rows differ from frozen outer manifest')
                shutil.copyfile(source/folder/f'{role}_real_q01.csv', staging/folder/f'{role}_real_q01.csv')
            eligible = extra.loc[~extra[GROUP].isin(test[GROUP])].copy()
            if eligible.empty:
                raise ValueError(f'outer fold {fold} has no fitting augmentation')
            validate_fit_augmentation(train, eligible, test, columns['b_qds'])
            eligible.to_csv(staging/folder/'fit_neg20.csv', index=False)
            templates = {a: yaml.safe_load((source/'configs'/f'fold_{fold}'/f'{a}.yaml').read_text()) for a in ('b', 'b_qds')}
            # Only feature arms may differ between paired jobs.
            if any(templates['b'][k] != templates['b_qds'][k] for k in ('model', 'training', 'operating_point')):
                raise ValueError('reference B and QDS training settings differ')
            for arm, feature_arm in ARMS.items():
                cfg = copy.deepcopy(templates['b_qds' if 'qds' in arm else 'b'])
                if cfg['data'].get('fit_augmentation_files'):
                    raise ValueError('reference training must not already contain fit augmentation')
                cfg['data'].update(train_files=[str(root/folder/'train_real_q01.csv')],
                                   test_files=[str(root/folder/'test_real_q01.csv')], feature_arm=feature_arm)
                reference._validate_inner(train, cfg)
                if arm.endswith('neg20'):
                    cfg['data']['fit_augmentation_files'] = [str(root/folder/'fit_neg20.csv')]
                directory = Path('training')/f'fold_{fold}'/arm
                cfg['output'] = {'model_path': str(root/directory/'models/cv.txt'),
                                 'result_path': str(root/directory/'training.cv.json')}
                cfg['structure_validation_contract'] = {'schema': SCHEMA, 'arm': arm,
                    'outer_folds_preserved': True, 'shared_inner_protocol': True,
                    'is_independent_confirmation': False, 'fit_augmentation_only': arm.endswith('neg20')}
                relative = Path('configs')/f'fold_{fold}'/f'{arm}.yaml'
                (staging/relative).parent.mkdir(parents=True, exist_ok=True)
                (staging/relative).write_text(yaml.safe_dump(cfg, sort_keys=False))
                jobs.append({'fold': fold, 'arm': arm, 'config': str(relative), 'result_dir': str(directory)})
            membership[str(fold)] = member_augmentation_audits(train, eligible, templates['b'])
        for item in sources.values():
            if _sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('input changed during neg20 build')
        _reference_protocol(source)
        _atomic_json(staging/'augmentation_audit.json', {**audit, 'members': membership})
        _atomic_json(staging/'protocol.json', {**SEMANTICS, 'schema': SCHEMA,
            'reference_root': str(source), 'reference_verification': verified, 'sources': sources,
            'arms': ARMS, 'feature_columns': columns, 'jobs': jobs, 'analysis': analysis,
            'contrasts': CONTRASTS, 'is_independent_confirmation': False,
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
    if protocol.get('schema') != SCHEMA or protocol.get('arms') != ARMS:
        raise ValueError('incompatible neg20 protocol')
    for path, digest in protocol['code_sha256'].items():
        if _sha256(reference.PROJECT/path) != digest:
            raise ValueError(f'implementation changed since freeze: {path}; choose a new output root')
    return verified, protocol


def validate_job(root, job):
    reference._validate_completed_job(root, job)
    result = json.loads((root/job['result_dir']/'training.cv.json').read_text())
    expected = json.loads((root/'augmentation_audit.json').read_text())['members'][str(job['fold'])]
    for member, audit in zip(result['train_fold_metrics'], expected, strict=True):
        actual = member.get('fit_augmentation')
        if job['arm'].endswith('neg20'):
            if actual != {k: v for k, v in audit.items() if k != 'member'}:
                raise ValueError('completed job did not fit the frozen augmentation rows')
        elif actual is not None:
            raise ValueError('baseline job unexpectedly used augmentation')


def paired_analysis(pooled, analysis):
    codes, groups = pd.factorize(pooled[GROUP], sort=True)
    group_labels = pd.DataFrame({'group': codes, 'label': pooled.label})
    if group_labels.groupby('group').label.nunique().gt(1).any():
        raise ValueError('paired bootstrap requires class-pure frozen groups')
    labels = group_labels.drop_duplicates('group').set_index('group').label
    strata = [labels[labels.eq(label)].index.to_numpy() for label in (0, 1)]
    if any(len(s) == 0 for s in strata):
        raise ValueError('paired bootstrap requires both classes')
    def points(weights=None):
        return {arm: _locked_vote_metrics(pooled.label, pooled[f'{arm}_fpr_5_vote'], weights) for arm in ARMS}
    def contrast(metrics, coefficients):
        return [sum(coefficients[a]*metrics[a][k] for a in coefficients) for k in ('error_recall', 'fpr')]
    observed = points()
    rng = np.random.default_rng(analysis['bootstrap_seed'])
    draws = {name: [] for name in CONTRASTS}
    for _ in range(analysis['bootstrap_reps']):
        counts = np.zeros(len(groups))
        for part in strata:
            counts += np.bincount(rng.choice(part, len(part), replace=True), minlength=len(groups))
        metrics = points(counts[codes])
        for name, coefficients in CONTRASTS.items():
            draws[name].append(contrast(metrics, coefficients))
    rows, changes = [], []
    for name, coefficients in CONTRASTS.items():
        recall, fpr = contrast(observed, coefficients)
        bounds = np.quantile(draws[name], [.025, .975], axis=0)
        primary = name == analysis['primary_comparison']
        row = {**SEMANTICS, 'comparison': name, 'role': 'primary' if primary else 'exploratory',
               'error_recall_delta_at_fpr5': recall, 'observed_fpr_delta_at_fpr5': fpr,
               'error_recall_delta_ci95_low': float(bounds[0, 0]), 'error_recall_delta_ci95_high': float(bounds[1, 0]),
               'observed_fpr_delta_ci95_low': float(bounds[0, 1]), 'observed_fpr_delta_ci95_high': float(bounds[1, 1]),
               'development_support': bool(recall >= analysis['minimum_recall_gain']
                   and fpr <= analysis['max_fpr_increase'] and bounds[0, 0] > 0
                   and bounds[1, 1] <= analysis['max_fpr_increase']) if primary else None}
        if len(coefficients) == 2:
            old = next(a for a, v in coefficients.items() if v == -1)
            new = next(a for a, v in coefficients.items() if v == 1)
            before, after = pooled[f'{old}_fpr_5_vote'].ge(.5), pooled[f'{new}_fpr_5_vote'].ge(.5)
            error = pooled.label.eq(0)
            for outcome, mask in {'recovered_fn': error & ~before & after, 'new_fn': error & before & ~after,
                                  'new_fp': ~error & ~before & after, 'recovered_fp': ~error & before & ~after}.items():
                row[outcome] = int(mask.sum())
                part = pooled.loc[mask, [SAMPLE, GROUP, OUTER, 'label', 'sequence']].copy()
                part['comparison'], part['transition'] = name, outcome
                changes.append(part)
        rows.append(row)
    # Interaction has no individual FN/FP transitions. Preserve JSON nulls
    # rather than emitting non-standard NaN values for these absent counts.
    table = pd.DataFrame(rows)
    for name in ('recovered_fn', 'new_fn', 'new_fp', 'recovered_fp'):
        table[name] = pd.Series([row.get(name) for row in rows], dtype=object)
    return table, pd.concat(changes, ignore_index=True)


def summarize_bundle(root):
    root = Path(root).resolve()
    verified, protocol = verify_bundle(root)
    for job in protocol['jobs']:
        validate_job(root, job)
    pooled = _load_pooled_predictions(root, 5, models=ARMS)
    expected = pd.read_csv(root/'outer_fold_manifest.csv')
    keys = [SAMPLE, GROUP, OUTER, 'label', 'sequence']
    if not pooled[keys].sort_values(SAMPLE).reset_index(drop=True).equals(expected[keys].sort_values(SAMPLE).reset_index(drop=True)):
        raise ValueError('pooled predictions changed the frozen q01 test population')
    models = {a: _model_metrics(pooled, a) for a in ARMS}
    comparisons, changes = paired_analysis(pooled, protocol['analysis'])
    rows = []
    for arm, model in models.items():
        point = model['operating_points']['fpr_5']['external_ensemble']['test_metrics']
        rows.append({**SEMANTICS, 'arm': arm, **{k: v for k, v in model.items() if k != 'operating_points'},
                     **{k: point[k] for k in ('tp', 'fp', 'fn', 'tn', 'n_actual_correct', 'n_actual_error')}})
    for name, frame in [('pooled_test_predictions.csv', pooled), ('summary.csv', pd.DataFrame(rows)),
                        ('paired_comparisons.csv', comparisons), ('transitions.csv', changes)]:
        for k, v in SEMANTICS.items():
            frame[k] = v
        _atomic_csv(root/name, frame)
    summary = {**SEMANTICS, 'schema': SCHEMA, 'models': models,
               'comparisons': comparisons.to_dict(orient='records'), 'analysis': protocol['analysis'],
               'bundle_verification': verified, 'is_independent_confirmation': False,
               'bootstrap_note': 'paired class-stratified frozen-group resampling, no retraining; exploratory intervals unadjusted',
               'interpretation': 'Only fitting augmented; unchanged real q01 early stopping, OOF calibration and test. '
                   'FPR names are training calibration targets; formal external decisions are member-majority votes. '
                   'Adding neg20 changes error content and quantity, not just class prevalence.'}
    lines = ['# neg20 拟合增强与 Q/D/S 配对验证', '',
             '所有测试均为原冻结 q01 样本。FPR5 是训练校准目标，实际测试 FPR 单独报告。', '',
             '| 特征/训练臂 | ROC-AUC | FP | FN | 实际 FPR | FNR |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, model in models.items():
        p = model['operating_points']['fpr_5']['external_ensemble']['test_metrics']
        lines.append(f'| {arm} | {model["roc_auc"]:.4f} | {p["fp"]} | {p["fn"]} | {p["fpr"]:.3%} | {p["fnr"]:.3%} |')
    lines += ['', '| 比较 | 错误召回变化及 95% 区间 | FPR 变化及 95% 区间 |', '| --- | --- | --- |']
    for r in comparisons.to_dict(orient='records'):
        lines.append(f'| {r["comparison"]} | {r["error_recall_delta_at_fpr5"]:.3%} '
            f'[{r["error_recall_delta_ci95_low"]:.3%}, {r["error_recall_delta_ci95_high"]:.3%}] | '
            f'{r["observed_fpr_delta_at_fpr5"]:.3%} [{r["observed_fpr_delta_ci95_low"]:.3%}, {r["observed_fpr_delta_ci95_high"]:.3%}] |')
    support = comparisons.set_index('comparison').loc[protocol['analysis']['primary_comparison'], 'development_support']
    lines += ['', '主比较：structure_neg20（增强训练下 B+QDS 对 B）。',
              '主比较达到预声明开发条件。' if support else '主比较未达到预声明开发条件。',
              'interaction 是两种训练条件下 Q/D/S 增益之差；其余比较为探索性分析。',
              '该数据已用于特征开发，不构成独立确认；增加真实 trap 的效果不能单独归因为类别比例。', '']
    (root/'report.md').write_text('\n'.join(lines))
    _atomic_json(root/'summary.json', summary)
    _atomic_json(root/'bundle_status.json', {**SEMANTICS, 'schema': SCHEMA, 'status': 'complete'})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check-reference', 'build', 'verify', 'train', 'summarize', 'run'])
    parser.add_argument('--reference-root', type=Path, default=Path('runs/spec_trainer/single-peptide-structure-validation'))
    parser.add_argument('--features', type=Path, default=Path('runs/baseline_2da_neg20_structure/features.csv'))
    parser.add_argument('--output-root', type=Path, default=Path('runs/spec_trainer/single-peptide-structure-neg20'))
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == 'check-reference':
        print(_reference_protocol(args.reference_root.resolve())[0], flush=True)
        return
    root = args.output_root.resolve()
    if args.command in ('build', 'run'):
        if args.command == 'run' and root.exists():
            _, protocol = verify_bundle(root)
            for key, path in [('neg20_features', args.features), ('reference_checksums', args.reference_root/'artifact_checksums.json')]:
                if _sha256(path) != protocol['sources'][key]['sha256']:
                    raise ValueError(f'resume input changed: {key}; choose a new output root')
            if args.bootstrap_reps != protocol['analysis']['bootstrap_reps']:
                raise ValueError('resume bootstrap settings differ from the frozen experiment')
        else:
            build_bundle(args.reference_root, args.features, root, bootstrap_reps=args.bootstrap_reps)
    if args.command == 'verify':
        print(verify_bundle(root)[0])
    if args.command in ('run', 'train'):
        reference.train_bundle(root, verifier=verify_bundle, job_validator=validate_job)
    if args.command in ('run', 'summarize'):
        summarize_bundle(root)
    print(f'{args.command} complete: {root}', flush=True)


if __name__ == '__main__':
    main()
