"""Frozen membership, group isolation and locked-decision validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import fragment_structure_validation as validation
from tools.spec_trainer.src.cohort import COHORT_DEFINITIONS
from tools.spec_trainer.src.cv_core import evaluate_at_threshold
from tools.spec_trainer.src.feature_cols import resolve_configured_feature_cols
from tools.spec_trainer.src.feature_groups import experiment_arm_features
from workflows.fragment_structure import FEATURE_NAMES


def frames():
    rng = np.random.default_rng(31)
    base = sorted(experiment_arm_features('ms1_ms2_no_prediction'))
    rows = []
    for label in (0, 1):
        for group in range(50):
            for rep in range(2):
                rows.append({**dict(zip(base, rng.uniform(.1, .9, len(base)))),
                    **dict(COHORT_DEFINITIONS['evidence_observed']),
                    'sequence': f'{"CORRECT" if label else "TRAP"}{group}K',
                    'charge': 2, 'precursor_mz': 500., 'rt': float(20+group),
                    'raw_title1': f'raw_rep{rep}', 'label': label,
                    'label_type': 'positive' if label else 'negative', 'q_value': .005})
    source = pd.DataFrame(rows)
    identity = validation._real_with_identity(source)
    manifest = identity[[validation.SAMPLE, validation.SOURCE_ROW, 'sequence', 'label',
                         'experiment_origin', 'peptide_group_id']].copy()
    manifest[validation.GROUP] = manifest.peptide_group_id
    manifest[validation.OUTER] = np.tile(np.repeat(np.arange(50) % 5, 2), 2)
    extracted = source.copy()
    for column in FEATURE_NAMES:
        extracted[column] = rng.uniform(.1, .9, len(source))
    extracted['fragment_structure_valid'] = 1
    extracted['fragment_structure_version'] = 'qds_v1'
    extracted['fragment_structure_status'] = 'ok'
    return source, manifest, extracted


def inputs(tmp_path):
    source, manifest, extracted = frames()
    paths = {}
    for name, frame in [('source_features', source), ('manifest', manifest), ('features', extracted)]:
        path = tmp_path/f'{name}.csv'
        frame.to_csv(path, index=False)
        paths[name] = path
    template = tmp_path/'template.yaml'
    template.write_text(yaml.safe_dump({
        'data': {}, 'model': {'type': 'lightgbm', 'params': {
            'objective': 'binary', 'metric': ['auc', 'binary_logloss'], 'num_threads': 1,
            'num_leaves': 3, 'min_data_in_leaf': 5, 'seed': 42, 'verbosity': -1}},
        'training': {'cv_folds': 5, 'cv_seed': 42, 'valid_size': .2,
                     'num_boost_round': 3, 'early_stopping_rounds': 2,
                     'min_class_groups_per_split': 1},
    }))
    config = tmp_path/'design.yaml'
    config.write_text(yaml.safe_dump({'schema': validation.SCHEMA,
        'training_template': 'template.yaml', 'analysis': {
            'bootstrap_reps': 20, 'bootstrap_seed': 12}}))
    paths['config'] = config
    return paths


def test_join_uses_identity_despite_reordering_and_keeps_missing_qds():
    source, manifest, extracted = frames()
    extracted.loc[0, list(FEATURE_NAMES)] = np.nan
    extracted.loc[0, 'fragment_structure_valid'] = 0
    extracted.loc[0, 'fragment_structure_status'] = 'no_ms2_scans'
    shuffled = extracted.sample(frac=1, random_state=2)
    result, columns, audit = validation.join_frozen_features(source, manifest, shuffled)
    assert len(result) == len(manifest)
    assert result[validation.SAMPLE].tolist() == manifest[validation.SAMPLE].tolist()
    assert result[validation.GROUP].tolist() == manifest[validation.GROUP].tolist()
    assert result[validation.OUTER].tolist() == manifest[validation.OUTER].tolist()
    assert result.loc[0, list(FEATURE_NAMES)].isna().all()
    assert audit['n_structure_available'] == len(source)-1
    assert audit['qds_quality_filter_applied'] is False
    assert set(columns['b_qds']) == set(columns['b']) | set(FEATURE_NAMES)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'label', 'drift', 'source_index',
                                   'group_split', 'sequence_split', 'fold_fraction', 'unknown_id',
                                   'all_unavailable', 'status', 'inf', 'missing_column'])
def test_join_rejects_unsafe_inputs(problem):
    source, manifest, extracted = frames()
    if problem == 'missing': extracted = extracted.iloc[1:]
    elif problem == 'duplicate': extracted = pd.concat([extracted, extracted.iloc[:1]])
    elif problem == 'label': extracted.loc[0, ['label', 'label_type']] = [1, 'positive']
    elif problem == 'drift': extracted.loc[0, 'precursor_pearson'] += .2
    elif problem == 'source_index': manifest.loc[0, validation.SOURCE_ROW] = -1
    elif problem == 'group_split': manifest.loc[0, validation.OUTER] = 4
    elif problem == 'sequence_split': manifest.loc[0, validation.GROUP] = 'other-group'
    elif problem == 'fold_fraction':
        manifest[validation.OUTER] = manifest[validation.OUTER].astype(float)
        manifest.loc[0, validation.OUTER] = 1.5
    elif problem == 'unknown_id': manifest.loc[0, validation.SAMPLE] = 'unrelated'
    elif problem == 'all_unavailable':
        extracted['fragment_structure_valid'] = 0
        extracted['fragment_structure_status'] = 'disabled'
        extracted[list(FEATURE_NAMES)] = np.nan
    elif problem == 'status': extracted.loc[0, 'fragment_structure_status'] = 'disabled'
    elif problem == 'inf': extracted.loc[0, FEATURE_NAMES[0]] = np.inf
    elif problem == 'missing_column': extracted = extracted.drop(columns=FEATURE_NAMES[0])
    with pytest.raises((ValueError, TypeError)):
        validation.join_frozen_features(source, manifest, extracted)


def test_bundle_freezes_shared_partitions_and_strict_feature_arms(tmp_path):
    paths = inputs(tmp_path)
    root = tmp_path/'bundle'
    validation.build_bundle(**paths, output_root=root)
    _, protocol = validation.verify_bundle(root)
    assert len(protocol['jobs']) == 25
    for fold in range(5):
        train = pd.read_csv(root/'folds'/f'fold_{fold}'/'train_real_q01.csv')
        test = pd.read_csv(root/'folds'/f'fold_{fold}'/'test_real_q01.csv')
        assert set(train[validation.GROUP]).isdisjoint(test[validation.GROUP])
        previous = None
        for arm in validation.ARMS:
            cfg = yaml.safe_load((root/'configs'/f'fold_{fold}'/f'{arm}.yaml').read_text())
            selected = resolve_configured_feature_cols(cfg['data'], cfg['data']['train_files'], 'label')
            assert selected == protocol['feature_columns'][arm]
            assert cfg['data']['predefined_cv_fold_col'] == 'experiment_inner_fold'
            assert cfg['data']['require_complete_arm'] is True
            validation._validate_inner(train, cfg)
            if previous:
                assert cfg['data']['train_files'] == previous['data']['train_files']
                assert cfg['data']['test_files'] == previous['data']['test_files']
                assert cfg['training'] == previous['training']
            previous = cfg
    with pytest.raises(FileExistsError):
        validation.build_bundle(**paths, output_root=root)
    cohort = root/'cohort.csv'
    cohort.write_text(cohort.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'):
        validation.verify_bundle(root)


def tiny_predictions():
    # Correct: B falsely flags row 0; QDS falsely flags rows 0 and 1.
    # Errors: B misses rows 4 and 5; QDS recovers 4 but newly misses 7.
    p = pd.DataFrame({'label': [1, 1, 1, 1, 0, 0, 0, 0],
        validation.SAMPLE: [f's{i}' for i in range(8)],
        validation.GROUP: [f'g{i//2}' for i in range(8)],
        validation.OUTER: [0]*8, 'sequence': [f'PEP{i}' for i in range(8)]})
    for arm in validation.ARMS:
        vote = [.8, 0, 0, 0, 0, 0, .8, .8] if arm == 'b' else [.8, .8, 0, 0, .8, 0, .8, 0]
        for target in (1, 5, 10): p[f'{arm}_fpr_{target}_vote'] = vote
        p[f'{arm}_trust_score'] = 1-np.asarray(vote)
    return p


def test_error_positive_values_and_both_directions_of_fn_fp_transfers():
    p = tiny_predictions()
    metric = validation._model_metrics(p, 'b')['operating_points']['fpr_5']['external_ensemble']['test_metrics']
    assert (metric['tp'], metric['fp'], metric['fn'], metric['tn']) == (2, 1, 2, 3)
    assert metric['fpr'] == .25 and metric['fnr'] == .5
    comparisons, changes = validation.paired_comparisons(p, {
        'bootstrap_reps': 50, 'bootstrap_seed': 3, 'minimum_recall_gain': .03, 'max_fpr_increase': .005})
    main = comparisons.set_index('arm').loc['b_qds']
    assert main['recovered_fn'] == 1 and main['new_fn'] == 1 and main['net_recovered_fn'] == 0
    assert main['new_fp'] == 1 and main['recovered_fp'] == 0
    assert main['error_recall_delta_at_fpr5'] == 0
    assert main['observed_fpr_delta_at_fpr5'] == .25
    assert not main['development_support']
    assert comparisons.loc[comparisons.arm.ne('b_qds'), 'development_support'].isna().all()
    assert len(changes) == 4*len(p)


def test_bootstrap_applies_one_weight_to_entire_group(monkeypatch):
    original = validation._locked_vote_metrics
    weights_seen = []
    def check(labels, votes, weights=None):
        if weights is not None:
            weights_seen.append(weights)
            assert np.array_equal(weights[::2], weights[1::2])
        return original(labels, votes, weights)
    monkeypatch.setattr(validation, '_locked_vote_metrics', check)
    validation.paired_comparisons(tiny_predictions(), {
        'bootstrap_reps': 10, 'bootstrap_seed': 3, 'minimum_recall_gain': .03, 'max_fpr_increase': .005})
    assert len(weights_seen) == 50


def test_failed_job_releases_lock_and_existing_lock_blocks_training(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    root = tmp_path/'bundle'
    validation.build_bundle(**paths, output_root=root)
    def fail(*args, **kwargs): raise RuntimeError('training stopped')
    monkeypatch.setattr(validation.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='training stopped'):
        validation.train_bundle(root)
    assert not (root/'.train.lock').exists()
    (root/'.train.lock').write_text('1')
    with pytest.raises(RuntimeError, match='training lock exists'):
        validation.train_bundle(root)


def test_resume_checks_template_and_inputs_before_training(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    root = tmp_path/'bundle'
    validation.build_bundle(**paths, output_root=root)
    actions = []
    monkeypatch.setattr(validation, 'train_bundle', lambda p: actions.append('train'))
    monkeypatch.setattr(validation, 'summarize_bundle', lambda p: actions.append('summary'))
    arguments = ['run', '--output-root', str(root)]
    for key, value in paths.items(): arguments.extend(['--'+key.replace('_', '-'), str(value)])
    validation.main(arguments)
    assert actions == ['train', 'summary']
    actions.clear()
    template = tmp_path/'template.yaml'
    template.write_text(template.read_text()+'\n# changed recipe\n')
    with pytest.raises(ValueError, match='resume input changed: training_template'):
        validation.main(arguments)
    assert actions == []


def test_full_formal_trainer_and_summary_smoke(tmp_path, monkeypatch):
    pytest.importorskip('lightgbm')
    from tools.spec_trainer.src import cv_train
    paths = inputs(tmp_path)
    root = tmp_path/'bundle'
    validation.build_bundle(**paths, output_root=root)
    # Same production entry and 125 models; avoid 25 Python startup costs.
    def in_process(command, **kwargs):
        assert command[1].endswith('cv_train.py')
        cv_train.main(command[2:])
    monkeypatch.setattr(validation.subprocess, 'run', in_process)
    # cv_train provenance also uses subprocess.run; provide a focused shim.
    import hashlib
    def provenance(args, cfg, train_files, test_files):
        return {'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
    monkeypatch.setattr(cv_train, '_provenance', provenance)
    validation.train_bundle(root)
    summary = validation.summarize_bundle(root)
    assert summary['is_independent_confirmation'] is False
    assert len(summary['models']) == 5
    assert summary['primary_comparison'] == ['b', 'b_qds']
    assert (root/'report.md').is_file() and (root/'group_transitions.csv').is_file()
    scores = pd.read_csv(root/'pooled_test_predictions.csv')
    assert len(scores) == 200
    assert scores[validation.SAMPLE].is_unique
    for arm in validation.ARMS:
        p = summary['models'][arm]['operating_points']['fpr_5']['external_ensemble']
        expected = evaluate_at_threshold(scores.label, 1-scores[f'{arm}_fpr_5_vote'], .5)
        assert p['test_metrics'] == expected
    monkeypatch.setattr(validation.subprocess, 'run', lambda *a, **k: pytest.fail('completed jobs must be skipped'))
    validation.train_bundle(root)
    path = root/'training/fold_0/b/training.cv.test_scores.csv'
    bad = pd.read_csv(path)
    bad.loc[0, 'member_0_trust_score'] = 0
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError):
        validation.summarize_bundle(root)
