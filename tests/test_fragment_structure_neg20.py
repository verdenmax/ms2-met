"""Frozen q01 membership and fitting-only real-error augmentation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import fragment_structure_neg20 as neg20
from tools import fragment_structure_validation as reference
from tools.spec_trainer.src import cv_train
from workflows.fragment_structure import FEATURE_NAMES
from tests.test_fragment_structure_validation import inputs, frames, tiny_predictions


def extended(extracted):
    linked = extracted.loc[extracted.label.eq(0)].iloc[::2].copy()
    linked['q_value'] = .1
    linked['rt'] += 1000
    novel = linked.iloc[:20].copy()
    novel['sequence'] = [f'NOVELTRAP{i}K' for i in range(len(novel))]
    extra = pd.concat([linked, novel], ignore_index=True)
    extra['all_p75'] = 10+np.arange(len(extra))
    return pd.concat([extracted, extra], ignore_index=True)


def bundle_inputs(tmp_path):
    paths = inputs(tmp_path)
    source = tmp_path/'reference'
    reference.build_bundle(**paths, output_root=source)
    features = tmp_path/'neg20.csv'
    extended(pd.read_csv(paths['features'])).to_csv(features, index=False)
    return source, features


def test_build_keeps_bit_identical_frozen_rows_and_separate_fit_pool(tmp_path):
    source, features = bundle_inputs(tmp_path)
    root = neg20.build_bundle(source, features, tmp_path/'experiment', bootstrap_reps=20)
    _, protocol = neg20.verify_bundle(root)
    assert len(protocol['jobs']) == 20
    audits = json.loads((root/'augmentation_audit.json').read_text())
    assert audits['n_actual_error'] == 70
    for fold in range(5):
        for role in ('train', 'test'):
            path = Path('folds')/f'fold_{fold}'/f'{role}_real_q01.csv'
            assert (root/path).read_bytes() == (source/path).read_bytes()
        train = pd.read_csv(root/f'folds/fold_{fold}/train_real_q01.csv')
        test = pd.read_csv(root/f'folds/fold_{fold}/test_real_q01.csv')
        extra = pd.read_csv(root/f'folds/fold_{fold}/fit_neg20.csv')
        assert set(extra.leakage_group_id).isdisjoint(test.leakage_group_id)
        cfg = yaml.safe_load((root/f'configs/fold_{fold}/b_neg20.yaml').read_text())
        expected = neg20.member_augmentation_audits(train, extra, cfg)
        assert expected == audits['members'][str(fold)]
        assert all(m['n_actual_error'] > 0 for m in expected)
        assert all(m['n_excluded_error'] > 0 for m in expected)
        baseline = yaml.safe_load((root/f'configs/fold_{fold}/b.yaml').read_text())
        assert 'fit_augmentation_files' not in baseline['data']
        assert cfg['training'] == baseline['training']
    with pytest.raises(FileExistsError):
        neg20.build_bundle(source, features, root)


@pytest.mark.parametrize('problem', ['missing_qds', 'old_drift', 'qds_drift', 'missing_sample',
                                   'duplicate', 'wrong_q', 'correct_q', 'status', 'inf', 'empty_pool'])
def test_rejects_incompatible_input(tmp_path, problem):
    source, features = bundle_inputs(tmp_path)
    d = pd.read_csv(features)
    if problem == 'missing_qds': d = d.drop(columns=FEATURE_NAMES[0])
    elif problem == 'old_drift': d.loc[0, 'all_p75'] += 1
    elif problem == 'qds_drift': d.loc[0, FEATURE_NAMES[0]] += 1
    elif problem == 'missing_sample': d = d.iloc[1:]
    elif problem == 'duplicate': d = pd.concat([d, d.iloc[:1]])
    elif problem == 'wrong_q': d.loc[d.index[-1], 'q_value'] = .21
    elif problem == 'correct_q': d.loc[d.label.eq(1), 'q_value'] = .1
    elif problem == 'status': d.loc[d.index[-1], 'fragment_structure_valid'] = 0
    elif problem == 'inf': d.loc[d.index[-1], FEATURE_NAMES[0]] = np.inf
    elif problem == 'empty_pool': d = d.loc[d.q_value.le(.01)]
    d.to_csv(features, index=False)
    with pytest.raises(ValueError):
        neg20.build_bundle(source, features, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_graph_bridges_are_excluded_before_cohort_filter_and_il_links_inherited():
    source, graph, extracted = frames()
    frozen, columns, _ = reference.join_frozen_features(source, graph, extracted)
    d = extended(extracted)
    # Two new rows link two old trap groups via a candidate-family token.
    # The second row fails the observed cohort but must still be a graph bridge.
    d.loc[len(extracted):len(extracted)+1, 'candidate_family_id'] = 'BRIDGE'
    d.loc[len(extracted)+1, 'heavy_in_raw'] = 0
    # An original non-model graph row exposes an I/L-equivalent new sequence.
    bridge = graph.iloc[[0]].copy()
    bridge['sequence'] = 'LLNOVELK'
    bridge[reference.SAMPLE] = 'graph-only'
    graph = pd.concat([graph, bridge], ignore_index=True)
    d.loc[d.index[-1], 'sequence'] = 'IINOVELK'
    extra, rejected, audit = neg20.prepare_neg20(frozen, graph, d, columns['b_qds'])
    assert len(rejected) == 2
    assert audit['n_rejected_bridge_rows'] == 2
    # This extra row is also linked into the bridged component and is excluded.
    assert not extra.sequence.eq('IINOVELK').any()


def test_il_equivalent_new_sequence_attaches_to_existing_group():
    source, graph, extracted = frames()
    frozen, columns, _ = reference.join_frozen_features(source, graph, extracted)
    bridge = graph.iloc[[0]].copy()
    bridge['sequence'], bridge[reference.SAMPLE] = 'LLNOVELK', 'graph-only'
    graph = pd.concat([graph, bridge], ignore_index=True)
    d = extended(extracted)
    d.loc[d.index[-1], 'sequence'] = 'IINOVELK'
    extra, rejected, _ = neg20.prepare_neg20(frozen, graph, d, columns['b_qds'])
    assert rejected.empty
    assert extra.loc[extra.sequence.eq('IINOVELK'), reference.GROUP].iloc[0] == graph.iloc[0][reference.GROUP]


def test_trainer_rejects_forged_groups_external_overlap_and_positive_augmentation(tmp_path):
    source, features = bundle_inputs(tmp_path)
    root = neg20.build_bundle(source, features, tmp_path/'out')
    train = pd.read_csv(root/'folds/fold_0/train_real_q01.csv')
    test = pd.read_csv(root/'folds/fold_0/test_real_q01.csv')
    extra = pd.read_csv(root/'folds/fold_0/fit_neg20.csv')
    cols = json.loads((root/'protocol.json').read_text())['feature_columns']['b']
    cv_train.validate_fit_augmentation(train, extra, test, cols)
    forged = extra.copy()
    forged.loc[0, reference.GROUP] = 'forged-isolation'
    with pytest.raises(ValueError, match='groups split'):
        cv_train.validate_fit_augmentation(train, forged, test, cols)
    leaked = extra.copy()
    leaked.loc[0, reference.GROUP] = test.iloc[0][reference.GROUP]
    with pytest.raises(ValueError):
        cv_train.validate_fit_augmentation(train, leaked, test, cols)
    extra.loc[0, 'label'] = 1
    with pytest.raises(ValueError, match='incorrect IDs only'):
        cv_train.validate_fit_augmentation(train, extra, test, cols)


def test_paired_contrasts_pin_error_semantics_and_interaction():
    p = tiny_predictions()
    for arm, src in [('b_neg20', 'b'), ('b_qds_neg20', 'b_qds')]:
        for target in (1, 5, 10): p[f'{arm}_fpr_{target}_vote'] = p[f'{src}_fpr_{target}_vote']
        p[f'{arm}_trust_score'] = p[f'{src}_trust_score']
    rows, changes = neg20.paired_analysis(p, {'bootstrap_reps': 30, 'bootstrap_seed': 4,
        'primary_comparison': 'structure_neg20', 'minimum_recall_gain': .03, 'max_fpr_increase': .005})
    by = rows.set_index('comparison')
    assert by.loc['structure_neg20', 'error_recall_delta_at_fpr5'] == 0
    assert by.loc['structure_neg20', 'observed_fpr_delta_at_fpr5'] == .25
    assert by.loc['structure_neg20', 'recovered_fn'] == 1
    assert by.loc['structure_neg20', 'new_fn'] == 1
    assert by.loc['structure_neg20', 'new_fp'] == 1
    assert by.loc['interaction', 'error_recall_delta_at_fpr5'] == 0
    assert by.loc['interaction', 'observed_fpr_delta_at_fpr5'] == 0
    assert by.loc['interaction', 'error_recall_delta_ci95_low'] == 0
    assert set(changes.transition) == {'recovered_fn', 'new_fn', 'new_fp'}


def test_actual_training_keeps_early_stop_and_oof_real_only_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip('lightgbm')
    from tools.spec_trainer.src.models.lgb_model import LGBModel
    source, features = bundle_inputs(tmp_path)
    root = neg20.build_bundle(source, features, tmp_path/'out', bootstrap_reps=10)
    fits = []
    original_fit = LGBModel.fit
    def capture_fit(self, x, y, vx, vy):
        fits.append((len(x), int(x.all_p75.gt(1).sum()), len(vx)))
        assert vx.all_p75.le(1).all()
        return original_fit(self, x, y, vx, vy)
    monkeypatch.setattr(LGBModel, 'fit', capture_fit)
    def in_process(command, **kwargs):
        assert command[1].endswith('cv_train.py')
        cv_train.main(command[2:])
    monkeypatch.setattr(reference.subprocess, 'run', in_process)
    def provenance(args, cfg, train, test):
        return {'config_sha256': neg20._sha256(Path(args.config))}
    monkeypatch.setattr(cv_train, '_provenance', provenance)
    reference.train_bundle(root, verifier=neg20.verify_bundle, job_validator=neg20.validate_job)
    summary = neg20.summarize_bundle(root)
    assert len(fits) == 100
    assert sum(n_extra > 0 for _, n_extra, _ in fits) == 50
    assert summary['metric_semantics'] == 'error_identification_positive_v1'
    json.dumps(summary, allow_nan=False)
    assert summary['analysis']['primary_comparison'] == 'structure_neg20'
    pooled = pd.read_csv(root/'pooled_test_predictions.csv')
    assert len(pooled) == 200 and pooled.experiment_sample_id.is_unique
    for job in json.loads((root/'protocol.json').read_text())['jobs']:
        oof = pd.read_csv(root/job['result_dir']/'training.cv.oof.csv')
        assert len(oof) == 160
        assert set(oof.negative_source) == {'real_correct_q01', 'real_entrapment_q01'}
    hashes = {p: neg20._sha256(p) for p in root.rglob('models/*.txt')}
    reference.train_bundle(root, verifier=neg20.verify_bundle, job_validator=neg20.validate_job)
    assert len(fits) == 100
    assert hashes == {p: neg20._sha256(p) for p in hashes}
    # A completed job cannot silently omit the promised augmentation.
    path = root/'training/fold_0/b_neg20/training.cv.json'
    bad = json.loads(path.read_text())
    bad['train_fold_metrics'][0]['fit_augmentation']['n_actual_error'] = 0
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='frozen augmentation'):
        neg20.validate_job(root, {'fold': 0, 'arm': 'b_neg20', 'result_dir': 'training/fold_0/b_neg20'})


def test_resume_rejects_changed_features_before_training(tmp_path):
    source, features = bundle_inputs(tmp_path)
    root = neg20.build_bundle(source, features, tmp_path/'out', bootstrap_reps=10)
    features.write_text(features.read_text()+'\n')
    with pytest.raises(ValueError, match='resume input changed'):
        neg20.main(['run', '--reference-root', str(source), '--features', str(features),
                    '--output-root', str(root), '--bootstrap-reps', '10'])


def test_preflight_rejects_incomplete_reference_and_manifest_changes(tmp_path):
    with pytest.raises(FileNotFoundError):
        neg20.main(['check-reference', '--reference-root', str(tmp_path)])
    source, features = bundle_inputs(tmp_path)
    neg20.main(['check-reference', '--reference-root', str(source)])
    path = source/'outer_fold_manifest.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='artifact changed'):
        neg20.build_bundle(source, features, tmp_path/'out')


def test_extraction_resume_checks_data_config_and_feature_hashes(tmp_path, monkeypatch):
    from tools import extract_fragment_structure as extract
    source = tmp_path/'source.ini'
    data = tmp_path/'input.json'
    data.write_text('[]')
    source.write_text(f'[input]\nlight_result_file={data}\n[general]\nfeature_type=0\n')
    root = tmp_path/'extraction'
    config, features = extract.prepare_run(source, root)
    features.write_text('fragment_structure_valid,fragment_structure_status\n1,ok\n')
    audit = json.loads((root/'structure_extraction.json').read_text())
    audit.update(status='complete', input_code_sha256=extract._extraction_fingerprints(source),
                 run_config_sha256=extract._sha256(config), features_sha256=extract._sha256(features))
    (root/'structure_extraction.json').write_text(json.dumps(audit))
    assert extract.verify_completed_run(source, root)['status'] == 'complete'
    data.write_text('[1]')
    with pytest.raises(ValueError, match='frozen inputs'):
        extract.verify_completed_run(source, root)
