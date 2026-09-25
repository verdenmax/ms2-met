"""Same populations and calibration, with R x legacy-count removal only."""
import configparser
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import extract_fragment_structure as extract
from tools import fragment_reliability_ablation as ablation
from tools import fragment_structure_neg20 as neg20
from tools import fragment_structure_validation as reference
from tools.spec_trainer.src import cv_train
from tools.spec_trainer.src.cv_core import evaluate_at_threshold
from tools.spec_trainer.src.feature_cols import resolve_configured_feature_cols
from workflows.fragment_reliability import FEATURE_NAMES, STATUS_COLUMNS
from workflows.fragment_structure import FEATURE_NAMES as QDS_FEATURES
from tests.test_fragment_structure_validation import inputs, tiny_predictions
from tests.test_fragment_structure_neg20 import extended


def bundle_inputs(tmp_path, *, unavailable=False):
    paths = inputs(tmp_path)
    if unavailable:
        data = pd.read_csv(paths['features'])
        data.loc[0, list(QDS_FEATURES)] = np.nan
        data.loc[0, 'fragment_structure_valid'] = 0
        data.loc[0, 'fragment_structure_status'] = 'no_ms2_scans'
        data.to_csv(paths['features'], index=False)
    structure = reference.build_bundle(**paths, output_root=tmp_path/'structure')
    data = extended(pd.read_csv(paths['features']))
    neg_features = tmp_path/'neg20.csv'
    data.to_csv(neg_features, index=False)
    source = neg20.build_bundle(structure, neg_features, tmp_path/'neg20', bootstrap_reps=10)
    n = data.sequence.str.len()
    cuts = np.minimum(np.floor(data.ms2_structure_main_cut_fraction*(n-1)*.5),
                      np.floor(n*(1-data.ms2_structure_main_longest_gap_fraction)))
    data[FEATURE_NAMES[0]] = cuts/(n-1)
    data[FEATURE_NAMES[1]] = (n-cuts)/n
    data['fragment_reliability_valid'] = data.fragment_structure_valid
    data['fragment_reliability_version'] = 'r_v1'
    data['fragment_reliability_status'] = data.fragment_structure_status
    features = tmp_path/'reliability.csv'
    data.sample(frac=1, random_state=91).to_csv(features, index=False)
    return source, features


def test_four_arms_preserve_ids_all_splits_pool_and_old_features(tmp_path):
    source, features = bundle_inputs(tmp_path)
    root = ablation.build_bundle(source, features, tmp_path/'experiment', bootstrap_reps=10)
    _, protocol = ablation.verify_bundle(root)
    assert len(protocol['jobs']) == 20
    assert protocol['analysis']['primary_comparison'] == 'reliability_full'
    cols = protocol['feature_columns']
    assert set(cols['a1'])-set(cols['a0']) == set(FEATURE_NAMES)
    assert set(cols['a0'])-set(cols['a2']) == set(ablation.COUNT_DROPS)
    assert set(cols['a3'])-set(cols['a2']) == set(FEATURE_NAMES)
    for name in ('outer_fold_manifest.csv', 'source_outer_fold_manifest.csv'):
        assert (root/name).read_bytes() == (source/name).read_bytes()
    for fold in range(5):
        base = None
        for name in ('train_real_q01.csv', 'test_real_q01.csv', 'fit_neg20.csv'):
            path = Path('folds')/f'fold_{fold}'/name
            old, new = pd.read_csv(source/path), pd.read_csv(root/path)
            pd.testing.assert_frame_equal(old, new[list(old)], check_exact=False, atol=1e-12)
        for arm in ablation.ARMS:
            cfg = yaml.safe_load((root/f'configs/fold_{fold}/{arm}.yaml').read_text())
            assert resolve_configured_feature_cols(cfg['data'], cfg['data']['train_files'], 'label') == cols[arm]
            reference._validate_inner(pd.read_csv(cfg['data']['train_files'][0]), cfg)
            common = {key: cfg[key] for key in ('model','training','operating_point')}
            if base is not None: assert common == base
            base = common
            assert cfg['data']['fit_augmentation_files'] == [str(root/f'folds/fold_{fold}/fit_neg20.csv')]
    old = json.loads((source/'augmentation_audit.json').read_text())['members']
    new = json.loads((root/'augmentation_audit.json').read_text())['members']
    assert new == old
    with pytest.raises(FileExistsError):
        ablation.build_bundle(source, features, root)


@pytest.mark.parametrize('problem', ['missing_r','missing_row','new_pool','pool_drift','q01_drift',
    'duplicate','version','invalid_flag','availability','nan_valid','value_range','cut_increase',
    'gap_decrease','fractional_cut','relationship','fake_group'])
def test_rejects_data_or_r_drift_before_building(tmp_path, problem):
    source, features = bundle_inputs(tmp_path)
    data = pd.read_csv(features)
    q01 = data.index[data.q_value.le(.01)][0]
    extra = data.index[data.q_value.gt(.01)][0]
    if problem == 'missing_r': data = data.drop(columns=FEATURE_NAMES[0])
    elif problem == 'missing_row': data = data.drop(q01)
    elif problem == 'new_pool': data = data.drop(extra)
    elif problem == 'pool_drift': data.loc[extra, 'all_p75'] += 1
    elif problem == 'q01_drift': data.loc[q01, 'y_count'] += 1
    elif problem == 'duplicate': data = pd.concat([data, data.iloc[:1]])
    elif problem == 'version': data.loc[q01, 'fragment_reliability_version'] = 'unknown'
    elif problem == 'invalid_flag': data.loc[q01, 'fragment_reliability_valid'] = 2
    elif problem == 'availability':
        data.loc[q01, 'fragment_reliability_valid'] = 0
        data.loc[q01, 'fragment_reliability_status'] = 'missing_peak_identity'
        data.loc[q01, list(FEATURE_NAMES)] = np.nan
    elif problem == 'nan_valid': data.loc[q01, FEATURE_NAMES[0]] = np.nan
    elif problem == 'value_range': data.loc[q01, FEATURE_NAMES[1]] = np.inf
    elif problem == 'cut_increase': data.loc[q01, FEATURE_NAMES[0]] = 1
    elif problem == 'gap_decrease': data.loc[q01, FEATURE_NAMES[1]] = 0
    elif problem == 'fractional_cut': data.loc[q01, FEATURE_NAMES[0]] = .0003
    elif problem == 'relationship': data.loc[q01, 'candidate_family_id'] = 'new-unfrozen-family'
    elif problem == 'fake_group': data.loc[extra, 'peptide_group_id'] = 'false-isolation'
    data.to_csv(features, index=False)
    with pytest.raises(ValueError):
        ablation.build_bundle(source, features, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_unavailable_r_is_retained_without_changing_cohort(tmp_path):
    source, features = bundle_inputs(tmp_path, unavailable=True)
    root = ablation.build_bundle(source, features, tmp_path/'out')
    cohort = pd.read_csv(root/'cohort.csv')
    assert len(cohort) == 200
    missing = cohort.fragment_reliability_valid.eq(0)
    assert missing.any() and cohort.loc[missing, list(FEATURE_NAMES)].isna().all().all()


def test_paired_error_metrics_have_fixed_meaning_for_all_four_arms():
    pooled = tiny_predictions()
    for arm, old in [('a0','b'),('a1','b_qds'),('a2','b'),('a3','b_qds')]:
        for level in (1, 5, 10): pooled[f'{arm}_fpr_{level}_vote'] = pooled[f'{old}_fpr_{level}_vote']
    rows, changes = neg20.paired_analysis(pooled, {'bootstrap_reps':30,'bootstrap_seed':7,
        'primary_comparison':'reliability_full','minimum_recall_gain':.03,'max_fpr_increase':.005},
        arms=ablation.ARMS, contrasts=ablation.CONTRASTS)
    result = rows.set_index('comparison')
    assert result.loc['reliability_full','recovered_fn'] == 1
    assert result.loc['reliability_full','new_fn'] == 1
    assert result.loc['reliability_full','new_fp'] == 1
    assert result.loc['reliability_full','error_recall_delta_at_fpr5'] == 0
    assert result.loc['reliability_full','observed_fpr_delta_at_fpr5'] == .25
    assert result.loc['interaction','observed_fpr_delta_at_fpr5'] == 0
    point = evaluate_at_threshold([1,1,1,0,0], [.1,.9,.8,.1,.9], error_threshold=.5)
    assert (point['fp'],point['fn']) == (1,1)
    assert point['fpr'] == pytest.approx(1/3) and point['fnr'] == .5
    assert changes.label.isin([0,1]).all()


def test_complete_training_uses_same_fit_rows_and_can_resume(tmp_path, monkeypatch):
    pytest.importorskip('lightgbm')
    from tools.spec_trainer.src.models.lgb_model import LGBModel
    source, features = bundle_inputs(tmp_path)
    root = ablation.build_bundle(source, features, tmp_path/'out', bootstrap_reps=10)
    fits = []
    original_fit = LGBModel.fit
    def capture_fit(self, x, y, vx, vy):
        fits.append((set(x),len(x),int(x.all_p75.gt(1).sum()),len(vx)))
        assert vx.all_p75.le(1).all()
        return original_fit(self,x,y,vx,vy)
    monkeypatch.setattr(LGBModel,'fit',capture_fit)
    monkeypatch.setattr(reference.subprocess,'run',lambda command,**kw: cv_train.main(command[2:]))
    monkeypatch.setattr(cv_train,'_provenance',lambda args,*rest:{'config_sha256':neg20._sha256(Path(args.config))})
    reference.train_bundle(root,verifier=ablation.verify_bundle,job_validator=ablation.validate_job)
    summary = ablation.summarize_bundle(root)
    assert len(fits) == 100 and all(n_extra>0 for _,_,n_extra,_ in fits)
    for fold in range(5):
        block=fits[fold*20:(fold+1)*20]
        assert [x[1:] for x in block[:5]] == [x[1:] for x in block[5:10]] == [x[1:] for x in block[10:15]] == [x[1:] for x in block[15:20]]
    assert summary['metric_semantics'] == 'error_identification_positive_v1'
    assert set(summary['models']) == set(ablation.ARMS)
    json.dumps(summary,allow_nan=False)
    pooled = pd.read_csv(root/'pooled_test_predictions.csv')
    assert len(pooled) == 200 and pooled.experiment_sample_id.is_unique
    for job in json.loads((root/'protocol.json').read_text())['jobs']:
        oof = pd.read_csv(root/job['result_dir']/'training.cv.oof.csv')
        assert len(oof) == 160 and set(oof.negative_source) == {'real_correct_q01','real_entrapment_q01'}
    hashes = {p:neg20._sha256(p) for p in root.rglob('models/*.txt')}
    reference.train_bundle(root,verifier=ablation.verify_bundle,job_validator=ablation.validate_job)
    assert len(fits) == 100 and hashes == {p:neg20._sha256(p) for p in hashes}
    path=root/'training/fold_0/a1/training.cv.json'
    data=json.loads(path.read_text());data['train_fold_metrics'][0]['fit_augmentation']['n_actual_error']=0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='frozen augmentation'):
        ablation.validate_job(root,{'fold':0,'arm':'a1','result_dir':'training/fold_0/a1'})


def test_r_extraction_cannot_resume_a_qds_only_output(tmp_path):
    data=tmp_path/'input.json';data.write_text('[]')
    source=tmp_path/'source.ini'
    source.write_text(f'[input]\nlight_result_file={data}\n[general]\nfeature_type=0\nxic_cycle_window=6\n')
    root=tmp_path/'extraction';config,features=extract.prepare_run(source,root,reliability=True)
    cfg=configparser.ConfigParser();cfg.read(config)
    assert cfg.getboolean('general','fragment_reliability_features')
    assert cfg.getint('general','xic_cycle_window') == 6
    features.write_text('complete fixture')
    audit=json.loads((root/'structure_extraction.json').read_text())
    audit.update(status='complete',input_code_sha256=extract._extraction_fingerprints(source),
                 run_config_sha256=extract._sha256(config),features_sha256=extract._sha256(features))
    (root/'structure_extraction.json').write_text(json.dumps(audit))
    assert extract.verify_completed_run(source,root,reliability=True)['include_reliability']
    with pytest.raises(ValueError,match='frozen inputs'):
        extract.verify_completed_run(source,root,reliability=False)


def test_resume_rejects_changed_input_and_incomplete_reference(tmp_path):
    source, features=bundle_inputs(tmp_path)
    root=ablation.build_bundle(source,features,tmp_path/'out',bootstrap_reps=10)
    features.write_text(features.read_text()+'\n')
    with pytest.raises(ValueError,match='resume input changed'):
        ablation.main(['run','--reference-root',str(source),'--features',str(features),
                      '--output-root',str(root),'--bootstrap-reps','10'])
    with pytest.raises(FileNotFoundError):
        ablation.main(['check-reference','--reference-root',str(tmp_path/'missing')])
