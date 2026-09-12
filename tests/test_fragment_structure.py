"""Physical counterexamples for candidate-local Q/D/S evidence."""
import configparser
from dataclasses import replace

import numpy as np
import pytest

from spectrum.dia_data import DIAData, XIC_DTYPE, pool_fragment_charges
from spectrum.spectrum_utils import match_peak_targets_ppm
from workflows.fragment_structure import (
    CHARGE_FEATURES, DEDUP_FEATURES, STRUCTURE_FEATURES, FEATURE_NAMES,
    FragmentEvidence, fragment_structure_features,
)
from workflows.q1a_helpers import is_signal_present_heavy


DTYPE = np.dtype(XIC_DTYPE.descr + [('peak_ids', 'O')])
PEAK = [0, 200, 800, 200, 0, 0, 0]


def trace(values=PEAK, *, peak_id=1, heavy=False):
    x = np.empty(len(values), dtype=DTYPE)
    x['rt'] = 10 + np.arange(len(values))*.04 + (.003 if heavy else 0)
    x['intensity'] = values; x['ppm_error'] = 0
    x['cycle_idx'] = np.arange(len(values)) + 100
    for j,v in enumerate(values):
        x['peak_ids'][j] = (j*2+int(heavy), (peak_id,) if v>0 else ())
    return x


def fragment(ion_type='y', ordinal=3, *, peak_id=1, values=PEAK, shifted=True):
    zero = np.zeros(len(values))
    return FragmentEvidence(ion_type,ordinal,500,508 if shifted else 500,
        {1:trace(values,peak_id=peak_id),2:trace(zero)},
        {1:trace(values,peak_id=peak_id,heavy=True),2:trace(zero,heavy=True)})


def features(records, sequence='AAKAAAAK', charge=2, **kwargs):
    return fragment_structure_features(sequence,charge,records,
        split_window=True,center_rt=10.08,**kwargs)


def test_cross_charge_only_signal_is_not_a_same_charge_pair():
    f = fragment()
    f = replace(f,light={1:f.light[2],2:f.light[1]})
    assert is_signal_present_heavy(pool_fragment_charges(f.light),pool_fragment_charges(f.heavy))
    out = features([f])
    assert out['fragment_structure_valid']==1
    assert out['ms2_charge_paired_target_count']==0
    assert out['ms2_charge_pooled_only_fraction']==1
    assert out['ms2_charge_dominant_mismatch_fraction']==1
    assert out['ms2_structure_main_trace_count']==0
    assert np.isnan(out['ms2_charge_z1_pearson_median'])


def test_precursor_charge_one_cannot_gain_support_from_charge_two():
    f = fragment()
    f = replace(f,light={1:f.light[2],2:f.light[1]},heavy={1:f.heavy[2],2:f.heavy[1]})
    assert features([f],charge=2)['ms2_charge_paired_target_count']==1
    out=features([f],charge=1)
    assert out['ms2_charge_paired_target_count']==0
    assert np.isnan(out['ms2_charge_z2_pearson_median'])


def test_exact_peak_reuse_does_not_create_two_sequence_cuts():
    a,b=fragment('b',2,shifted=False),fragment('b',4,shifted=False)
    out=features([a,b])
    assert out['ms2_charge_paired_target_count']==2
    assert out['ms2_dedup_paired_trace_count']==1
    assert out['ms2_dedup_reused_target_fraction']==.5
    assert out['ms2_dedup_reused_intensity_fraction']==.5
    assert out['ms2_structure_main_ambiguous_cut_fraction']==1
    assert out['ms2_structure_main_cut_fraction']==0


def test_equal_intensities_and_equal_masses_are_not_peak_identity():
    a,b=fragment('b',2,peak_id=1),fragment('b',4,peak_id=2)
    # The helper intentionally supplies equal masses and intensities here.
    out=features([a,b])
    assert out['ms2_dedup_paired_trace_count']==2
    assert out['ms2_dedup_reused_target_fraction']==0
    assert out['ms2_structure_main_cut_fraction']==pytest.approx(2/7)


def test_same_peak_called_b_and_y_is_not_complementary_support():
    out=features([fragment('b',3),fragment('y',5)])
    assert out['ms2_structure_main_cut_fraction']==pytest.approx(1/7)
    assert out['ms2_structure_main_complementary_cut_fraction']==0


def test_independent_b_y_evidence_brackets_internal_label_site():
    out=features([fragment('b',2,peak_id=1),fragment('b',3,peak_id=2),
                  fragment('y',5,peak_id=3)])
    assert out['ms2_structure_main_complementary_cut_fraction']==pytest.approx(1/7)
    assert out['ms2_structure_main_internal_kr_bracket_fraction']==1
    assert out['ms2_structure_main_b_run_fraction']==pytest.approx(2/7)
    assert out['ms2_structure_main_y_run_fraction']==pytest.approx(1/7)
    assert np.isnan(features([fragment()],silac=False)['ms2_structure_main_internal_kr_bracket_fraction'])


def test_peak_group_must_share_both_channels_and_not_chain_neighbors():
    records=[fragment('y',i+2,peak_id=i+1,values=v) for i,v in enumerate([
        [0,800,0,0,0,0,0], [0,0,800,0,0,0,0], [0,0,0,800,0,0,0]])]
    out=features(records)
    # One-cycle boundary tolerance may group adjacent peaks, never all three.
    assert out['ms2_structure_main_trace_count']==2
    assert out['ms2_structure_main_trace_fraction']==pytest.approx(2/3)
    a,b=fragment('y',2,peak_id=1),fragment('y',3,peak_id=2)
    b=replace(b,heavy={1:trace([0,0,0,0,800,200,0],heavy=True),2:b.heavy[2]})
    assert features([a,b])['ms2_charge_paired_target_count']==1


def test_broad_peak_keeps_sequence_support_despite_apex_differences():
    records=[fragment('y',i+2,peak_id=i+1,values=v) for i,v in enumerate([
        [0,800,750,700,600,0,0], [0,600,700,750,800,0,0]])]
    assert features(records)['ms2_structure_main_trace_count']==2


def test_ordinal_one_background_cannot_locate_long_sequence():
    out=features([fragment('b',1),fragment('y',1,peak_id=2)])
    assert out['ms2_charge_paired_ion_count']==2
    assert out['ms2_structure_main_cut_fraction']==0
    assert out['ms2_structure_main_longest_gap_fraction']==1
    assert out['ms2_structure_main_trace_count']==0


def test_empty_observed_signal_differs_from_missing_acquisition():
    f=fragment(values=np.zeros(7))
    out=features([f])
    assert out['fragment_structure_valid']==1
    assert out['ms2_charge_paired_target_count']==0
    assert np.isnan(out['ms2_charge_paired_fraction'])
    f=replace(f,heavy={1:np.empty(0,dtype=DTYPE),2:np.empty(0,dtype=DTYPE)})
    out=features([f])
    assert out['fragment_structure_valid']==0
    assert all(np.isnan(out[k]) for k in FEATURE_NAMES)


def test_unshifted_coisolated_signal_is_not_independent_evidence():
    out=fragment_structure_features('AAAAK',2,[fragment('b',2,shifted=False)],
        split_window=False,center_rt=10.08)
    assert out['fragment_structure_valid']==0


def test_centroid_ids_preserve_unsorted_original_indices_and_overlap():
    mz=np.array([500.,200.001,200.0,200.002]);y=np.array([8.,40.,90.,0.])
    old=match_peak_targets_ppm(mz,y,np.array([200.,200.0001]),10)
    new=match_peak_targets_ppm(mz,y,np.array([200.,200.0001]),10,return_peak_indices=True)
    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
    assert new[2]==[(1,2),(1,2)]


def test_panel_peak_metadata_does_not_change_legacy_pooling():
    d=DIAData.__new__(DIAData)
    d.rt_values=np.array([10.,10.04,10.08])
    d._select_ms2_xic_indices=lambda *args:[0,1,2]
    d._ms2_cycle_idx=lambda i:i
    d.get_spectrum_by_index=lambda i:(np.array([201.00727646677,101.00727646677]),np.array([200.,800.]))
    panel,_=d.xic_ms2_fragment_panel_extract(10,1,500,[200,400],10,include_peak_ids=True)
    assert panel[0][1]['peak_ids'][0]==(0,(0,))
    assert panel[0][2]['peak_ids'][0]==(0,(1,))
    legacy,_=d.xic_ms2_peaks_extract(10,1,500,200,10)
    np.testing.assert_array_equal(pool_fragment_charges(panel[0]),legacy)


def test_registry_matches_extractor_and_keeps_historical_arms_unchanged():
    from tools.spec_trainer.src.feature_groups import (
        MS2_CHARGE_FEATURES,MS2_DEDUP_FEATURES,MS2_STRUCTURE_FEATURES,
        experiment_arm_features,ELIGIBILITY_FEATURES,METADATA_COLUMNS)
    assert set(CHARGE_FEATURES)==MS2_CHARGE_FEATURES
    assert set(DEDUP_FEATURES)==MS2_DEDUP_FEATURES
    assert set(STRUCTURE_FEATURES)==MS2_STRUCTURE_FEATURES
    old=experiment_arm_features('ms1_ms2_no_prediction')
    assert not old.intersection(FEATURE_NAMES)
    assert experiment_arm_features('ms1_ms2_qds')==old.union(FEATURE_NAMES)
    assert 'fragment_structure_valid' in ELIGIBILITY_FEATURES
    assert {'fragment_structure_status','fragment_structure_version'} <= METADATA_COLUMNS


def test_single_candidate_pipeline_reuses_scans_and_preserves_all_old_values():
    from spectrum.psm_info import PSMInfo
    from spectrum.labeling import HeavyType
    from workflows.single_work import single_pair_work
    p=PSMInfo(sequence='AAKAAK',charge=2,modify=[],rt=np.float32(10.08),
              precursor_mz=np.float32(500),raw_title='toy',protein_names='')
    hprec,fs=p.get_heavy_info(HeavyType.SILAC)
    dia=DIAData.__new__(DIAData)
    dia.rt_values=np.array([10+j*.04+k*.003 for j in range(7) for k in range(3)])
    dia._select_ms2_xic_indices=lambda rt,rad,mz:[3*j+(1 if mz==p._precursor_mz else 2) for j in range(7)]
    dia._ms2_cycle_idx=lambda i:i//3
    dia.check_in_raw=lambda mz:True
    dia.check_in_same_ms2=lambda *a,**kw:False
    dia.get_window_info=lambda mz,rt=None:{'width':2.,'centering':.5,'lower':float(mz)-1,'upper':float(mz)+1}
    dia.xic_peaks_extreact=lambda *a,**kw:trace()
    dia.xic_peaks_panel_extract=lambda *a,**kw:trace()
    calls=[]
    def spectrum(i):
        calls.append(i)
        side=2 if i%3==1 else 3
        mz=np.unique([f[side]/z+1.00727646677 for f in fs for z in (1,2)])
        return mz,np.full(len(mz),PEAK[i//3],dtype='f8')
    dia.get_spectrum_by_index=spectrum
    cfg=configparser.ConfigParser();cfg['general']={'mass_tol_ppm':'10','xic_cycle_window':'3','fragment_structure_features':'false'}
    old=single_pair_work(p,dia,cfg)
    old_calls=len(calls);calls.clear()
    cfg.set('general','fragment_structure_features','true')
    new=single_pair_work(p,dia,cfg)
    assert len(calls)==14 and old_calls>len(calls)
    assert new['fragment_structure_valid']==1
    additions=set(FEATURE_NAMES)|{'fragment_structure_valid','fragment_structure_status','fragment_structure_version'}
    assert set(old)==set(new)
    for name,value in old.items():
        if name in additions:continue
        if isinstance(value,(float,np.floating)) and np.isnan(value):assert np.isnan(new[name]),name
        else:assert value==new[name],name


def test_full_extraction_preparation_preserves_inputs_and_is_non_destructive(tmp_path):
    from tools.extract_fragment_structure import prepare_run
    source=tmp_path/'source.ini'
    source.write_text('[input]\nraw_num=1\nraw_path_1=/data/a.pfb\nlight_result_file=/data/full.json\nsearch_engine_type=0\n[general]\nfeature_type=0\nmass_tol_ppm=10\nxic_cycle_window=6\nresult_file=/data/old.csv\n')
    original=source.read_bytes();output=tmp_path/'new'
    path,result=prepare_run(source,output)
    c=configparser.ConfigParser();c.read(path)
    assert c['input']['light_result_file']=='/data/full.json'
    assert c['general']['xic_cycle_window']=='6'
    assert c.getboolean('general','fragment_structure_features')
    assert source.read_bytes()==original and not result.exists()
    assert prepare_run(source,output)==(path,result)
    result.write_text('completed')
    with pytest.raises(FileExistsError,match='Completed output'):
        prepare_run(source,output)
    assert result.read_text()=='completed'
