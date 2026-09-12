"""Re-extract audited PFB cases with Q/D/S on and off; check legacy parity.

This verifies implementation against concrete spectra, not classification
utility. No training, label changes or decision threshold calibration.
"""
import configparser
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from manager.data_manager import DataManager
from spectrum.psm_info import PSMInfo
from workflows.single_work import single_pair_work
from workflows.fragment_structure import FEATURE_NAMES

HERE=Path(__file__).resolve().parent
DATA=Path('/home/verden/share/2026_04_27_kongweisa_diann_ZHOUHUdataset')
CASES=['FN007','FN018','FN033','FN091','FN097','FN102','FN129','C007','C091','C097']


def main():
    config_path=DATA/'feature_result/ms2-met-runs-08-20/baseline_2da_clean/config.ini'
    cfg=configparser.ConfigParser();cfg.read(config_path)
    rows=pd.read_csv(HERE/'all_fn_review/selected_cases.csv')
    selected=rows[rows.case_id.isin(CASES)]
    assert len(selected)==len(CASES)
    records=[];new_columns=set(FEATURE_NAMES)|{'fragment_structure_valid','fragment_structure_status','fragment_structure_version'}
    n_checked=0
    for raw, group in selected.groupby('raw_title1'):
        print('PFB',raw,'cases',','.join(group.case_id),flush=True)
        dia=DataManager(cfg).get_dia_data_object(str(DATA/'2th'/f'{raw}.pfb'))
        for row in group.itertuples():
            with gzip.open(HERE/'all_fn_review/traces'/f'{row.case_id}.json.gz','rt') as f:
                rec=json.load(f)
            p=PSMInfo.from_dict(rec['psm'])
            cfg.set('general','fragment_structure_features','false')
            legacy=single_pair_work(p,dia,cfg)
            cfg.set('general','fragment_structure_features','true')
            current=single_pair_work(p,dia,cfg)
            assert set(legacy)==set(current)
            for name,a in legacy.items():
                if name in new_columns:continue
                b=current[name]
                if isinstance(a,(float,np.floating)) and np.isnan(a):
                    assert np.isnan(b),(row.case_id,name,a,b)
                else:assert a==b,(row.case_id,name,a,b)
                n_checked+=1
            assert current['y_count']==row.y_count,(row.case_id,'frozen y_count')
            assert current['q1a_TP_count']==row.q1a_TP_count,(row.case_id,'frozen Q1A count')
            records.append({'case_id':row.case_id,'sequence':p._sequence,
                            'legacy_y_count':current['y_count'],'legacy_q1a_paired_count':current['q1a_TP_count'],
                            **{k:current[k] for k in sorted(new_columns)}})
            print(row.case_id,current['fragment_structure_status'],
                  'targets',current['ms2_charge_paired_target_count'],
                  'dedup',current['ms2_dedup_paired_trace_count'],
                  'cuts',current['ms2_structure_main_cut_fraction'],flush=True)
        del dia;gc.collect()
    out=HERE/'structure_implementation_check';out.mkdir(exist_ok=True)
    pd.DataFrame(records).sort_values('case_id').to_csv(out/'cases.csv',index=False)
    files=['spectrum/spectrum_utils.py','spectrum/dia_data.py','workflows/single_work.py',
           'workflows/q1a_helpers.py','workflows/fragment_structure.py','tools/spec_trainer/src/feature_groups.py']
    (out/'validation.json').write_text(json.dumps({
        'purpose':'implementation_canary_not_model_evaluation','cases':CASES,
        'n_legacy_field_values_checked':n_checked,'all_legacy_values_exactly_equal':True,
        'frozen_y_count_and_q1a_counts_equal':True,'n_new_numeric_features':len(FEATURE_NAMES),
        'sources_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
    },indent=2)+'\n')
    print('All canary cases passed',n_checked,'legacy values checked',flush=True)


if __name__=='__main__':main()
