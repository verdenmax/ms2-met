"""Run ordinary full-cohort extraction with additive Q/D/S features.

Uses the source config's original input JSON and extraction settings, writes
to a separate run directory, and never substitutes the 130 reviewed FNs for
the full input population. Existing completed features are not overwritten.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def _path(value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT/p).resolve()


def prepare_run(config_path: Path, output_dir: Path) -> tuple[Path,Path]:
    config_path,output_dir=_path(config_path),_path(output_dir)
    config=configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f'Cannot read extraction config: {config_path}')
    if not config.has_section('input') or not config.has_section('general'):
        raise ValueError('Extraction config requires [input] and [general]')
    if config.getint('general','feature_type',fallback=0)!=0:
        raise ValueError('Q/D/S extraction requires single-candidate feature_type=0')
    if (output_dir/'features.csv').exists():
        raise FileExistsError(f'Completed output already exists: {output_dir / "features.csv"}; choose a new output directory')
    result=output_dir/'features.csv'
    run_config=output_dir/'config.ini'
    if run_config==config_path or result==_path(config.get('general','result_file',fallback='runs/features.csv')):
        raise ValueError('Q/D/S run must use a separate output directory')
    config.set('general','feature_type','0')
    config.set('general','fragment_structure_features','true')
    config.set('general','result_file',str(result))
    config.set('general','work_directory',str(output_dir/'workspace'))
    output_dir.mkdir(parents=True,exist_ok=True)
    if run_config.exists():
        existing=configparser.ConfigParser();existing.read(run_config)
        if dict(existing)!=dict(config):
            raise FileExistsError(f'Run config differs: {run_config}; choose a new output directory')
    else:
        with run_config.open('x',encoding='utf-8') as f:
            config.write(f)
    (output_dir/'structure_extraction.json').write_text(json.dumps({
        'schema':'fragment_structure_extraction_v1','status':'prepared',
        'source_config':str(config_path),
        'source_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'run_config':str(run_config),'features':str(result),
        'input_selection':'unchanged source input and existing postfilters; no Q/D/S quality filter',
        'relative_input_paths_base':str(PROJECT),
        'note':'Feature extraction only; frozen experiment groups must be joined before ablation training.',
    },indent=2)+'\n')
    return run_config,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('runs/baseline_2da_clean/config.ini'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/baseline_2da_structure'))
    p.add_argument('--prepare-only',action='store_true',help='Write reviewable run config without starting extraction')
    args=p.parse_args()
    config,result=prepare_run(args.config,args.output_dir)
    print(f'Q/D/S config: {config}\nOutput: {result}',flush=True)
    if args.prepare_only:
        return
    audit_path=result.parent/'structure_extraction.json'
    audit=json.loads(audit_path.read_text());audit['status']='running'
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    try:
        subprocess.run([sys.executable,str(PROJECT/'main.py'),
                        '--configpath',str(config),'--logpath',str(result.parent/'extract.log')],
                       cwd=PROJECT,check=True)
        import pandas as pd
        rows=pd.read_csv(result,usecols=['fragment_structure_valid','fragment_structure_status'])
        audit.update(status='complete',n_rows=len(rows),
                     n_structure_available=int(rows.fragment_structure_valid.eq(1).sum()),
                     structure_status_counts=rows.fragment_structure_status.value_counts(dropna=False).to_dict())
    except Exception as exc:
        audit.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
