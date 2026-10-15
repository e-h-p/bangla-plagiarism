"""Reproduce deterministic analyses in a new directory; never run paid APIs.

The dataset is obtained separately from DOI 10.17632/ws5c8ppx5m.1.
Existing input archives and existing output directories are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare_reference_results(output):
    import numpy as np
    import pandas as pd
    checked=[]
    for ref in sorted((ROOT/'reference_results').rglob('*.csv')):
        relative=ref.relative_to(ROOT/'reference_results')
        generated=output/relative
        if not generated.exists():
            raise RuntimeError(f'Missing generated result: {relative}')
        expected=pd.read_csv(ref);actual=pd.read_csv(generated)
        if set(expected.columns)!=set(actual.columns):
            raise RuntimeError(f'Changed result columns: {relative}')
        order=[c for c in ['Metric','Threshold'] if c in expected]
        if order:
            expected=expected.sort_values(order).reset_index(drop=True)
            actual=actual.sort_values(order).reset_index(drop=True)
        if len(expected)!=len(actual):raise RuntimeError(f'Changed row count: {relative}')
        for col in expected:
            if pd.api.types.is_numeric_dtype(expected[col]):
                tol=0 if col in ['TP','FP','TN','FN'] else 1.01e-6
                if not np.allclose(expected[col],actual[col],atol=tol,rtol=0,equal_nan=True):
                    raise RuntimeError(f'Numeric result mismatch: {relative}/{col}')
            elif not expected[col].fillna('').equals(actual[col].fillna('')):
                raise RuntimeError(f'Result label mismatch: {relative}/{col}')
        checked.append({'file':relative.as_posix(),'rows':len(actual),'passed':True})
    if not checked:raise RuntimeError('No reference result files were found')
    return checked

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path,help='Unmodified Version 1 dataset .xlsx')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs',help='New output directory (must not exist)')
    parser.add_argument('--regenerate-banglabert',action='store_true',help='Opt in to fresh encoder inference; requires requirements-models.txt and model downloads')
    parser.add_argument('--figures',action='store_true',help='Also render analysis figures from reproduced results')
    args=parser.parse_args()
    dataset=args.dataset.resolve();output=args.output.resolve()
    if not dataset.is_file():parser.error('The supplied dataset workbook does not exist')
    if output.exists():parser.error('Output directory already exists; choose a new directory to preserve existing results')
    if output==ROOT or output in ROOT.parents:parser.error('Output must not replace the package directory or its ancestors')
    archives={p.relative_to(ROOT).as_posix():sha256(p) for folder in ['open_llm_scores','paid_llm_scores','sentence_embedding_scores','prompt_engineering_scores','banglabert_scores'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    dataset_hash=sha256(dataset)
    output.mkdir(parents=True)
    (output/'scripts').mkdir()
    for p in (ROOT/'scripts').iterdir():
        if p.is_file() and p.suffix in ['.py','.txt','.tsv']:
            shutil.copy2(p,output/'scripts'/p.name)
    (output/'dataset').mkdir()
    shutil.copy2(dataset,output/'dataset/BanglaPlagPair-1K_source_links_final.xlsx')
    inputs=[
        ('open_llm_scores/open_llm_scores.xlsx','free llm evualation/combined_1000_testcases2.xlsx'),
        ('paid_llm_scores/paid_llm_scores.xlsx','paid llm evualation/combined_1000_testcases.xlsx'),
        ('sentence_embedding_scores/sentence_embedding_scores.xlsx','Sentence embedding models evualtion/combined_1000_testcases_ scores3.xlsx'),
    ]
    for source,target in inputs:
        dest=output/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
    shutil.copytree(ROOT/'prompt_engineering_scores',output/'8 prompting results')
    shutil.copytree(ROOT/'banglabert_scores',output/'banglabert_scores')
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    def run(script,*extra):
        subprocess.run([sys.executable,str(output/'scripts'/script),*extra],cwd=output,env=env,check=True)
    run('run_all_experiments.py','--only','1','2','3','5')
    run('step4_run_banglabert_evaluation.py' if args.regenerate_banglabert else 'evaluate_archived_banglabert.py')
    run('run_all_experiments.py','--only','6','7','8','9','10')
    checks=compare_reference_results(output)
    if args.figures:run('generate_report_figures.py')
    if sha256(dataset)!=dataset_hash:raise RuntimeError('Input dataset changed')
    if any(sha256(ROOT/p)!=digest for p,digest in archives.items()):raise RuntimeError('An original score archive changed')
    report={'reference_files_checked':len(checks),'reference_rows_checked':sum(c['rows'] for c in checks),
            'all_reference_results_match':True,'input_dataset_unchanged':True,'input_score_archives_unchanged':True,
            'banglabert_source':'fresh encoder inference' if args.regenerate_banglabert else 'archived pair-level scores',
            'llm_api_calls':0,'checks':checks}
    (output/'reproduction_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'PASS: {len(checks)} reference result files reproduced. No LLM API calls.')

if __name__=='__main__':
    main()
