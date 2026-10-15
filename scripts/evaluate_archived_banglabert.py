"""Evaluate unchanged archived BanglaBERT scores without model inference."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from pipeline_common import WORK_ROOT, BANGLABERT_DIR, METRICS, evaluate_scores, write_evaluation_outputs

BERT_METRICS = ['BanglaBERT_Cosine_BUETCSE', 'BanglaBERT_Cosine_SagorSarkar']

def main():
    for condition, scores_path in [
        ('normal', WORK_ROOT/'dataset/banglaplagpair1k_similarity_scores.csv'),
        ('polished', WORK_ROOT/'polish dataset/banglaplagpair1k_similarity_scores_polished.xlsx'),
    ]:
        fresh = pd.read_excel(scores_path) if scores_path.suffix=='.xlsx' else pd.read_csv(scores_path)
        archive_dir = WORK_ROOT/'banglabert_scores'/condition
        archived = pd.read_csv(archive_dir/'labeled_scores.csv')
        if fresh.pair_no.tolist()!=archived.pair_no.tolist() or fresh.Label.tolist()!=archived.Label.tolist():
            raise ValueError(f'{condition}: archived identifiers/labels do not match dataset')
        for col in ['text1','text2']:
            if not fresh[col].astype(str).equals(archived[col].astype(str)):
                raise ValueError(f'{condition}: archived passage texts differ')
        for col in METRICS:
            if not np.allclose(fresh[col],archived[col],atol=5.1e-7,rtol=0):
                raise ValueError(f'{condition}: traditional score mismatch for {col}')
        for col in BERT_METRICS:
            values=pd.to_numeric(archived[col],errors='raise')
            if not np.isfinite(values).all() or not values.between(0,1).all():
                raise ValueError(f'{condition}: invalid archived scores for {col}')
            fresh[col]=values
        labeled,fixed,sweep,best=evaluate_scores(fresh,METRICS+BERT_METRICS)
        out=BANGLABERT_DIR/condition
        write_evaluation_outputs(out,labeled,fixed,sweep,best,METRICS+BERT_METRICS,'interactive_threshold_evaluation.xlsx')
        if (archive_dir/'encoder_diagnostics.csv').exists():
            shutil.copy2(archive_dir/'encoder_diagnostics.csv',out/'encoder_diagnostics.csv')
        print(f'Evaluated archived BanglaBERT scores: {condition}')

if __name__=='__main__':
    main()
