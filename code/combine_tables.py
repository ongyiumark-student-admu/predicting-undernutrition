import os
import pandas as pd

CLEANED_DIR = '../cleaned-data'
LATEX_DIR = '../latex'


if __name__ == '__main__':
    wf = pd.read_excel(os.path.join(CLEANED_DIR, 'wf.xlsx'), sheet_name='tags')
    tasks = wf['Workflow'].unique()

    with open(os.path.join(LATEX_DIR, 'all_model_results.tex'), 'w') as f:
        for task in tasks:
            for suffix in ['_kfold.tex', '_test_results.tex']:
                path = os.path.join(LATEX_DIR, task+suffix)
                if not os.path.exists(path):
                    continue
                print(f"Found {task+suffix}.")
                with open(path) as g:
                    print(g.read(), file=f)
