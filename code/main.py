import predunder.hypertuning as puh
import pandas as pd
import os

DATA_DIR = '../train-test-data'
RESULTS_DIR = '../results'
TASK = '2aii'


def run_dnns():
    train = pd.read_csv(os.path.join(DATA_DIR, f'{TASK}_train.csv'), index_col=0)
    n = len(train.columns)-1
    results = puh.tune_dnn(train, TASK, 10, n)
    results.to_csv(os.path.join(RESULTS_DIR, 'preliminary_dnn.csv'), index=False)

    results = puh.tune_dnn(train, TASK, 10, n, "smote")
    results.to_csv(os.path.join(RESULTS_DIR, 'preliminary_dnn_sm.csv'), index=False)

    results = puh.tune_dnn(train, TASK, 10, n, "adasyn")
    results.to_csv(os.path.join(RESULTS_DIR, 'preliminary_dnn_adasyn.csv'), index=False)

    results = puh.tune_dnn(train, TASK, 10, n, "borderline")
    results.to_csv(os.path.join(RESULTS_DIR, 'preliminary_dnn_borderline.csv'), index=False)


if __name__ == '__main__':
    run_dnns()
