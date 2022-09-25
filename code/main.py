import predunder.hypertuning as puh
import pandas as pd
import os

DATA_DIR = '../train-test-data'
RESULTS_DIR = '../results'
TASK = '2aii'

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(DATA_DIR, f'{TASK}_train.csv'), index_col=0)
    results = puh.tune_dnn(train, TASK, 10, 17, False)
    results.to_csv(os.path.join(RESULTS_DIR, 'preliminary_dnn.csv'), index=False)

    results = puh.tune_dnn(train, TASK, 10, 17, True)
    results.to_csv(os.path.join(RESULTS_DIR, 'preliminary_dnn_sm.csv'), index=False)
