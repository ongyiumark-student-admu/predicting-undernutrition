from scipy.stats import t
import pandas as pd
import numpy as np
import os
from predunder.hypertuning import tune_model
from predunder.training import train_random_forest, train_xgboost, train_dnn, train_nnrf, train_kfold
from predunder.functions import get_metrics, convert_labels, kfold_metrics_to_df
from typing import Dict, Union, Any, Tuple
import argparse

from global_variables import (TRAIN_TEST_DIR, LATEX_DIR, RESULTS_DIR, OVERSAMPLING)


def get_95_CI(samp_mean, samp_sd, N):
    alpha = 0.05
    t_alpover2 = t.ppf(1-alpha/2, df=N-1)
    CI_LCL = samp_mean - t_alpover2 * samp_sd / (N**0.5)
    CI_UCL = samp_mean + t_alpover2 * samp_sd / (N**0.5)
    return f'$({CI_LCL:.3f},{CI_UCL:.3f})$'


def kfold_to_latex(metrics, task):
    caption = f'Results of 10-fold cross validation on the {task} task with the best hyperparameters based on Cohen\'s $\\kappa$'
    label = f'tab:{task}_kfold_results'
    position = 'c | c c c c'

    algorithms = ['RF', 'XGBoost', 'DNN', 'NNRF']

    res = '\\begin{table}[!htb]\n'
    res += '\\centering\n'

    res += f'\\caption{{{caption}}}\n'
    res += f'\\label{{{label}}}\n'
    res += '\\footnotesize\n'
    res += f'\\begin{{tabular}}{{{position}}}\n'

    res += '\\hline\n'
    res += ' & \\multicolumn{4}{c}{Algorithms}\\\\ \n'
    res += 'Metrics &' + ' & '.join(algorithms) + '\\\\ \n'
    res += '\\hline\n'

    cutoff = []
    for i, over_tech in enumerate(['None', 'SMOTE', 'Borderline', 'ADASYN']):
        res += f'Oversampling &\\multicolumn{{4}}{{c}}{{\\textbf{{{over_tech}}}}}' + '\\\\ \n'
        res += '\\hline\n'
        for met in ['Accuracy', 'Sensitivity', 'Specificity', 'Kappa']:
            met_name = met
            if met == 'Kappa':
                met_name = "Cohen's $\\kappa$"

            row = [metrics[(over_tech.lower(), algo)] for algo in algorithms]

            row_str = []
            CI_row = []
            for metric in row:
                if type(metric) == float:
                    row_str.append('')
                    CI_row.append('')
                else:
                    CI_str = get_95_CI(metric[met.upper()+"_MEAN"].iloc[0], metric[met.upper()+"_STDEV"].iloc[0], 10)
                    row_str.append(f'{metric[met.upper()+"_MEAN"].iloc[0]:.4f}')
                    CI_row.append(CI_str)
            res += f'{met_name} & ' + ' & '.join(row_str) + '\\\\ \n'
            res += '(95\\% CI) & ' + ' & '.join(CI_row) + '\\\\ \n'

        res += '\\hline\n'

        if i in cutoff:
            res += '\\end{tabular}\n'
            res += '\\end{table}\n'

            res += '\\begin{table}\n'
            res += '\\centering\n'
            res += f'\\label{{{label}_cont}}\n'
            res += f'\\begin{{tabular}}{{{position}}}\n'
            res += '\\hline\n'

            res += f'\\multicolumn{{{5}}}{{c}}{{Continuation of Table \\ref{{{label}}}}}\\\\ \n'
            res += '\\hline\n'
            res += ' & \\multicolumn{4}{c}{Algorithms}\\\\ \n'
            res += 'Metrics &' + ' & '.join(algorithms) + '\\\\ \n'
            res += '\\hline\n'

    res += '\\end{tabular}\n'
    res += '\\end{table}\n'
    return res


def results_to_latex(results, task):
    caption = f'Results of test data on the {task} task using the best hyperparameters based on Cohen\'s $\\kappa$'
    label = f'tab:{task}_test_results'
    position = 'c | c c c c'

    algorithms = ['RF', 'XGBoost', 'DNN', 'NNRF']

    res = '\\begin{table}[!htb]\n'
    res += '\\centering\n'

    res += f'\\caption{{{caption}}}\n'
    res += f'\\label{{{label}}}\n'
    res += f'\\begin{{tabular}}{{{position}}}\n'

    res += '\\hline\n'
    res += ' & \\multicolumn{4}{c}{Algorithms}\\\\ \n'
    res += 'Metrics &' + ' & '.join(algorithms) + '\\\\ \n'
    res += '\\hline\n'

    for i, over_tech in enumerate(['None', 'SMOTE', 'Borderline', 'ADASYN']):
        res += f'Oversampling &\\multicolumn{{4}}{{|c}}{{\\textbf{{{over_tech}}}}}' + '\\\\ \n'
        res += '\\hline\n'
        for met in ['Accuracy', 'Sensitivity', 'Specificity', 'Kappa']:
            met_name = met
            if met == 'Kappa':
                met_name = "Cohen's $\\kappa$"

            row = [results[(over_tech.lower(), algo)] for algo in algorithms]

            nrow = []
            for metric in row:
                if type(metric) == float:
                    nrow.append(np.nan)
                else:
                    nrow.append(metric[met.upper()])

            row_str = ' & '.join([f'{x:.4f}' for x in nrow])
            row_str = row_str.replace('nan', '')
            res += f'{met_name} & ' + row_str + '\\\\ \n'

        res += '\\hline\n'

    res += '\\end{tabular}\n'
    res += '\\end{table}\n'
    return res


metrics: Dict[Tuple[str, str], Any] = dict()
results: Dict[Tuple[str, str], Union[float, Tuple[float, float, float, float]]] = dict()


def run_algo(train_func, best_params, over_tech, task):
    preds = train_func(train_df, test_df, task, **best_params, oversample=over_tech, random_state=42)
    kfold_metrics = train_kfold(train_df, task, 10, train_func, **best_params, oversample=over_tech, random_state=42)
    kfold_df = kfold_metrics_to_df(kfold_metrics)

    results = get_metrics(preds, convert_labels(test_df[task]))
    return kfold_df, {key: val for key, val in zip(['ACCURACY', 'SENSITIVITY', 'SPECIFICITY', 'KAPPA'], results)}


def hypertune(train_func, grid_params, over_tech, task, algo):
    best_params = tune_model(train_df, task, 10, train_func, grid_params, oversample=over_tech, random_state=42)

    with open(os.path.join(RESULTS_DIR, f'{task}_best_params.txt'), 'a') as f:
        print(over_tech, algo, best_params, sep='-', file=f)

    return run_algo(train_func, best_params, over_tech, task)


def save_results(over_tech, algo, train_func, grid_params, task):
    if (over_tech, algo) in best_saved.keys():
        print(f'Found existing parameters for {algo} with "{over_tech}" over-sampling.')
        print(best_saved[over_tech, algo])
        print(f'Running {algo} on the these parameters...')
        met, res = run_algo(train_func, best_saved[over_tech, algo], over_tech, task)
        metrics[(over_tech, algo)] = met
        results[(over_tech, algo)] = res
        print(f'Done saving results for {algo} with "{over_tech}" over-sampling.')
        return
    try:
        met, res = hypertune(train_func, grid_params, over_tech, task, algo)
        metrics[(over_tech, algo)] = met
        results[(over_tech, algo)] = res
    except (ValueError):
        metrics[(over_tech, algo)] = np.nan
        results[(over_tech, algo)] = np.nan


def read_bests(task):
    if not os.path.exists(os.path.join(RESULTS_DIR, f'{task}_best_params.txt')):
        return dict()
    res = dict()
    with open(os.path.join(RESULTS_DIR, f'{task}_best_params.txt'), 'r') as f:
        o_params = [x.split('-') for x in f.read().split('\n') if len(x) > 0]
        for over_tech, algo, param_str in o_params:
            param = eval(param_str)
            res[over_tech, algo] = param
    return res


def initialize_parser():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters and evaluate on the testing group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-t", "--task", help="classification task code",
                        choices=['1b', '2ai', '2aii', '2aiii', '2biv1', '2biv2', '2biv3', '2biv4'],
                        default='1b'
                        )
    parser.add_argument("-r", "--reset", action="store_true", help="reset saved parameters")
    args = parser.parse_args()
    config = vars(args)

    return config['task'], config['reset']


if __name__ == '__main__':
    TASK, to_reset = initialize_parser()
    train_df = pd.read_csv(os.path.join(TRAIN_TEST_DIR, f"{TASK}_train.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(TRAIN_TEST_DIR, f"{TASK}_test.csv"), index_col=0)

    global best_saved
    best_saved = read_bests(TASK)

    if to_reset:
        with open(os.path.join(RESULTS_DIR, f'{TASK}_best_params.txt'), 'w') as f:
            print('', end='', file=f)

    for over_tech in OVERSAMPLING:
        rf_grid_params = {
            'bootstrap': [True, False],
            'n_estimators': [50, 100, 200],
            'max_depth': [1, 2, 3],
            'max_features': [1, 2, 'sqrt']
        }
        save_results(over_tech, 'RF', train_random_forest, rf_grid_params, TASK)

        xgb_grid_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [1, 2, 3],
            'learning_rate': [0.001, 0.01, 0.1],
            'reg_lambda': [0.1, 1, 10, 100, 1000],
            'reg_alpha': [0.1, 1, 10, 100, 1000]
        }
        save_results(over_tech, 'XGBoost', train_xgboost, xgb_grid_params, TASK)

        dnn_grid_params = {
            'layers': [[13], [6], [3], [13, 6], [13, 3], [6, 3]],
            'epochs': [1, 10, 30],
        }
        save_results(over_tech, 'DNN', train_dnn, dnn_grid_params, TASK)
        nnrf_grid_params = {
            'n': [50],
            'd': [4],
            'l1': [0, 0.01, 0.1, 1, 10],
            'l2': [0, 0.01, 0.1, 1, 10],
            'max_iter': [1, 10, 30],
            'to_normalize': [True]
        }
        save_results(over_tech, 'NNRF', train_nnrf, nnrf_grid_params, TASK)

    print(f"Saving tex file for task {TASK}...")
    with open(os.path.join(LATEX_DIR, f'{TASK}_kfold.tex'), 'w') as f:
        print(kfold_to_latex(metrics, TASK), file=f)
    with open(os.path.join(LATEX_DIR, f'{TASK}_test_results.tex'), 'w') as f:
        print(results_to_latex(results, TASK), file=f)
    print("Done.")
