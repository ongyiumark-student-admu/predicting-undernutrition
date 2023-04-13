from scipy.stats import t
import pandas as pd
import numpy as np
import os
import sys
from predunder.hypertuning import tune_model
from predunder.training import train_random_forest, train_xgboost, train_dnn, train_nnrf, train_kfold
from predunder.functions import get_metrics, convert_labels, kfold_metrics_to_df


DATA_DIR = '../train-test-data'
OVERSAMPLING = ['none', 'smote', 'borderline', 'adasyn']
LATEX_DIR = '../latex'
RESULTS_DIR = '../results'


def get_95_CI(samp_mean, samp_sd, N):
    alpha = 0.05
    t_alpover2 = t.ppf(1-alpha/2, df=N-1)
    CI_LCL = samp_mean - t_alpover2 * samp_sd / (N**0.5)
    CI_UCL = samp_mean + t_alpover2 * samp_sd / (N**0.5)
    return f'$({CI_LCL:.3f},{CI_UCL:.3f})$'


def kfold_to_latex(metrics, task):
    caption = f'Results of 10-fold cross validation on the {task} task with the best hyperparameters based on Cohen\'s $\\kappa$'
    label = 'tab:kfold_results'
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
    label = 'tab:kfold_results'
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


metrics = dict()
results = dict()


def hypertune(train_func, grid_params, over_tech, task):
    best_params = tune_model(train_df, task, 10, train_func, grid_params, oversample=over_tech)

    with open(os.path.join(RESULTS_DIR, f'{task}_best_params.txt'), 'a') as f:
        print(over_tech, best_params, file=f)

    preds = train_func(train_df, test_df, task, **best_params)
    kfold_metrics = train_kfold(train_df, task, 10, train_func, **best_params)
    kfold_df = kfold_metrics_to_df(kfold_metrics)

    results = get_metrics(preds, convert_labels(test_df[task]))
    return kfold_df, {key: val for key, val in zip(['ACCURACY', 'SENSITIVITY', 'SPECIFICITY', 'KAPPA'], results)}


def save_results(over_tech, algo, train_func, grid_params, task):
    try:
        met, res = hypertune(train_func, grid_params, over_tech, task)
        metrics[(over_tech, algo)] = met
        results[(over_tech, algo)] = res
    except (ValueError):
        metrics[(over_tech, algo)] = np.nan
        results[(over_tech, algo)] = np.nan


if __name__ == '__main__':
    TASK = sys.argv[1]
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"{TASK}_train.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(DATA_DIR, f"{TASK}_test.csv"), index_col=0)

    with open(os.path.join(RESULTS_DIR, f'{TASK}_best_params.txt'), 'w') as f:
        print('', file=f)

    for over_tech in OVERSAMPLING:
        rf_grid_params = {
            'bootstrap': [True, False],
            'n_estimators': [50, 100, 200],
            'max_depth': [1, 2, 3],
            'max_features': [1, 2, 'sqrt']
        }
        save_results(over_tech, 'RF', train_random_forest, rf_grid_params, TASK)

        xgb_grid_params = {
            'n_estimators': [50, 100, 200][0:1],
            'max_depth': [1, 2, 3],
            'learning_rate': [0.001, 0.01, 0.1],
            'reg_lambda': [0.1, 1, 10, 100],
            'reg_alpha': [0.1, 1, 10, 100]
        }
        save_results(over_tech, 'XGBoost', train_xgboost, xgb_grid_params, TASK)

        dnn_grid_params = {
            'layers': [[13], [6], [3], [13, 6], [13, 3], [6, 3]],
            'epochs': [1, 10, 30, 60, 100],
        }
        save_results(over_tech, 'DNN', train_dnn, dnn_grid_params, TASK)
        nnrf_grid_params = {
            'n': [50, 100, 200][0:1],
            'd': [1, 2, 3][0:1],
            'learning_rate': [0.01, 0.1, 1],
            'reg_factor': [0.1, 1, 10, 100],
            'to_normalize': [True]
        }
        save_results(over_tech, 'NNRF', train_nnrf, nnrf_grid_params, TASK)

    print(f"Saving tex file for task {TASK}...")
    with open(os.path.join(LATEX_DIR, f'{TASK}_kfold.tex'), 'w') as f:
        print(kfold_to_latex(metrics, TASK), file=f)
    with open(os.path.join(LATEX_DIR, f'{TASK}_test_results.tex'), 'w') as f:
        print(results_to_latex(results, TASK), file=f)
    print("Done.")
