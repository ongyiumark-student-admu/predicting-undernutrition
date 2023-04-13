import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency
from typing import Dict, Union, List

CLEANED_DIR = '../cleaned-data'
TRAIN_TEST_DIR = '../train-test-data'
LATEX_DIR = '../latex'
task = '2aii'

SHORTEN = {
    'CHILD_SEX': 'Sex',
    'FOOD_INSECURITY': 'FI',
    'BEN_4PS': '4Ps',
    'AREA_TYPE': 'Area',
    'AGE': 'Age',
    'IDD_SCORE': 'IDD',
    'HDD_SCORE': 'HDD',
    'HHID_count': 'HH Count',
    'HH_AGE': 'HH Age',
    'FOOD_EXPENSE_WEEKLY': 'FE',
    'NON-FOOD_EXPENSE_WEEKLY': 'NE',
    'FOOD_EXPENSE_WEEKLY_pc': 'FE PC',
    'NON-FOOD_EXPENSE_WEEKLY_pc': 'NE PC'
}

BINS: Dict[str, List[float]] = {
    'AGE': [-float('inf'), 3, 6, 10, 13, 19, float('inf')],  # a[i] <= x < a[i+1]
    'IDD_SCORE': [0, 3, 7, 11, 16+1],
    'HDD_SCORE': [0, 3, 7, 11, 16+1]
}

CAT_IDX: Dict[str, List[Union[float, str]]] = {
    'CHILD_SEX': ['Male', 'Female'],
    'FOOD_INSECURITY': [np.nan, 'None', 'Mild', 'Moderate', 'Severe'],
    'BEN_4PS': ['No', 'Yes'],
    'AREA_TYPE': ['Rural', 'Urban']
}


def latex_friendly(s):
    return SHORTEN[s]


def format_results(res):
    statistic = np.round(res.statistic, 2)
    pval = np.round(res.pvalue, 2)
    if res.pvalue < 0.01:
        pval = '$< 0.01$'
    elif res.pvalue < 0.05:
        pval = '$< 0.05$'
    return statistic, str(pval)


def chitest_category(data, task, col):
    bins = sorted(data[col].unique())
    labels = data[task].unique()

    bin_df = pd.DataFrame()
    indices = pd.MultiIndex.from_tuples(
        [(col, CAT_IDX[col][int(x)]) for x in bins],
        names=('Features', 'Values')
    )
    obs = []
    for label in labels:
        label_data = data.query(f"`{task}` == '{label}'")

        bin_data = [len(label_data.query(f"{col} == {x}")) for x in bins]
        bin_df[label] = pd.Series(data=np.array(bin_data), index=indices)
        obs.append(bin_data)

    npobs = np.asarray(obs)

    res = chi2_contingency(npobs, correction=(len(bins) == 2))
    stat, pval = format_results(res)
    bin_df['Statistic'] = pd.Series(data=[stat]*len(indices), index=indices)
    bin_df['p-value'] = pd.Series(data=[pval]*len(indices), index=indices)

    return bin_df, res.statistic, res.pvalue


def chitest_manual(data, task, col):
    bins = BINS[col]
    labels = data[task].unique()

    bin_df = pd.DataFrame()

    def parse_range(a, b):
        if a == -float('inf'):
            return f"$< {b}$"
        if b == float('inf'):
            return f"$> {a}$"
        return f"${a}-{b-1}$"

    indices = pd.MultiIndex.from_tuples(
        [(col, parse_range(x, bins[i+1])) for i, x in enumerate(bins[:-1])],
        names=('Features', 'Values')
    )
    obs = []
    for label in labels:
        label_data = data.query(f"`{task}` == '{label}'")
        bin_data = [len(label_data.query(f"{x} <= `{col}` < {bins[i+1]}")) for i, x in enumerate(bins[:-1])]
        bin_df[label] = pd.Series(data=np.array(bin_data), index=indices)
        obs.append(bin_data)
    npobs = np.asarray(obs)

    res = chi2_contingency(npobs)
    stat, pval = format_results(res)
    bin_df['Statistic'] = pd.Series(data=[stat]*len(indices), index=indices)
    bin_df['p-value'] = pd.Series(data=[pval]*len(indices), index=indices)

    return bin_df, res.statistic, res.pvalue


def chitest_quartile(data, task, col):
    labels = data[task].unique()
    data_quartiles = data[col].describe()

    bins = [-float('inf')] + [data_quartiles[f"{x}%"] for x in range(25, 100, 25)] + [float('inf')]
    bin_df = pd.DataFrame()

    def parse_range_int(a, b):
        if a == -float('inf'):
            return f"$< {int(b)}$"
        if b == float('inf'):
            return f"$> {int(a)}$"
        return f"${int(a)}-{int(b)}$"

    def parse_range(a, b):
        if data[col].dtype == 'int64':
            return parse_range_int(a, b)
        if a == -float('inf'):
            return f"$< {b:.1f}$"
        if b == float('inf'):
            return f"$> {a:.1f}$"
        return f"${a:.1f}-{b:.1f}$"

    indices = pd.MultiIndex.from_tuples(
        [(col, parse_range(x, bins[i+1])) for i, x in enumerate(bins[:-1])],
        names=('Features', 'Values')
    )

    obs = []
    for label in labels:
        label_data = data.query(f"`{task}` == '{label}'")
        q1 = label_data.query(f"`{col}` < {data_quartiles['25%']}")
        q2 = label_data.query(f"{data_quartiles['25%']} <= `{col}` < {data_quartiles['50%']}")
        q3 = label_data.query(f"{data_quartiles['50%']} <= `{col}` < {data_quartiles['75%']}")
        q4 = label_data.query(f"{data_quartiles['75%']} <= `{col}`")

        bin_data = [len(label_data.query(f"{x} <= `{col}` < {bins[i+1]}")) for i, x in enumerate(bins[:-1])]
        assert (bin_data == [len(q1), len(q2), len(q3), len(q4)])
        bin_df[label] = pd.Series(data=np.array(bin_data), index=indices)

        obs.append(bin_data)
    npobs = np.asarray(obs)

    res = chi2_contingency(npobs)
    stat, pval = format_results(res)
    bin_df['Statistic'] = pd.Series(data=([stat]*len(indices)), index=indices)
    bin_df['p-value'] = pd.Series(data=([pval]*(len(indices))), index=indices)

    return bin_df, res.statistic, res.pvalue


def df_to_latex(df, caption, label, position):
    unique_tasks = data[task].unique()

    res = '\\begin{table}[!htb]\n'
    res += '\\centering\n'

    res += f'\\caption{{{caption}}}\n'
    res += f'\\label{{{label}}}\n'
    res += f'\\begin{{tabular}}{{{position}}}\n'

    res += '\\hline\n'
    res += '\\multicolumn{2}{c|}{Features}& ' \
        + f'\\multicolumn{{{len(unique_tasks)}}}{{c|}}{{Labels}}' \
        + '& \\multirow{2}{*}{$\\chi^2$} & \\multirow{2}{*}{p-value}\\\\ \n'
    res += '& & ' + ' & '.join([x.title() for x in unique_tasks]) + ' & & \\\\ \n'
    res += '\\hline\n'

    feature_names = np.unique([x[0] for x in chitest_df.index.to_list()])
    cutoff = [5]

    for feature_num, name in enumerate(feature_names):
        feature_df = df.loc[name]
        res += f'{latex_friendly(name)} & ' + ' & '*len(unique_tasks) \
            + f'& {feature_df["Statistic"].iloc[0]} & {feature_df["p-value"].iloc[0]} \\\\ \n'

        for i, (idx, row) in enumerate(feature_df.iterrows()):
            res += f'& {idx} & ' + ' & '.join([str(x) for x in row[unique_tasks]]) + '& & \\\\ \n'

        res += '\\hline \n'

        if feature_num in cutoff:
            res += '\\end{tabular}\n'
            res += '\\end{table}\n'

            res += '\\begin{table}\n'
            res += '\\centering\n'
            res += f'\\label{{{label}_cont}}\n'
            res += f'\\begin{{tabular}}{{{position}}}\n'
            res += '\\hline\n'

            res += f'\\multicolumn{{{len(unique_tasks)+4}}}{{c}}{{Continuation of Table \\ref{{{label}}}}}\\\\ \n'
            res += '\\hline\n'
            res += '\\multicolumn{2}{c|}{Features}& ' \
                + f'\\multicolumn{{{len(unique_tasks)}}}{{c|}}{{Labels}}' \
                + '& \\multirow{2}{*}{$\\chi^2$} & \\multirow{2}{*}{p-value}\\\\ \n'
            res += '& & ' + ' & '.join([x.title() for x in unique_tasks]) + ' & & \\\\ \n'
            res += '\\hline\n'

    res += '\\end{tabular}\n'
    res += '\\end{table}\n'

    return res


if __name__ == '__main__':
    wf = pd.read_excel(os.path.join(CLEANED_DIR, 'wf.xlsx'), sheet_name='tags')
    tasks = wf['Workflow'].unique()
    with open(os.path.join(LATEX_DIR, 'chitest_tables.tex'), 'w') as f:
        print('', end='', file=f)

    for task in tasks:
        print(f"Generating chi squared test table for {task}...")

        train_df = pd.read_csv(os.path.join(TRAIN_TEST_DIR, f"{task}_train.csv"), index_col=0)
        test_df = pd.read_csv(os.path.join(TRAIN_TEST_DIR, f"{task}_test.csv"), index_col=0)
        data = pd.concat([train_df, test_df], axis=0)
        data_quartiles = data.describe()

        CATEGORICAL_VARIABLES = ['CHILD_SEX', 'FOOD_INSECURITY', 'BEN_4PS', 'AREA_TYPE']
        MANUAL_VARIABLES = ['AGE', 'IDD_SCORE', 'HDD_SCORE']
        QUARTILE_VARIABLES = [x for x in data.drop(task, axis=1).columns.to_list() if x not in CATEGORICAL_VARIABLES+MANUAL_VARIABLES]

        chitest_df = pd.DataFrame()
        for col in CATEGORICAL_VARIABLES:
            bin_df, stat, pval = chitest_category(data, task, col)
            chitest_df = pd.concat([chitest_df, bin_df], axis=0)

        for col in MANUAL_VARIABLES:
            try:
                bin_df, stat, pval = chitest_manual(data, task, col)
            except (ValueError):
                bin_df, stat, pval = chitest_quartile(data, task, col)
            chitest_df = pd.concat([chitest_df, bin_df], axis=0)

        for col in QUARTILE_VARIABLES:
            bin_df, stat, pval = chitest_quartile(data, task, col)
            chitest_df = pd.concat([chitest_df, bin_df], axis=0)

        col_format = "c c |" + " c"*len(data[task].unique()) + "| c | c"
        caption = f"Asssessing the association between features and labels in the {task} task using $\\chi^2$ test for independence"

        with open(os.path.join(LATEX_DIR, 'chitest_tables.tex'), 'a') as f:
            print(df_to_latex(chitest_df, caption, f"tab:chitest_{task}", col_format), file=f)
        print("Done.")
