import pandas as pd
import os

DATA_DIR = '../cleaned-data'


if __name__ == '__main__':
    X_df = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_X.csv'))
    y_df = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_y.csv'))
    wf = pd.read_excel(os.path.join(DATA_DIR, 'wf.xlsx'), sheet_name='tags')
    reni = pd.read_excel(os.path.join(DATA_DIR, 'wf.xlsx'), sheet_name='reni')

    def get_reni_idx(row):
        return f"{row['CHILD_SEX']}_{int(row['AGE'])}"

    X_df['RENI_IDX'] = X_df.apply(get_reni_idx, axis=1)
    data = y_df.merge(X_df[['idx', 'RENI_IDX']], on='idx')

    def get_limits(sex_age, threshold):
        if ',' in str(threshold):
            lb, ub = [float(x) for x in threshold.split(',')]
        elif threshold == "RENI":
            task_reni = reni.query(f"SEX_AGE == '{row['RENI_IDX']}'")
            reni_col = target[:-8]

            if len(task_reni) == 0:
                return None, None

            lb = float(task_reni[reni_col].values[0])/2
            ub = float(task_reni[reni_col].values[0])*2
        else:
            lb, ub = 100.0, float('inf')

        return lb, ub

    tasks = wf['Workflow'].unique()
    with pd.ExcelWriter(os.path.join(DATA_DIR, 'tags.xlsx'), engine='xlsxwriter') as writer:
        for task in tasks:
            print(f"Processing {task}...")
            task_wf = wf.query(f"Workflow == '{task}'")
            targets = task_wf['Target'].to_list()

            task_df = data[['idx', 'RENI_IDX'] + targets].copy()
            task_df.dropna(inplace=True)
            task_tags = pd.DataFrame()

            for j, row in task_df.iterrows():
                new_row = dict()
                for i, (wf_name, target, threshold) in task_wf.iterrows():
                    if wf_name[0] == '3':
                        labels = ['UNDER', 'ADEQUATE', 'OVER']
                    else:
                        labels = ['INCREASED_RISK', 'REDUCED RISK']

                    # compute lower and upper bounds
                    lb, ub = get_limits(row['RENI_IDX'], threshold)

                    new_row['idx'] = row['idx']

                    if lb is None:
                        continue

                    if len(labels) == 3:
                        if row[target] < lb:
                            new_row[target] = labels[0]
                        elif row[target] > ub:
                            new_row[target] = labels[2]
                        else:
                            new_row[target] = labels[1]
                    else:
                        new_row[target] = labels[lb <= float(row[target]) <= ub]

                task_tags = pd.concat([task_tags, pd.DataFrame(new_row, index=[0])])

            task_tags.to_excel(writer, sheet_name=task, index=False)
            print(f"Completed {task}.")
