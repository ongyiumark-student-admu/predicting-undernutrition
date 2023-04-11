import pandas as pd
import os

DATA_DIR = '../cleaned-data'


if __name__ == '__main__':

    TASKS = ['1b', '2ai', '2aii', '2aiii', '2biv1', '2biv2', '2biv3', '2biv4', '3a', '3b', '3c', '3d']
    tags = pd.ExcelFile(os.path.join(DATA_DIR, 'tags.xlsx'))

    data = pd.DataFrame()
    for task in TASKS:
        task_df = tags.parse(sheet_name=task)
        targets = [col for col in task_df.columns.to_list() if col != 'idx']
        new_df = pd.DataFrame()

        for i, row in task_df.iterrows():
            new_row = dict()

            if task[0] == '3':
                new_row[task] = row[targets[0]]
            else:
                is_healthy = True
                for col in targets:
                    is_healthy &= (row[col] == 'REDUCED RISK')
                new_row[task] = "REDUCED RISK" if is_healthy else "INCREASED RISK"

            new_df = pd.concat([new_df, pd.DataFrame(new_row, index=[row['idx'].values])])

        data = pd.concat([data, new_df], axis=1)
    data.to_csv(os.path.join(DATA_DIR, 'final_tags.csv'))
