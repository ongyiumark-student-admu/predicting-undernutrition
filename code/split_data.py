import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_DIR = '../cleaned-data'
OUT_DIR = '../train-test-data'


if __name__ == '__main__':
    X = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_X.csv'), index_col=0)
    tags = pd.read_csv(os.path.join(DATA_DIR, 'final_tags.csv'), index_col=0)

    for task in tags.columns.to_list():
        print(f"Splitting {task}")
        data = pd.concat([X, tags[[task]]], axis=1)
        data.dropna(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(task, axis=1),
            data[[task]],
            test_size=0.5,
            random_state=42,
            stratify=data[task]
        )
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)

        train_df.to_csv(os.path.join(OUT_DIR, f'{task}_train.csv'))
        test_df.to_csv(os.path.join(OUT_DIR, f'{task}_test.csv'))
        print(f"Task {task} completed.")
