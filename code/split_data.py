import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

from global_variables import CLEANED_DIR, TRAIN_TEST_DIR


def initialize_parser():
    parser = argparse.ArgumentParser(
        description="Split data into training group and testing group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-tsz", "--testsize", help="proportion size of the testing group",
                        type=int,
                        default=50
                        )
    parser.add_argument("-s", "--seed", help="random seed",
                        type=int,
                        default=42
                        )
    args = parser.parse_args()
    config = vars(args)

    return config['testsize'], config['seed']


if __name__ == '__main__':
    testsize, seed = initialize_parser()

    X = pd.read_csv(os.path.join(CLEANED_DIR, 'cleaned_X.csv'), index_col=0)
    tags = pd.read_csv(os.path.join(CLEANED_DIR, 'final_tags.csv'), index_col=0)

    for task in tags.columns.to_list():
        print(f"Splitting {task}")
        data = pd.concat([X, tags[[task]]], axis=1)
        data.dropna(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(task, axis=1),
            data[[task]],
            test_size=testsize/100,
            random_state=seed,
            stratify=data[task]
        )
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)

        train_df.to_csv(os.path.join(TRAIN_TEST_DIR, f'{task}_train.csv'))
        test_df.to_csv(os.path.join(TRAIN_TEST_DIR, f'{task}_test.csv'))
        print(f"Task {task} completed.", end='\n\n')
