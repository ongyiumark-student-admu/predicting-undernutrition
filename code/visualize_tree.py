import argparse
from predunder.functions import oversample_data, convert_labels
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import dtreeviz


def initialize_parser():
    parser = argparse.ArgumentParser(
        description="Visualize a decision tree of a classification task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-t", "--task", help="classification task code",
                        choices=['1b', '2ai', '2aii', '2aiii', '2biv1', '2biv2', '2biv3', '2biv4'],
                        default='1b'
                        )
    parser.add_argument("-n", "--number", help="decision tree number", default=0)
    parser.add_argument("-o", "--oversampling", help="oversampling technique",
                        choices=['none', 'smote', 'borderline', 'adasyn'],
                        default='none'
                        )
    parser.add_argument("-s", "--save", action="store_true", help="to save file")
    args = parser.parse_args()
    config = vars(args)

    return config['task'], config['number'], config['oversampling'], config['save']


RESULTS_DIR = '../results'
TRAIN_TEST_DIR = '../train-test-data'
FIGURES_DIR = '../figures'


def find_best_params(task, oversampling):
    print(f"Looking for the best parameters for task {task} with '{oversampling}' oversampling...")
    with open(os.path.join(RESULTS_DIR, f"{task}_best_params.txt"), 'r') as f:
        for line in f.read().split('\n'):
            curr_oversampling, model, params_str = line.split('-')
            if model != 'RF' or curr_oversampling != oversampling:
                continue
            print(f"Found parameters {params_str}")
            return eval(params_str)


if __name__ == '__main__':
    task, number, oversampling, to_save = initialize_parser()
    best_params = find_best_params(task, oversampling)

    train_df = pd.read_csv(os.path.join(TRAIN_TEST_DIR, f"{task}_train.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(TRAIN_TEST_DIR, f"{task}_test.csv"), index_col=0)

    train_df = oversample_data(train_df, task)

    rf = RandomForestClassifier(**best_params, random_state=42)

    features = ['AGE', 'IDD_SCORE', 'AREA_TYPE']
    X_tr, y_tr = train_df[features].values, train_df[task].values
    X_te, y_te = test_df[features].values, test_df[task].values

    print(f"Training model with the following features: {features}...")
    rf.fit(X_tr, convert_labels(y_tr))
    print("Done.")

    viz_model = dtreeviz.model(
        rf.estimators_[number],
        X_train=X_tr, y_train=convert_labels(y_tr),
        feature_names=features,
        target_name=task,
        class_names=['REDUCED RISK', 'INCREASED RISK']
    )

    v = viz_model.view()

    if to_save:
        print("Saving file...")
        v.save(os.path.join(FIGURES_DIR, f"{task}_{oversampling}_{number}.svg"))
        print("Saved.")
    else:
        v.show()
