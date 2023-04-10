import pandas as pd
import numpy as np
import os

DATA_DIR = "../data/processed/cleaned.csv"
OUT_DIR = "../data/processed"


def process_data(data):
    data = data.copy()

    TARGET_VARIABLES = [
            ('CARBS_PERCENT_AVE_ALL', 45, 65),
            ('PROT_PERCENT_AVE_ALL', 10, 35),
            ('FAT_PERCENT_AVE_ALL', 20, 35)
        ]

    CAT_IDX = {
        'CHILD_SEX': ['Male', 'Female'],
        'FOOD_INSECURITY': [np.nan, 'None', 'Mild', 'Moderate', 'Severe'],
        'BEN_4PS': ['No', 'Yes'],
        'AREA_TYPE': ['RURAL', 'URBAN']
    }

    for cat, cat_list in CAT_IDX.items():
        data[cat] = data[cat].apply(cat_list.index)

    def is_healthy(row):
        res = True
        for col, lb, ub in TARGET_VARIABLES:
            res &= (lb <= row[col] <= ub)
        return "REDUCED RISK" if res else "INCREASED RISK"

    data['TARGET'] = data.apply(is_healthy, axis=1)

    for col, lb, ub in TARGET_VARIABLES:
        data.drop(col, inplace=True, axis=1)

    return data


if __name__ == '__main__':
    print("Reading cleaned data...")
    cleaned_data = pd.read_csv(DATA_DIR, index_col=0)
    print("Reading cleaned data completed.")

    processed_data = process_data(cleaned_data)
    print("Saving processed data...")
    processed_data.to_csv(os.path.join(OUT_DIR, 'processed.csv'))
    print("Saved.")
