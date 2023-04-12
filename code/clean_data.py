import pandas as pd
import numpy as np
import os

URBAN_SHEET = "Valenzuela_20201016_Feb"
RURAL_SHEET = "ComVal_20200118_Oct"
DATA_DIR = "../data/Data.xlsx"
OUT_DIR = "../cleaned-data"

INDIVIDUAL_VARIABLES = ['CHILD_SEX', 'IDD_SCORE', 'AGE']
CAT_IDX = {
    'CHILD_SEX': ['Male', 'Female'],
    'FOOD_INSECURITY': [np.nan, 'None', 'Mild', 'Moderate', 'Severe'],
    'BEN_4PS': ['No', 'Yes'],
    'AREA_TYPE': ['RURAL', 'URBAN']
}
TARGET_VARIABLES = [
    'ENERGY_%_ADEQ_ALL',
    'IRON_%_ADEQ_ALL',
    'VIT_A_%_ADEQ_ALL',
    'PROTEIN_%_ADEQ_ALL',
    'CARBS_PERCENT_AVE_ALL',
    'PROT_PERCENT_AVE_ALL',
    'FAT_PERCENT_AVE_ALL',
    'PROTEIN_g_AVE_ALL',
    'CALCIUM_mg_AVE_ALL',
    'PHOSPHORUS_mg_AVE_ALL',
    'IRON_mg_AVE_ALL',
    'VIT_A_ug_RE_AVE_ALL',
    'THIAMIN_mg_AVE_ALL',
    'RIBOFLAVIN_mg_AVE_ALL',
    'NIACIN_mg_NE_AVE_ALL',
    'VIT_C_mg_AVE_ALL'
]
INTEGER_VARIABLES = [
    'IDD_SCORE',
    'AGE',
    'HHID_count',
    'HDD_SCORE',
    'FOOD_INSECURITY'
]
BOOLEAN_VARIABLES = [
    'CHILD_SEX',
    'BEN_4PS',
    'AREA_TYPE'
]


def clean_data(data, children_data, idx_str):
    X = pd.DataFrame()
    y = pd.DataFrame()

    for idx, row in children_data.iterrows():
        new_row = dict()

        # index
        new_row['idx'] = idx_str + str(row['Index'])

        # individual variables
        for col in INDIVIDUAL_VARIABLES:
            new_row[col] = row[col]

        # household variables
        house_data = data.query(f'HHID == {row["HHID"]}')
        new_row['HHID_count'] = len(house_data)
        new_row['HH_AGE'] = house_data['AGE'].mean()
        new_row['FOOD_EXPENSE_WEEKLY'] = house_data['FOOD_EXPENSE_WEEKLY'].sum()
        new_row['NON-FOOD_EXPENSE_WEEKLY'] = house_data['NON-FOOD_EXPENSE_WEEKLY'].sum()
        new_row['HDD_SCORE'] = house_data['HDD_SCORE'].sum()

        for fi in house_data['FOOD_INSECURITY']:
            if fi in ['None', 'Mild', 'Moderate', 'Severe']:
                new_row['FOOD_INSECURITY'] = fi

        for ben_4ps in house_data['BEN_4PS']:
            if ben_4ps in ['Yes', 'No']:
                new_row['BEN_4PS'] = ben_4ps

        new_row['AREA_TYPE'] = 'URBAN' if idx_str == "VZ" else "RURAL"

        new_row['FOOD_EXPENSE_WEEKLY_pc'] = new_row['FOOD_EXPENSE_WEEKLY'] / new_row['HHID_count']
        new_row['NON-FOOD_EXPENSE_WEEKLY_pc'] = new_row['NON-FOOD_EXPENSE_WEEKLY'] / new_row['HHID_count']

        target = dict()
        target['idx'] = new_row['idx']
        for col in TARGET_VARIABLES:
            if col in row.index:
                if str(row[col]).strip() == '':
                    target[col] = np.nan
                    continue
                target[col] = row[col]
            elif col.endswith("AVE_ALL"):
                nutrient = col[:-7]
                # alternate names
                if nutrient == "PROT_PERCENT_" and idx_str == 'CV':
                    nutrient = "PROTEIN_PERCENT_"
                if nutrient == "VIT_C_mg_" and idx_str == 'VZ':
                    nutrient = "VIT._C_mg_"

                suffixes = ['Wkday1', 'Wkday2', 'Wkend'] if idx_str == "VZ" else ['WKDAY1', 'WKDAY2', 'WKEND']
                target[col] = row[[nutrient + day for day in suffixes]].mean()

        # convert categorical variables
        for cat_name, cat_list in CAT_IDX.items():
            if cat_name not in new_row.keys():
                new_row[cat_name] = np.nan
            elif new_row[cat_name] in cat_list:
                new_row[cat_name] = cat_list.index(new_row[cat_name])
            else:
                new_row[cat_name] = np.nan

        X = pd.concat([X, pd.DataFrame(new_row, index=[0])])
        y = pd.concat([y, pd.DataFrame(target, index=[0])])

    # drop rows with missing values
    X.dropna(inplace=True)

    for col in INTEGER_VARIABLES:
        X[col] = X[col].astype('int')
    for col in BOOLEAN_VARIABLES:
        X[col] = X[col].astype('bool')

    return X, y


if __name__ == '__main__':
    print("Reading raw data...")
    data_rural = pd.read_excel(DATA_DIR, sheet_name=RURAL_SHEET)
    data_urban = pd.read_excel(DATA_DIR, sheet_name=URBAN_SHEET)
    print("Reading raw data completed.")

    print("Gathering child data...")
    children_urban = data_urban.query("FR_Child == 1")
    children_rural = data_rural.query("FR_Child == 1")
    print("Gathering child data completed.")

    print("Cleaning urban data...")
    X_urban, y_urban = clean_data(data_urban, children_urban, "VZ")
    print("Cleaning urban data completed.")

    print("Cleaning rural data...")
    X_rural, y_rural = clean_data(data_rural, children_rural, "CV")
    print("Cleaning rural data completed.")

    cleaned_X = pd.concat([X_rural, X_urban])
    cleaned_y = pd.concat([y_rural, y_urban])

    print("Saving cleaned data...")
    cleaned_X.to_csv(os.path.join(OUT_DIR, 'cleaned_X.csv'), index=False)
    cleaned_y.to_csv(os.path.join(OUT_DIR, 'cleaned_y.csv'), index=False)
    print("Saved.")
