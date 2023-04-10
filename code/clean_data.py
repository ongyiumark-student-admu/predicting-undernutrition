import pandas as pd
import os

URBAN_SHEET = "Valenzuela_20201016_Feb"
RURAL_SHEET = "ComVal_20200118_Oct"
DATA_DIR = "../data/Data.xlsx"
OUT_DIR = "../data/processed"


def clean_data(data, children_data, idx_str):
    X = pd.DataFrame()
    INDIVIDUAL_VARIABLES = ['CHILD_SEX', 'IDD_SCORE', 'AGE']

    TARGET_VARIABLES = [
        ('CARBS_PERCENT_AVE_ALL', 45, 65),
        ('PROT_PERCENT_AVE_ALL', 10, 35),
        ('FAT_PERCENT_AVE_ALL', 20, 35)
    ]
    for idx, row in children_data.iterrows():
        new_row = dict()

        # index
        new_row['idx'] = idx_str + str(row['Index'])

        for col in INDIVIDUAL_VARIABLES:
            new_row[col] = row[col]

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

        for col, lb, ub in TARGET_VARIABLES:
            new_row[col] = row[col]

        X = pd.concat([X, pd.DataFrame(new_row, index=[0])])
        X.dropna(inplace=True)
    return X


if __name__ == '__main__':
    data_rural = pd.read_excel(DATA_DIR, sheet_name=RURAL_SHEET)
    data_urban = pd.read_excel(DATA_DIR, sheet_name=URBAN_SHEET)

    children_urban = data_urban.query("FR_Child == 1")
    children_rural = data_rural.query("FR_Child == 1")

    cleaned_urban = clean_data(data_urban, children_urban, "VZ")
    cleaned_rural = clean_data(data_rural, children_rural, "CV")

    cleaned_data = pd.concat([cleaned_rural, cleaned_urban])
    cleaned_data.to_csv(os.path.join(OUT_DIR, 'cleaned.csv'), index=False)
