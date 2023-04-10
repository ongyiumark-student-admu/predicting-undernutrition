import pandas as pd
import os

URBAN_SHEET = "Valenzuela_20201016_Feb"
RURAL_SHEET = "ComVal_20200118_Oct"
DATA_DIR = "../data/Data.xlsx"
OUT_DIR = "../cleaned-data"


def clean_data(data, children_data, idx_str):
    X = pd.DataFrame()
    y = pd.DataFrame()

    INDIVIDUAL_VARIABLES = ['CHILD_SEX', 'IDD_SCORE', 'AGE']
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
        'NIACIN_mg_NE_AVE_ALL'
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

        target = dict()
        target['idx'] = new_row['idx']
        for col in TARGET_VARIABLES:
            if col in row.index:
                target[col] = row[col]
            elif col.endswith("AVE_ALL"):
                nutrient = col[:-7]
                if nutrient == "PROT_PERCENT_":
                    nutrient = "PROTEIN_PERCENT_"
                target[col] = row[[nutrient + day for day in ['WKDAY1', 'WKDAY2', 'WKEND']]].mean()

        X = pd.concat([X, pd.DataFrame(new_row, index=[0])])
        y = pd.concat([y, pd.DataFrame(target, index=[0])])

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
    cleaned_X.to_csv(os.path.join(OUT_DIR, 'cleaned_y.csv'), index=False)
    print("Saved.")
