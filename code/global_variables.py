import numpy as np
from typing import Dict, Union, List

URBAN_SHEET: str = "Valenzuela_20201016_Feb"
RURAL_SHEET: str = "ComVal_20200118_Oct"
DATA_DIR: str = "../data"
DATA_FILE: str = "../data/Data.xlsx"
CLEANED_DIR: str = "../cleaned-data"
TRAIN_TEST_DIR: str = '../train-test-data'
LATEX_DIR: str = '../latex'
RESULTS_DIR: str = '../results'

INDIVIDUAL_VARIABLES: List[str] = ['CHILD_SEX', 'IDD_SCORE', 'AGE']
CAT_IDX: Dict[str, List[Union[float, str]]] = {
    'CHILD_SEX': ['Male', 'Female'],
    'FOOD_INSECURITY': [np.nan, 'None', 'Mild', 'Moderate', 'Severe'],
    'BEN_4PS': ['No', 'Yes'],
    'AREA_TYPE': ['RURAL', 'URBAN']
}

TARGET_VARIABLES: List[str] = [
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
INTEGER_VARIABLES: List[str] = [
    'IDD_SCORE',
    'AGE',
    'HHID_count',
    'HDD_SCORE',
    'FOOD_INSECURITY'
]
BOOLEAN_VARIABLES: List[str] = [
    'CHILD_SEX',
    'BEN_4PS',
    'AREA_TYPE'
]

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

OVERSAMPLING = ['none', 'smote', 'borderline', 'adasyn']
