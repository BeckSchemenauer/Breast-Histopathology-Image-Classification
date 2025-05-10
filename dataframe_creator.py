import pandas as pd
import re


def extract_info_from_filename(df):
    # Mapping for tumor types
    tumor_type_map = {
        'A': 'adenosis',
        'F': 'fibroadenoma',
        'PT': 'phyllodes tumor',
        'TA': 'tubular adenoma',
        'DC': 'carcinoma',
        'LC': 'lobular carcinoma',
        'MC': 'mucinous carcinoma',
        'PC': 'papillary carcinoma'
    }

    def parse_filename(path):
        filename = path.split('/')[-1]
        parts = filename.split('-')

        mop_class_type_parts = parts[0].split('_')

        # Method of procedure biopsy
        mop = mop_class_type_parts[0]

        # Tumor class: 0 for 'B', 1 otherwise
        tumor_class = 0 if mop_class_type_parts[1] == 'B' else 1

        # Tumor type code
        tumor_type = tumor_type_map.get(mop_class_type_parts[2], 'Unknown')

        # Patient ID
        patient_id = parts[1] + '-' + parts[2]

        return pd.Series([mop, tumor_class, tumor_type, patient_id])

    df[['mop', 'tumor_class', 'tumor_type', 'patient_id']] = df['filename'].apply(
        parse_filename)
    return df


df = pd.read_csv('Folds.csv')

df = extract_info_from_filename(df)
df.to_csv('folds_updated.csv', index=False)
