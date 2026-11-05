#!/usr/bin/env python

import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from features import ad_words


DATA_FILE = 'data/raw/djinni-jobs-en.parquet'
MODEL_FILE = 'model.bin'
CATEGORIES = ['JavaScript', 'Java', 'DevOps', 'QA Automation', 'Python', 'Data Science']
MIN_ADS = 5
C = 0.03


def load_data():
    df = pd.read_parquet(DATA_FILE)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    description_norm = df.long_description.str.lower().str.split().str.join(' ')
    df['description_start'] = description_norm.str[:300]
    df['company_norm'] = df.company_name.str.lower().str.strip()

    df['company_group'] = df.company_norm
    while True:
        before = df.company_group.copy()
        df['company_group'] = df.groupby('description_start').company_group.transform('min')
        df['company_group'] = df.groupby('company_norm').company_group.transform('min')
        if (df.company_group == before).all():
            break

    df = df[df.primary_keyword.isin(CATEGORIES)].reset_index(drop=True)
    df['text'] = (df.position + ' ' + df.long_description).str.split().str.join(' ')
    return df


def split(df):
    groups = df.company_group.drop_duplicates()
    groups_full_train, groups_test = train_test_split(groups, test_size=0.2, random_state=1)
    groups_train, groups_val = train_test_split(groups_full_train, test_size=0.25, random_state=1)

    df_train = df[df.company_group.isin(groups_train)].reset_index(drop=True)
    df_val = df[df.company_group.isin(groups_val)].reset_index(drop=True)
    df_test = df[df.company_group.isin(groups_test)].reset_index(drop=True)
    return df_train, df_val, df_test


def train(df):
    dicts = df.text.apply(ad_words).tolist()
    counts = pd.Series([w for d in dicts for w in d]).value_counts()

    dv = DictVectorizer()
    dv.fit([{w: 1 for w in counts[counts >= MIN_ADS].index}])

    model = LogisticRegression(C=C, max_iter=1000, class_weight='balanced')
    model.fit(dv.transform(dicts), df.primary_keyword.values)
    return dv, model


def macro_f1(y_true, y_pred):
    f1_values = []
    for c in CATEGORIES:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_values.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return sum(f1_values) / len(f1_values)


def main():
    df_train, df_val, df_test = split(load_data())
    print(f'train={len(df_train)} val={len(df_val)} test={len(df_test)}')

    df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)
    dv, model = train(df_full_train)

    X_test = dv.transform(df_test.text.apply(ad_words).tolist())
    y_test = df_test.primary_keyword.values
    y_pred = model.predict(X_test)
    print(f'test macro F1: {macro_f1(y_test, y_pred):.4f}')
    print(f'test accuracy: {(y_pred == y_test).mean():.4f}')

    with open(MODEL_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f'model saved to {MODEL_FILE}')


if __name__ == '__main__':
    main()
