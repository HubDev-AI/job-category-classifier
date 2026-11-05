#!/usr/bin/env python

import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from features import tokenize


DATA_FILE = 'data/raw/djinni-jobs-en.parquet'
MODEL_FILE = 'model.bin'
CATEGORIES = ['JavaScript', 'Java', 'DevOps', 'QA Automation', 'Python', 'Data Science']
MIN_ADS = 5
C = 0.01


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
    pipeline = make_pipeline(
        CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, binary=True, min_df=MIN_ADS),
        LogisticRegression(C=C, max_iter=1000, class_weight='balanced'),
    )
    return pipeline.fit(df.text, df.primary_keyword.values)


def main():
    df_train, df_val, df_test = split(load_data())
    print(f'train={len(df_train)} val={len(df_val)} test={len(df_test)}')

    df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)
    pipeline = train(df_full_train)

    y_test = df_test.primary_keyword.values
    y_pred = pipeline.predict(df_test.text)
    print(f'test macro F1: {f1_score(y_test, y_pred, average="macro"):.4f}')
    print(f'test accuracy: {accuracy_score(y_test, y_pred):.4f}')

    with open(MODEL_FILE, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    print(f'model saved to {MODEL_FILE}')


if __name__ == '__main__':
    main()
