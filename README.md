# Job Category Classifier

ML-project for ML Zoomcamp midterm. Given the job ad, try to determine to which of 6 job board categories this ad should be assigned.

## Problem description

On the job board, all the ads are grouped by categories (e.g. Java or DevOps) so the candidates could easily locate it. Someone has to define the category of each new job ad. When there are thousands of ads every month, it is a cumbersome process and could lead to the mistakes that prevent candidate from seeing the ad.

For this project we create model, which would read the title and text of the ad and suggest a category. An editor who reviews new ads can agree with the model's suggestion or change it. This model is designed solely for the job ad category assignment. The model does not assess candidates and is not involved in any hiring process.

## Data

The data used for the project is a public dataset of English job ads from the Djinni (job board for Ukrainian IT market) for the period from Jan 2020 to Sep 2023. It is hosted on Hugging Face under the name `lang-uk/recruitment-dataset-job-descriptions-english`. It contains 141 897 ads across 45 categories.

We use 6 biggest technical categories: JavaScript, Java, DevOps, QA Automation, Python, Data Science. It is 49 638 ads in total. The dataset is unbalanced. JavaScript has 17 903 ads. Data Science only has 2 262 ads.

Model makes prediction for column `Primary Keyword` which is a category that job board assigned. Only `Position` (ad title) and `Long Description` (text) from the ad are used.

## Model evaluation

We use main metric - macro F1. We calculate F1 for each category and take the mean of 6 values. So the small category Data Science has the same weight as big category JavaScript. We split ads by company so the model is always evaluated on the companies unseen during training.

## Restrictions

- model knows only about 6 categories. If the ad is in some different field, it is still assigned to one of these 6 categories.
- job board categories may overlap. E.g. sometimes Data Science ads contain data engineering jobs
- all ads are from Ukraine 2020-2023. The other markets/years could be different.
