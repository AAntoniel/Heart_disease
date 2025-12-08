# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st

from sklearn import model_selection
from sklearn import tree
from sklearn import pipeline
from sklearn import ensemble
from sklearn import metrics

from feature_engine import discretisation, encoding

import os

# %%


def read_files(file_name: str):
    df = pd.read_csv(
        f"Data/database/{file_name}",
        header=None,
        na_values="?",
    )

    return df


dfs = []
file_names = os.listdir("Data/database")
for i in file_names:
    dfs.append(read_files(i))

# dfs[2]

df_full = pd.concat(dfs).reset_index(drop=True)
df_full.columns = [
    "age",
    "sex",
    "chest_pain",
    "restbps",
    "chol",
    "fast_blood_sug",
    "rest_electcard",
    "max_heart_rate",
    "exc_angina",
    "oldpeak",
    "slope",
    "n_fl_maj_ves",
    "thal",
    "heart_disease",
]

df_full

# df_full[df_full["restbps"].isna()].index.tolist()
# %%

# Data pre-processing

# Solving target column first
df_full["heart_disease"] = df_full["heart_disease"].apply(lambda x: 1 if x > 0 else 0)

df_full["heart_disease"].value_counts()

# Verifying 0s
print((df_full == 0).sum())

# %%

# Starting by replacing non-sense 0s by NaNs
df_full["chol"] = df_full["chol"].replace(0, np.nan)
df_full["restbps"] = df_full["restbps"].replace(0, np.nan)

print(df_full.isna().sum())

# %%

# Summary global

summary = df_full.groupby(by="heart_disease").agg(["mean", "median"]).T
summary["abs_diff"] = summary[0] - summary[1]
summary["diff_rel"] = summary[0] / summary[1]
summary

# %%

# Numerical features preprocessing

# Summary by male
summary_male = (
    df_full[df_full["sex"] == 1]
    .groupby(by="heart_disease")[["restbps", "chol", "max_heart_rate", "oldpeak"]]
    .agg(["mean", "median"])
    .T
)
summary_male["abs_diff"] = summary_male[0] - summary_male[1]
summary_male["diff_rel"] = summary_male[0] / summary_male[1]
summary_male

# %%

# Summary by female
summary_female = (
    df_full[df_full["sex"] == 0]
    .groupby(by="heart_disease")[["restbps", "chol", "max_heart_rate", "oldpeak"]]
    .agg(["mean", "median"])
    .T
)
summary_female["abs_diff"] = summary_female[0] - summary_female[1]
summary_female["diff_rel"] = summary_female[0] / summary_female[1]
summary_female

# %%

# Summary by sex, chest pain type and heart disease

summary_sex_cp = df_full.groupby(by=["sex", "chest_pain", "heart_disease"])[
    ["restbps", "chol", "max_heart_rate", "oldpeak"]
].agg(["mean", "median"])
summary_sex_cp

# %%

# Numerical data imputation
nums1 = ["restbps", "chol", "max_heart_rate", "oldpeak"]

for i in nums1:
    df_full[i] = df_full[i].fillna(
        df_full.groupby(by=["sex", "chest_pain", "heart_disease"])[i].transform(
            "median"
        )
    )

print(df_full.isna().sum())
# df_full.loc[[393,610,620,623,626,627,633,635,639,641,645,648,654,655,657,665,666,669,674,684,686,691,693,706,707,708,709,710,711,712,716,717,721,726,730,733,734,738,739,741,742,744,746,752,755,756,757,758,760,761,764,765,771,778,782,793,795,799,914]]

# %%
# Categorical features preprocessing

summary_sex_cp_cat = df_full.groupby(by=["sex", "chest_pain", "heart_disease"])[
    ["fast_blood_sug", "rest_electcard", "exc_angina", "slope", "n_fl_maj_ves", "thal"]
].agg(lambda x: x.mode())
summary_sex_cp_cat

# %%

# sex (1 = male; 0 = female)
# chest_pain: chest pain type
#         -- Value 1: typical angina
#         -- Value 2: atypical angina
#         -- Value 3: non-anginal pain
#         -- Value 4: asymptomatic
# fast_blood_sug: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
# rest_electcard: resting electrocardiographic results
#         -- Value 0: normal
#         -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#         -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# exc_angina: exercise induced angina (1 = yes; 0 = no)
# slope: the slope of the peak exercise ST segment
#         -- Value 1: upsloping
#         -- Value 2: flat
#         -- Value 3: downsloping
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

nums2 = [
    "fast_blood_sug",
    "rest_electcard",
    "exc_angina",
    "slope",
    "n_fl_maj_ves",
    "thal",
]

for i in nums2:
    # Only first value of mode is considered (ex: n_fl_maj_ves with [1.0, 2.0]). If the group is empty, return NaNs
    df_full[i] = df_full[i].fillna(
        df_full.groupby(["sex", "chest_pain", "heart_disease"])[i].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
    )

print(df_full.isna().sum())

# %%

df_full

df_full["n_fl_maj_ves"].min()

# %%

df_full[df_full.isna().any(axis=1)].index.tolist()

df_full = df_full.dropna(how="any").reset_index(drop=True)
df_full.isna().sum()

# %%

# Using SEMMA (SAS) approach
# Sample

features = df_full.columns[0:-1]
target = "heart_disease"

# We use stratify=y to ensure the train and test sets have the same proportion
X, y = df_full[features], df_full[target]
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

# Checking the distribution of the target variable in train and test sets
# w/o stratify -> train 54.4% vs test 60.1%
# with stratify -> train 55% vs test 55%

print(f"Overall target rate {y.mean():.4f}")
print(f"Train target rate {y_train.mean():.4f}")
print(f"Test target rate {y_test.mean():.4f}")

# %%

# Explore

# feature_importances
tree_fi = tree.DecisionTreeClassifier(random_state=42)
tree_fi.fit(X_train, y_train)

# Cumsum to capture the importances
feature_importances = (
    pd.Series(tree_fi.feature_importances_, index=X_train.columns)
    .sort_values(ascending=False)
    .reset_index()
)
feature_importances["acum"] = feature_importances[0].cumsum()

feature_importances

# %%

best_features = feature_importances[feature_importances["acum"] < 1]["index"].to_list()
best_features

#  To make it easier to user, some technical features will be excluded
remove = [
    "thal",
    "n_fl_maj_ves",
    "oldpeak",
    "slope",
    "rest_electcard",
    "fast_blood_sug",
]

best_features = [i for i in best_features if i not in remove]
best_features

# %%

# With the best features, it's possible to continue with the model, or we can modify it, and that's what we are going to do

# Modify and Modeling

# Discretisation

tree_disc = discretisation.DecisionTreeDiscretiser(
    variables=["chol", "max_heart_rate", "restbps", "age"],
    regression=False,
    bin_output="bin_number",
    cv=3,
)

onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

model = ensemble.RandomForestClassifier(
    random_state=42,
    n_jobs=2,
)

params = {
    "min_samples_leaf": [10, 20, 25, 30],
    "n_estimators": [100, 300, 500],
    "criterion": ["gini", "entropy", "log_loss"],
}

grid_search = model_selection.GridSearchCV(
    model, params, cv=3, scoring="roc_auc", verbose=4
)

model_pipeline = pipeline.Pipeline(
    steps=[("Discretize", tree_disc), ("Onehot", onehot), ("Grid", grid_search)]
)

model_pipeline.fit(X_train[best_features], y_train)

# %%

# Assess

y_pred = model_pipeline.predict(X_train[best_features])
y_pred_proba = model_pipeline.predict_proba(X_train[best_features])[:, 1]

acc_train = metrics.accuracy_score(y_train, y_pred)
auc_train = metrics.roc_auc_score(y_train, y_pred_proba)
roc_train = metrics.roc_curve(y_train, y_pred_proba)
recall_train = metrics.recall_score(y_train, y_pred)
precision_train = metrics.precision_score(y_train, y_pred)

print("Train accuracy: ", acc_train)
print("Train AUC: ", auc_train)
print("Train Recall: ", recall_train)
print("Train Precision: ", precision_train)

print("-" * 20)

y_pred_test = model_pipeline.predict(X_test[best_features])
y_pred_proba_test = model_pipeline.predict_proba(X_test[best_features])[:, 1]

acc_test = metrics.accuracy_score(y_test, y_pred_test)
auc_test = metrics.roc_auc_score(y_test, y_pred_proba_test)
roc_test = metrics.roc_curve(y_test, y_pred_proba_test)
recall_test = metrics.recall_score(y_test, y_pred_test)
precision_test = metrics.precision_score(y_test, y_pred_test)

print("Test accuracy: ", acc_test)
print("Test AUC: ", auc_test)
print("Test Recall: ", recall_test)
print("Test Precision: ", precision_test)

plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.grid(True)
plt.title("Curva Roc")
plt.legend(
    [
        f"Treino: {100*auc_train: .2f}",
        f"Teste: {100*auc_test: .2f}",
    ]
)

# %%

pd.Series({"model": model_pipeline, "features": best_features}).to_pickle(
    "model_heart_dis.pkl"
)
