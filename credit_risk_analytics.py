import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================================
# CREDIT RISK ANALYTICS - LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DATA IMPORT
# ------------------------------------------------------------

bankloans = pd.read_csv("bankloans.csv")

print(bankloans.head())


# ------------------------------------------------------------
# 2. USER DEFINED FUNCTIONS
# ------------------------------------------------------------

def continuous_var_summary(x):

    n_total = x.shape[0]
    n_miss = x.isna().sum()
    perc_miss = n_miss * 100 / n_total

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    iqr = q3 - q1

    lc_iqr = q1 - 1.5 * iqr
    uc_iqr = q3 + 1.5 * iqr

    return pd.Series(
        [
            x.dtype,
            x.nunique(),
            n_total,
            x.count(),
            n_miss,
            perc_miss,
            x.sum(),
            x.mean(),
            x.std(),
            x.var(),
            lc_iqr,
            uc_iqr,
            x.min(),
            x.quantile(0.01),
            x.quantile(0.05),
            x.quantile(0.10),
            x.quantile(0.25),
            x.quantile(0.50),
            x.quantile(0.75),
            x.quantile(0.90),
            x.quantile(0.95),
            x.quantile(0.99),
            x.max()
        ],
        index=[
            "dtype",
            "cardinality",
            "n_tot",
            "n",
            "nmiss",
            "perc_miss",
            "sum",
            "mean",
            "std",
            "var",
            "lc_iqr",
            "uc_iqr",
            "min",
            "p1",
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "max"
        ]
    )


def categorical_var_summary(x):

    mode = (
        x.value_counts()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )

    return pd.Series(
        [
            x.count(),
            x.isnull().sum(),
            mode.iloc[0, 0],
            mode.iloc[0, 1],
            round(mode.iloc[0, 1] * 100 / x.count(), 2)
        ],
        index=[
            "N",
            "NMISS",
            "MODE",
            "FREQ",
            "PERCENT"
        ]
    )


def missing_imputation(x, stats="mean"):

    if (x.dtypes == "float64") | (x.dtypes == "int64"):

        if stats == "mean":
            x = x.fillna(x.mean())
        else:
            x = x.fillna(x.median())

    return x


# ------------------------------------------------------------
# 3. DATA UNDERSTANDING
# ------------------------------------------------------------

bankloans.info()

print("Number of default observations:",
      bankloans.default.count())

print(
    "Default cardinality:",
    bankloans.default.nunique()
)

print(
    "Default class distribution:"
)

print(
    bankloans.default.value_counts()
    / bankloans.default.count()
)


# ------------------------------------------------------------
# 4. SEPARATE EXISTING AND NEW CUSTOMERS
# ------------------------------------------------------------

bankloans_existing = bankloans.loc[
    bankloans.default.notna()
]

bankloans_new = bankloans.loc[
    bankloans.default.isna()
]

print(
    "Existing customers shape:",
    bankloans_existing.shape
)

print(
    "New customers shape:",
    bankloans_new.shape
)


# ------------------------------------------------------------
# 5. EDA
# ------------------------------------------------------------

print(
    bankloans_existing.apply(
        continuous_var_summary
    )
)


# ------------------------------------------------------------
# 6. DATA CLEANING
# ------------------------------------------------------------

# Outlier treatment

bankloans_existing = bankloans_existing.apply(
    lambda x: x.clip(
        lower=x.quantile(0.01),
        upper=x.quantile(0.99)
    )
)


# ------------------------------------------------------------
# 7. CORRELATION ANALYSIS
# ------------------------------------------------------------

corr_matrix = bankloans_existing.corr()

print(corr_matrix)

plt.figure(figsize=(15, 9))

sns.heatmap(
    corr_matrix,
    annot=True
)

plt.title("Correlation Matrix")

plt.show()


# ------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# ------------------------------------------------------------

train, test = train_test_split(
    bankloans_existing,
    test_size=0.3,
    random_state=42
)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("Train columns:")
print(train.columns)


# ------------------------------------------------------------
# 9. MODEL 1 - ALL FEATURES
# ------------------------------------------------------------

model_eq = (
    "default ~ "
    + " + ".join(
        train.columns.difference(["default"])
    )
)

print(model_eq)

m1 = smf.logit(
    formula=model_eq,
    data=train
).fit()

print(m1.summary2())


# ------------------------------------------------------------
# 10. SOMERS' D - VARIABLE REDUCTION
# ------------------------------------------------------------

somarsd_score = pd.DataFrame()

for var_name in bankloans_existing.columns.difference(
    ["default"]
):

    log_mod = smf.logit(
        formula="default ~ " + var_name,
        data=bankloans_existing
    ).fit(disp=False)

    y_predicted_proba = log_mod.predict(
        bankloans_existing
    )

    concordance = roc_auc_score(
        bankloans_existing.default,
        y_predicted_proba
    )

    somars_d = 2 * concordance - 1

    temp_data = pd.DataFrame(
        [
            var_name,
            concordance,
            somars_d
        ]
    ).T

    somarsd_score = pd.concat(
        [
            somarsd_score,
            temp_data
        ],
        axis=0
    )


somarsd_score.columns = [
    "variable",
    "roc_auc_score",
    "somars_d"
]

print("Somers' D results:")

print(somarsd_score)


# Keep variables with Somers' D >= 0.2

features = list(
    somarsd_score.loc[
        somarsd_score.somars_d >= 0.2,
        "variable"
    ]
)

print("Selected features:")

print(features)


# ------------------------------------------------------------
# 11. MULTICOLLINEARITY CHECK - VIF
# ------------------------------------------------------------

equation = (
    "default ~ "
    + " + ".join(features)
)

a, b = dmatrices(
    equation,
    data=bankloans_existing,
    return_type="dataframe"
)

vif = pd.DataFrame()

vif["features"] = b.columns

vif["VIF Factor"] = [
    variance_inflation_factor(
        b.values,
        i
    )
    for i in range(b.shape[1])
]

print("VIF results:")

print(vif)


# ------------------------------------------------------------
# 12. MODEL 2 - AFTER VARIABLE REDUCTION
# ------------------------------------------------------------

model_eq = (
    "default ~ "
    + " + ".join(features)
)

m2 = smf.logit(
    model_eq,
    data=train
).fit()

print(m2.summary2())


# ------------------------------------------------------------
# 13. MODEL 3 - FINAL MODEL
# ------------------------------------------------------------

features = [
    "address",
    "creddebt",
    "debtinc",
    "employ"
]

model_eq = (
    "default ~ "
    + " + ".join(features)
)

m3 = smf.logit(
    model_eq,
    data=train
).fit()

print(m3.summary2())


# ------------------------------------------------------------
# 14. PREDICT PROBABILITIES
# ------------------------------------------------------------

train_predict = m3.predict(train)

test_predict = m3.predict(test)

print("Train predicted probabilities:")

print(train_predict.head())

print("Test predicted probabilities:")

print(test_predict.head())


# ------------------------------------------------------------
# 15. MODEL SCORING - AUC
# ------------------------------------------------------------

train_auc = roc_auc_score(
    train.default,
    train_predict
)

test_auc = roc_auc_score(
    test.default,
    test_predict
)

print(
    "AUC for Train Data =",
    train_auc
)

print(
    "AUC for Test Data =",
    test_auc
)


# ------------------------------------------------------------
# 16. ACTUAL VS PREDICTED PROBABILITY
# ------------------------------------------------------------

train_predicted_prob = pd.DataFrame(
    train_predict
)

train_pf = pd.concat(
    [
        train.default,
        train_predicted_prob
    ],
    axis=1
).reset_index(drop=True)

train_pf.columns = [
    "actual",
    "prob"
]


test_predicted_prob = pd.DataFrame(
    test_predict
)

test_pf = pd.concat(
    [
        test["default"],
        test_predicted_prob
    ],
    axis=1
).reset_index(drop=True)

test_pf.columns = [
    "actual",
    "prob"
]


# ------------------------------------------------------------
# 17. FIND BEST CUTOFF
# ------------------------------------------------------------

df_best_cutoff = pd.DataFrame()

for iproba in np.arange(
    0,
    1.01,
    0.01
):

    y_predicted = pd.Series(
        np.where(
            train_pf.prob >= iproba,
            1,
            0
        )
    )

    cm = confusion_matrix(
        train_pf.actual,
        y_predicted
    )[::-1, ::-1]

    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[1, 0]
    fn = cm[0, 1]

    row_tots = cm.sum(axis=1)

    tpr = tp / row_tots[0]

    fpr = fp / row_tots[1]

    specificity = 1 - fpr

    accuracy = (
        (tp + tn)
        / cm.sum()
    )

    sen_spec = (
        tpr
        + specificity
    )

    temp_df = pd.DataFrame(
        [
            iproba,
            tp,
            tn,
            fp,
            fn,
            tpr,
            fpr,
            specificity,
            accuracy,
            sen_spec
        ]
    ).T

    df_best_cutoff = pd.concat(
        [
            df_best_cutoff,
            temp_df
        ],
        axis=0
    )


df_best_cutoff.columns = [
    "proba",
    "tp",
    "tn",
    "fp",
    "fn",
    "tpr",
    "fpr",
    "specificity",
    "accuracy",
    "sen_spec"
]

df_best_cutoff = (
    df_best_cutoff
    .reset_index(drop=True)
)


print("Best cutoff analysis:")

print(df_best_cutoff)


# ------------------------------------------------------------
# 18. SELECT BEST CUTOFF
# ------------------------------------------------------------

best_cutoff = df_best_cutoff.loc[
    df_best_cutoff.sen_spec
    == df_best_cutoff.sen_spec.max(),
    "proba"
]

best_cutoff = best_cutoff.iloc[0]

print(
    "Best cutoff =",
    best_cutoff
)


# ------------------------------------------------------------
# 19. FINAL CLASSIFICATION
# ------------------------------------------------------------

train_pf["predicted"] = pd.Series(
    np.where(
        train_pf.prob >= best_cutoff,
        1,
        0
    )
)

test_pf["predicted"] = pd.Series(
    np.where(
        test_pf.prob >= best_cutoff,
        1,
        0
    )
)


# ------------------------------------------------------------
# 20. ACCURACY
# ------------------------------------------------------------

train_accuracy = accuracy_score(
    train_pf.actual,
    train_pf.predicted
)

test_accuracy = accuracy_score(
    test_pf.actual,
    test_pf.predicted
)

print(
    "Overall accuracy for Train Data =",
    train_accuracy
)

print(
    "Overall accuracy for Test Data =",
    test_accuracy
)


# ------------------------------------------------------------
# 21. CONFUSION MATRIX - TRAIN
# ------------------------------------------------------------

print("Train Confusion Matrix:")

print(
    confusion_matrix(
        train_pf.actual,
        train_pf.predicted
    )
)


print("Train Classification Report:")

print(
    classification_report(
        train_pf.actual,
        train_pf.predicted
    )
)


# ------------------------------------------------------------
# 22. DEFAULT 0.5 CUTOFF COMPARISON
# ------------------------------------------------------------

y_pred = pd.Series(
    np.where(
        train_pf.prob >= 0.5,
        1,
        0
    )
)

print(
    "Classification Report at 0.5 cutoff:"
)

print(
    classification_report(
        train_pf.actual,
        y_pred
    )
)


# ------------------------------------------------------------
# 23. CONFUSION MATRIX - TEST
# ------------------------------------------------------------

print("Test Confusion Matrix:")

print(
    confusion_matrix(
        test_pf.actual,
        test_pf.predicted
    )
)


print("Test Classification Report:")

print(
    classification_report(
        test_pf.actual,
        test_pf.predicted
    )
)


# ------------------------------------------------------------
# 24. DECILE ANALYSIS
# ------------------------------------------------------------

train_pf["Deciles"] = pd.qcut(
    train_pf.prob,
    10,
    labels=False
)

test_pf["Deciles"] = pd.qcut(
    test_pf.prob,
    10,
    labels=False
)


train_deciles = (
    train_pf
    .groupby("Deciles")[
        ["prob", "actual"]
    ]
    .agg(
        {
            "prob": [np.min, np.max],
            "actual": [np.sum, "count"]
        }
    )
    .reset_index()
    .sort_values(
        by="Deciles",
        ascending=False
    )
)


test_deciles = (
    test_pf
    .groupby("Deciles")[
        ["prob", "actual"]
    ]
    .agg(
        {
            "prob": [np.min, np.max],
            "actual": [np.sum, "count"]
        }
    )
    .reset_index()
    .sort_values(
        by="Deciles",
        ascending=False
    )
)


print("Train Decile Analysis:")

print(train_deciles)


print("Test Decile Analysis:")

print(test_deciles)


# ------------------------------------------------------------
# 25. EXPORT DECILE ANALYSIS
# ------------------------------------------------------------

train_deciles.to_csv(
    "train_deciles.csv",
    index=False
)

test_deciles.to_csv(
    "test_deciles.csv",
    index=False
)


# ------------------------------------------------------------
# 26. PREDICT NEW CUSTOMERS
# ------------------------------------------------------------

print("New customers:")

print(bankloans_new.head())


bankloans_new.loc[:, "prob"] = (
    m3.predict(bankloans_new)
)


bankloans_new.loc[:, "default"] = (
    bankloans_new["prob"]
    .apply(
        lambda x:
        1 if x > best_cutoff else 0
    )
)


print(
    "New customer default prediction:"
)

print(
    bankloans_new.default.value_counts()
)


print(
    "New customers sorted by risk:"
)

print(
    bankloans_new.sort_values(
        "prob",
        ascending=False
    )
)


# ------------------------------------------------------------
# 27. PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

sns.histplot(
    train_pf.loc[
        train_pf.actual == 0,
        "prob"
    ],
    kde=True,
    label="Non-default"
)

sns.histplot(
    train_pf.loc[
        train_pf.actual == 1,
        "prob"
    ],
    kde=True,
    label="Default"
)

plt.title(
    "Predicted Probability Distribution"
)

plt.xlabel(
    "Probability of Default"
)

plt.legend()

plt.show()


# ------------------------------------------------------------
# END OF CREDIT RISK ANALYTICS
# ------------------------------------------------------------
