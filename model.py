## Model Training
##
## Trains Logistic Regression, Random Forest, and XGBoost models on the cleaned dataset
## Includes train-test split, feature scaling, SMOTE balancing, evaluation metrics,
## and saves a CSV of model performance metrics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load cleaned dataset
df = pd.read_csv("cleaned_order_dataset.csv")

#   1. FEATURES WITHOUT LEAKAGE
# Features known BEFORE a return happens
features = [
    "Product_Return_Rate",
    "Product_Popularity",
    "Customer_Return_Rate",
    "Customer_Order_Count",
    "Category_Return_Rate",
    "Price Reductions",
    "Discount_Rate",
    "Has_Discount",
    "Sales Tax",
    "Tax_Rate",
    "Order_Month",
    "Order_DayOfWeek",
    "Order_Quarter",
    "Is_Weekend"
]

X = df[features]
y = df["Is_Returned"]

#   2. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#   3. SMOTE FOR BALANCING
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_bal.value_counts())

#   4. SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

#   5. MODELS WITHOUT LEAKAGE
# Logistic Regression
metrics = {}
lr = LogisticRegression(max_iter=1500, class_weight="balanced")
lr.fit(X_train_scaled, y_train_bal)
lr_pred = lr.predict(X_test_scaled)

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

metrics["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, lr_pred),
    "precision": precision_score(y_test, lr_pred),
    "recall": recall_score(y_test, lr_pred),
    "f1": f1_score(y_test, lr_pred)
}

# Random Forest
rf = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train_bal, y_train_bal)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred))

metrics["Random Forest"] = {
    "accuracy": accuracy_score(y_test, rf_pred),
    "precision": precision_score(y_test, rf_pred),
    "recall": recall_score(y_test, rf_pred),
    "f1": f1_score(y_test, rf_pred)
}

# XGBoost
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    scale_pos_weight=imbalance_ratio
)

xgb_model.fit(X_train_bal, y_train_bal)
xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Report:")
print(classification_report(y_test, xgb_pred))

metrics["XGBoost"] = {
    "accuracy": accuracy_score(y_test, xgb_pred),
    "precision": precision_score(y_test, xgb_pred),
    "recall": recall_score(y_test, xgb_pred),
    "f1": f1_score(y_test, xgb_pred)
}

#   6. SAVE METRICS
pd.DataFrame(metrics).to_csv("model_metrics.csv")
print("\nFinal metrics saved to model_metrics.csv")
