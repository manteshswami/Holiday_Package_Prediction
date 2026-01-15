import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib


# ================================
# Load Data
# ================================
def load_data():
    return pd.read_csv("Travel.csv")


# ================================
# Data Cleaning
# ================================
def clean_data(df):
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["TypeofContact"].fillna(df["TypeofContact"].mode()[0], inplace=True)
    df["DurationOfPitch"].fillna(df["DurationOfPitch"].median(), inplace=True)
    df["NumberOfFollowups"].fillna(df["NumberOfFollowups"].mode()[0], inplace=True)
    df["PreferredPropertyStar"].fillna(df["PreferredPropertyStar"].mode()[0], inplace=True)
    df["NumberOfTrips"].fillna(df["NumberOfTrips"].median(), inplace=True)
    df["NumberOfChildrenVisiting"].fillna(df["NumberOfChildrenVisiting"].mode()[0], inplace=True)
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)

    df.drop("CustomerID", axis=1, inplace=True)

    return df


# ================================
# Feature Engineering
# ================================
def feature_engineering(df):
    df["TotalVisiting"] = df["NumberOfChildrenVisiting"] + df["NumberOfPersonVisiting"]
    df.drop(["NumberOfChildrenVisiting", "NumberOfPersonVisiting"], axis=1, inplace=True)
    return df


# ================================
# Train Test Split + Preprocessing
# ================================
def preprocess(df):
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    cat_features = X.select_dtypes(include="object").columns
    num_features = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("OneHotEncoder", OneHotEncoder(drop="first"), cat_features),
        ("StandardScaler", StandardScaler(), num_features)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor


# ================================
# Model Training
# ================================
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=2,
        max_features=8,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


# ================================
# Model Evaluation
# ================================
def evaluate(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))
    print("Testing Accuracy:", accuracy_score(y_test, test_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, test_pred))

    roc = roc_auc_score(y_test, test_pred)
    print("ROC-AUC Score:", roc)


# ================================
# Save Model
# ================================
def save_model(model, preprocessor):
    joblib.dump(model, "holiday_package_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("\nModel & Preprocessor saved successfully.")


# ================================
# Main Pipeline
# ================================
def main():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

    model = train_model(X_train, y_train)
    evaluate(model, X_train, X_test, y_train, y_test)

    save_model(model, preprocessor)


if __name__ == "__main__":
    main()
