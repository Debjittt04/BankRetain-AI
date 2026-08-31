import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

df = pd.read_csv("data/churn_modelling.csv")

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)


# =========================================================
# 2. REMOVE UNNECESSARY COLUMNS
# =========================================================

df = df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname"
    ]
)

print("\nRemoved unnecessary columns:")
print("RowNumber, CustomerId, Surname")

print("\nDataset shape after removing columns:")
print(df.shape)


# =========================================================
# 3. CHECK TARGET
# =========================================================

print("\nTarget distribution:")

print(
    df["Exited"].value_counts()
)

print("\nTarget percentage:")

print(
    df["Exited"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(
    "Exited",
    axis=1
)

y = df["Exited"]


# =========================================================
# 5. DEFINE FEATURE TYPES
# =========================================================

categorical_features = [
    "Geography",
    "Gender"
]

numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


# =========================================================
# 6. CREATE PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# 8. DEFINE MACHINE LEARNING MODELS
# =========================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),


    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        ),


    "Gradient Boosting":

        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# =========================================================
# 9. TRAIN MODELS
# =========================================================

results = {}

trained_models = {}


for name, model in models.items():

    print("\n")
    print("=" * 60)
    print("TRAINING:", name)
    print("=" * 60)


    # Create pipeline
    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    # Train
    pipeline.fit(
        X_train,
        y_train
    )


    # Predictions
    y_pred = pipeline.predict(
        X_test
    )


    # Probability of churn
    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]


    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # Store results

    results[name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    }


    trained_models[name] = pipeline


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# =========================================================
# 10. COMPARE MODELS
# =========================================================

results_df = pd.DataFrame(
    results
).T


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.round(4)
)


# =========================================================
# 11. SELECT BEST MODEL
# =========================================================

best_model_name = (
    results_df[
        "ROC-AUC"
    ]
    .idxmax()
)


best_model = trained_models[
    best_model_name
]


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    best_model_name
)


# =========================================================
# 12. CREATE MODELS FOLDER
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)


# =========================================================
# 13. SAVE BEST MODEL
# =========================================================

model_path = (
    "models/churn_model.pkl"
)


joblib.dump(
    best_model,
    model_path
)


print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "Saved to:",
    model_path
)


print("\nTraining completed successfully! 🎉")