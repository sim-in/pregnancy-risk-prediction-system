import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("dataset_1.csv")


df = df.dropna(subset=["Risk Level"])


X = df.drop("Risk Level", axis=1)
y = df["Risk Level"].map({"High": 1, "Low": 0})


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)


def create_pipeline(model):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return pipeline


experiments = [
    ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)),

    ("Decision Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=None, random_state=42)),

    ("Logistic Regression", LogisticRegression(C=0.1, max_iter=1000, solver="liblinear")),
    ("Logistic Regression", LogisticRegression(C=1, max_iter=1000, solver="liblinear")),
    ("Logistic Regression", LogisticRegression(C=10, max_iter=1000, solver="liblinear"))
]


best_model = None
best_model_name = ""
best_score = 0


# Compare models using validation data
for model_name, model in experiments:
    pipeline = create_pipeline(model)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(y_valid, predictions, zero_division=0)
    recall = recall_score(y_valid, predictions, zero_division=0)
    f1 = f1_score(y_valid, predictions, zero_division=0)

    overall_score = (
        accuracy * 0.20 +
        precision * 0.20 +
        recall * 0.35 +
        f1 * 0.25
    )

    print("\nModel:", model_name)
    print("Model Details:", model)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Overall Score:", overall_score)

    print("\nClassification Report:")
    print(classification_report(y_valid, predictions, zero_division=0))

    if overall_score > best_score:
        best_score = overall_score
        best_model = pipeline
        best_model_name = model_name
        best_model_details = model


print("\nBest Model Selected:", best_model_details)
print("Best Validation Overall Score:", best_score)


test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, zero_division=0)
test_recall = recall_score(y_test, test_predictions, zero_division=0)
test_f1 = f1_score(y_test, test_predictions, zero_division=0)

print("\nFinal Test Result")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)

print("\nFinal Classification Report:")
print(classification_report(y_test, test_predictions, zero_division=0))


joblib.dump(best_model, "best_model_dataset_1.pkl", compress=3)

print("\nModel saved successfully as best_model_dataset_1.pkl")