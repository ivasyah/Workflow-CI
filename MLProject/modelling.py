import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow config untuk CI (tanpa ngrok / localhost)
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("Loan Prediction CI")

def main():
    df = pd.read_csv("loan_prediction_preprocessed.csv")

    X = df.drop(columns=["loan_approved"])
    y = df["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
