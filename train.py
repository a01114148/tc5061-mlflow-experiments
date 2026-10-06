import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop the 'duration' column as it is not useful for prediction
    df = df.drop(columns=["duration"])

    # Convert categorical "yes"/"no" columns to binary
    binary_cols = ["default", "housing", "loan"]
    for col in binary_cols:
        df[col] = (df[col] == "yes").astype(int)

    # Impute missing values
    df["poutcome"] = df["poutcome"].fillna("none")
    other_cols = ["job", "education", "contact"]
    df[other_cols] = df[other_cols].fillna("unknown")

    # Create new column: "previously_contacted" and drop pdays
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    df = df.drop(columns=["pdays"])

    return df

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    X = df.drop(columns=[target])
    y = df[target]
    if stratify:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_preprocessor(power_features: list[str], scale_features: list[str], onehot_features: list[str]):
    preprocessor = ColumnTransformer(
        transformers = [
            ("power", PowerTransformer(method="yeo-johnson"), power_features),
            ("scale", StandardScaler(), scale_features),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), onehot_features),
        ],
        remainder="passthrough"
    )
    return preprocessor

def build_model_pipeline(
        preprocessor,
        model_type: str,
        model_params: dict,
):
    if model_type == "logistic_regression":
        model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return full_pipeline


def train_model(
        data_path: str,
        target: str,
        model_type: str,
        params: dict,
        experiment_name: str,
        run_name: str | None = None,
):
    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df, target=target)

    preprocessor = build_preprocessor(
        power_features=["balance", "campaign", "previous"],
        scale_features=["age", "day_of_month"],
        onehot_features=["job", "marital", "education", "contact", "poutcome", "month"]
    )
    model_pipeline = build_model_pipeline(preprocessor, model_type, params)


    with mlflow.start_run(run_name=run_name):
        # Log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        model_pipeline.fit(X_train, y_train)

        signature = infer_signature(X_train, model_pipeline.predict(X_train))

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name="bank_marketing_model"
        )

        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "recall_yes": recall_score(y_test, y_pred, pos_label=1),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # Log confusion matrix csv as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["actual_no", "actual_yes"],
            columns=["pred_no", "pred_yes"]
        )
        cm_filename = f"confusion_matrix_{run_name if run_name else 'run'}.csv"
        cm_df.to_csv(cm_filename, index=True)
        mlflow.log_artifact(cm_filename)


        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    return model_pipeline, {"X_test": X_test, "y_test": y_test}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an ML model with MLflow tracking.")

    # Data
    parser.add_argument("--data-path", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--target-column", type=str, default="target", help="Target column name")

    # Model / Hyperparameters
    parser.add_argument("--model-type", type=str, default="logistic_regression",
                        choices=["logistic_regression"])
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for logistic regression")
    parser.add_argument("--l1-ratio", type=float, default=0, help="The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=1 corresponds to L1 penalty, l1_ratio=0 to L2.")
    parser.add_argument("--class-weight", type=str, default=None, help="Class weight for logistic regression. Use 'balanced' or None.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum number of iterations for the solver to converge.")

    # MLFlow
    parser.add_argument("--experiment-name", type=str, default="bank-marketing-predictor", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_params = {
        "C": args.C,
        "l1_ratio": args.l1_ratio,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "solver": "saga"
    }

    model, data = train_model(
        data_path=args.data_path,
        target=args.target_column,
        model_type=args.model_type,
        params=model_params,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )