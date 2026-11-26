"""
This script aims to train an XGBoost model to predict estrogen receptor status based on metabolomic data. 
The model will be trained using a labeled dataset, and its performance will be evaluated using cross-validation techniques. 
The script includes data preprocessing, model training, hyperparameter tuning, and evaluation metrics to assess the model's 
accuracy and robustness.
"""

from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from functools import partial
from shutil import rmtree
from pathlib import Path
import argparse as ap
import joblib as jb
import pandas as pd
import numpy as np
import json as js
import os

__author__ = "Pablo Salazar-Méndez"
NTHREAD = int(os.environ.get("NSLOTS", 10))

# ============== FUNCTIONS ==============
def generate_df(file_path: list[Path], sep: str = ",", index_col: int | None = None) -> list[pd.DataFrame]:
    if not isinstance(file_path, list):
        raise ValueError("File path must be a list of Path objects.")
    if not all(isinstance(f, Path) for f in file_path):
        raise ValueError("All file paths must be Path objects.")
    if not all(f.is_file() for f in file_path):
        raise FileNotFoundError("One or more input files do not exist.")
    return [pd.read_csv(f, sep=sep, index_col=index_col) for f in file_path]

def generate_np(file_path: list[Path], dtype = np.float64) -> np.ndarray:
    if not isinstance(file_path, list):
            raise ValueError("File path must be a list of Path objects.")
    if not all(isinstance(f, Path) for f in file_path):
        raise ValueError("All file paths must be Path objects.")
    if not all(f.is_file() for f in file_path):
        raise FileNotFoundError("One or more input files do not exist.")
    return [np.loadtxt(f, dtype=dtype) for f in file_path]

def main():
    data = Path("./data")
    results = Path("./results")
    if not data.is_dir():
        data.mkdir(parents=True, exist_ok=True)
    if not results.is_dir():
        results.mkdir(parents=True, exist_ok=True)
    if not (results / "feature_selection").is_dir():
        (results / "feature_selection").mkdir(parents=True, exist_ok=True)

    # FULL DATASET
    F_train, f_test = generate_df([
        Path(data / "EstrogenReceptorStatus_Train.csv"), Path(data / "EstrogenReceptorStatus_Test.csv")
    ], index_col=0)
    y_train, y_test = generate_np([
        Path(data / "EstrogenReceptorStatus_Train_labels.txt"), Path(data / "EstrogenReceptorStatus_Test_labels.txt")
    ])

    percentile = 20
    selectors = {
        "Full": "passthrough",
        "ANOVA": SelectPercentile(score_func=f_classif, percentile=percentile),
        "MI": SelectPercentile(score_func=partial(mutual_info_classif, random_state=42), percentile=percentile)
    }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist", # Faster histogram optimized approximate greedy algorithm (taken from offical documentation)
        random_state=42,
        n_jobs=1,
        subsample=0.8
    )
    parameters = {
        "n_estimators": [128, 256],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_child_weight": [1, 5],
        "scale_pos_weight": [1, (y_train == 0).sum() / (y_train == 1).sum()],
        "reg_lambda": [1, 10]
    }
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for name, selector in selectors.items():
        
        print(f"Training XGBoost model on {name} dataset...")
        output = results / "xgboost" / name.lower()
        output.mkdir(parents=True, exist_ok=True)

        pipeline = Pipeline(
            [("selection", selector), ("model", model)],
            memory=str(output / "cache") if name == "MI" else None
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid={f"model__{key}": values for key, values in parameters.items()},
            scoring="f1_macro",
            cv=cv,
            n_jobs=NTHREAD,
            pre_dispatch=NTHREAD, # default 2*n_jobs
            verbose=1,
            return_train_score=True
        )

        # Train and select the best model
        grid.fit(F_train, y_train)
        best_model = grid.best_estimator_

        # Predict on test data
        y_pred = best_model.predict(f_test)
        y_prob = best_model.predict_proba(f_test)[:, 1]

        # Save fitted model
        jb.dump(best_model, output / f"model_{name}.joblib")
        best_model.named_steps["model"].save_model(str(output / f"model_{name}.json"))

        # Save selected metabolites and their XGBoost importance
        if name == "Full":
            features = F_train.columns
        else:
            features = best_model.named_steps["selection"].get_feature_names_out()

        pd.DataFrame({
            "metabolite": features,
            "importance": best_model.named_steps["model"].feature_importances_
        }).sort_values(
            "importance", ascending=False
        ).to_csv(
            output / f"feature_importance_{name}.csv",
            index=False
        )

        # Save cross-validation results
        pd.DataFrame(grid.cv_results_).to_csv(
            output / f"grid_results_{name}.csv", index=False
        )

        # Save best hyperparameters
        with open(output / f"best_parameters_{name}.json", "w") as f:
            js.dump({
                "parameters": grid.best_params_,
                "cv_f1_macro": float(grid.best_score_)
            }, f, indent=4)

        # Save predictions
        pd.DataFrame({
            "sample": f_test.index,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob
        }).to_csv(output / f"predictions_{name}.csv", index=False)

        # Save classification report
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        pd.DataFrame(report).T.to_csv(
            output / f"classification_report_{name}.csv"
        )

        # Save global metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        with open(output / f"metrics_{name}.json", "w") as f:
            js.dump(metrics, f, indent=4)

        print(f"Best CV F1 macro: {grid.best_score_:.4f}")
        print(f"Test F1 macro: {metrics['f1_macro']:.4f}")
        print(f"Results saved in {output}")

        # Deleting cache directory if it exists
        if (cache_dir := (output / "cache")).is_dir():
            rmtree(cache_dir)
            print(f"Cache directory {cache_dir} deleted.")

if __name__ == "__main__":
    main()