import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

search_space = hp.choice(
    "classifier_type",
    [
        {
            "type": "dt",
            "criterion": hp.choice("dtree_criterion", ["gini", "entropy"]),
            "max_depth": hp.choice(
                "dtree_max_depth",
                [None, hp.randint("dtree_max_depth_int", 1, 10)],
            ),
            "min_samples_split": hp.randint("dtree_min_samples_split", 2, 10),
        },
        {
            "type": "rf",
            "n_estimators": hp.randint("rf_n_estimators", 20, 500),
            "max_features": hp.randint("rf_max_features", 2, 9),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        },
        {
            "type": "gb",
            "loss": hp.choice("gb_loss", ["log_loss"]),
            "learning_rate": hp.uniform("gb_learning_rate", 0.05, 2),
            "n_estimators": hp.randint("gb_n_estimators", 20, 500),
            "subsample": hp.uniform("gb_subsample", 0.1, 1),
            "criterion": hp.choice(
                "gb_criterion", ["friedman_mse", "squared_error"]
            ),
            "max_depth": hp.choice(
                "gb_max_depth",
                [None, hp.randint("gb_max_depth_int", 1, 10)],
            ),
        },
    ],
)


def objective(params, X_train, y_train):
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        classifier_type = params["type"]
        del params["type"]
        if classifier_type == "dt":
            clf = DecisionTreeClassifier(**params)
        elif classifier_type == "rf":
            clf = RandomForestClassifier(**params)
        elif classifier_type == "gb":
            clf = GradientBoostingClassifier(**params)
        else:
            return 0
        acc = cross_val_score(clf, X_train, y_train).mean()
        clf.fit(X_train, y_train)

        mlflow.set_tag("Model", classifier_type)
        mlflow.set_tag("Data", "Training")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        mlflow.end_run()
        return {"loss": -acc, "status": STATUS_OK, "run_id": run_id}
