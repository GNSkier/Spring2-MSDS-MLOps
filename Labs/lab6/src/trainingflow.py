from metaflow import FlowSpec, step


class ClassifierTrainLab(FlowSpec):

    @step
    def start(self):
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Fetches data from UCIrepo
        iris = fetch_ucirepo(id=53)

        X = iris.data.features
        y = iris.data.targets

        X_encoded = X.copy()

        # Track column transformations
        column_mapping = {}
        label_encoders = {}

        # Find all object and category columns
        string_columns = X.select_dtypes(include=["object", "category"]).columns

        for col in string_columns:
            # For columns with many unique values, use label encoding
            le = LabelEncoder()
            X_encoded[col + "_encoded"] = le.fit_transform(X[col])

            # Drop the original column
            X_encoded = X_encoded.drop(col, axis=1)

            # Store mapping information
            column_mapping[col] = [col + "_encoded"]
            label_encoders[col] = le

        X_encoded = X_encoded.astype(float)
        # y_use = y["G1"]
        y_use = y.values.ravel()

        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X_encoded, y_use, test_size=0.2, shuffle=True
        )
        self.X_train_val, self.X_val, self.y_train_val, self.y_val = (
            train_test_split(self.X_train, self.y_train, test_size=0.2, shuffle=True)
        )
        self.next(self.reduce_xtrain)

    @step
    def reduce_xtrain(self):
        import pandas as pd
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor,
        )

        self.X_reduced_train = self.X_train.copy()
        VIF = [0]
        while len(VIF) > 0:
            X_numeric = pd.DataFrame()
            for col in self.X_reduced_train.columns:
                # Force everything through string conversion to be safe
                X_numeric[col] = pd.to_numeric(
                    self.X_train[col].astype(str), errors="coerce"
                )

            # Now calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_numeric.columns
            vif_data["VIF"] = [
                variance_inflation_factor(X_numeric.values, i)
                for i in range(X_numeric.shape[1])
            ]
            vif_test = vif_data.set_index("Feature").sort_values(
                by="VIF", ascending=False
            )
            if vif_test.max().iloc[0] > 5:
                print(VIF)
                VIF = vif_test.idxmax().iloc[0]
                self.X_reduced_train.drop(VIF, axis=1, inplace=True)
            else:
                print("Stopping")
                VIF = []
        self.next(self.train)

    @step
    def train(self):
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
        from utilities import objective, search_space
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Lab6_Iris")

        algo = tpe.suggest
        trials = Trials()

        def wrapped_objective(params):
            return objective(params, self.X_reduced_train, self.y_train)

        best_result = fmin(
            fn=wrapped_objective,
            space=search_space,
            algo=algo, 
            max_evals=32,
            trials=trials,
        )
        best_trial = min(trials.trials, key=lambda t: t["result"]["loss"])
        self.best_run_id = best_trial["result"]["run_id"]
        self.best_result = best_result
        self.next(self.report)

    @step
    def report(self):
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Lab6_Iris")

        model_uri = f"runs:/{self.best_run_id}/model"
        with mlflow.start_run():
            mlflow.register_model(model_uri=model_uri, name="best_iris_model")
        print("Model registered from run:", self.best_run_id)
        print("Best hyperparameters found")
        print(self.best_result)
        self.next(self.end)

    @step
    def end(self):
        print("Flow Completed Successfully")


if __name__ == "__main__":
    ClassifierTrainLab()
