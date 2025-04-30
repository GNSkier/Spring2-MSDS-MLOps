from metaflow import FlowSpec, step


class ClassifierTrainLab(FlowSpec):

    @step
    def start(self):
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Lab8.2_Testing")

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
        y_use = y.values.ravel()

        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X_encoded, y_use, test_size=0.2, shuffle=True
        )
        self.X_train_val, self.X_val, self.y_train_val, self.y_val = (
            train_test_split(self.X_train, self.y_train, test_size=0.2, shuffle=True)
        )

        self.next(self.train)

    @step
    def train(self):
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
        from utilities import objective, search_space
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Lab8.2_Testing")

        algo = tpe.suggest
        trials = Trials()

        def wrapped_objective(params):
            return objective(params, self.X_train_val, self.y_train_val)  # Use original data

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
        mlflow.set_experiment("Lab8.2_Testing")

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
