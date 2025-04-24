from metaflow import FlowSpec, step, Parameter


class ClassifierScoringFlow(FlowSpec):

    model_name = Parameter(
        "model_name",
        help="Name of the registered model to use for scoring",
        default="best_iris_model",
    )

    model_version = Parameter(
        "model_version",
        help="Version of the registered model to use (latest if not specified)",
        default=None,
    )

    @step
    def start(self):
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        iris = fetch_ucirepo(id=53)
        X = iris.data.features
        y = iris.data.targets

        X_encoded = X.copy()

        self.column_mapping = {}
        self.label_encoders = {}

        string_columns = X.select_dtypes(include=["object", "category"]).columns

        for col in string_columns:
            # For columns with many unique values, use label encoding
            le = LabelEncoder()
            X_encoded[col + "_encoded"] = le.fit_transform(X[col])

            # Drop the original column
            X_encoded = X_encoded.drop(col, axis=1)

            # Store mapping information
            self.column_mapping[col] = [col + "_encoded"]
            self.label_encoders[col] = le

        X_encoded = X_encoded.astype(float)
        y_use = y.values.ravel()

        _, self.X_score, _, self.y_true = train_test_split(
            X_encoded, y_use, test_size=0.2, random_state=42
        )

        print(f"Loaded {len(self.X_score)} samples for scoring")
        self.next(self.load_model)  # Skip reduce_features

    @step
    def load_model(self):
        """
        Load the registered model from MLflow
        """
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Lab8.1_Iris")

        # Load model based on name and version
        if self.model_version:
            model_uri = f"models:/{self.model_name}/{self.model_version}"
        else:
            model_uri = f"models:/{self.model_name}/latest"

        self.model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model: {model_uri}")
        self.next(self.score)

    @step
    def score(self):
        """
        Make predictions using the loaded model
        """
        import pandas as pd
        import numpy as np
        from sklearn.metrics import accuracy_score, classification_report

        # Make predictions
        self.predictions = self.model.predict(self.X_score)  # Use original X_score

        # Create a dataframe with predictions
        self.results_df = pd.DataFrame({"prediction": self.predictions})

        # If we have ground truth (as in this example), calculate metrics
        if hasattr(self, "y_true"):
            accuracy = accuracy_score(self.y_true, self.predictions)
            print(f"Model accuracy on score set: {accuracy:.4f}")

            # Create classification report
            report = classification_report(self.y_true, self.predictions)
            print("Classification Report:")
            print(report)

        self.next(self.end)

    @step
    def end(self):
        """
        Save or display the results
        """
        import pandas as pd

        # Save predictions to CSV

        self.results_df["true"] = self.y_true
        self.results_df.to_csv("predictions.csv", index=False)

        # Display the first few predictions
        print("Sample predictions:")
        print(self.results_df.head())

        print("Flow completed successfully. Results saved to 'predictions.csv'")
        print(self.y_true)


if __name__ == "__main__":
    ClassifierScoringFlow()