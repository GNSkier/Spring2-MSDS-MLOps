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
        self.next(self.reduce_features)

    @step
    def reduce_features(self):
        """
        Apply the same feature reduction logic to ensure compatibility with the model
        """
        import pandas as pd
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor,
        )

        self.X_reduced = self.X_score.copy()
        VIF = [0]
        while len(VIF) > 0:
            X_numeric = pd.DataFrame()
            for col in self.X_reduced.columns:
                X_numeric[col] = pd.to_numeric(
                    self.X_reduced[col].astype(str), errors="coerce"
                )

            # Calculate VIF
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
                VIF = vif_test.idxmax().iloc[0]
                self.X_reduced.drop(VIF, axis=1, inplace=True)
            else:
                VIF = []

        print(
            f"Data ready for scoring with {len(self.X_reduced.columns)} features"
        )
        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the registered model from MLflow
        """
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Lab6_Iris")

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
        self.predictions = self.model.predict(self.X_reduced)

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
