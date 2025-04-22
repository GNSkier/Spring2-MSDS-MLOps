from metaflow import FlowSpec, step, conda_base, schedule  # Removed kubernetes


@conda_base(
    libraries={"numpy": "1.23.5", "scikit-learn": "1.2.2"},
    python="3.9.16",
    packages={
        "google-cloud-storage": "2.5.0",
        "google-auth": "2.11.0",
        "google-cloud-secret-manager": "2.10.0",
    },
)
@schedule(hourly=True)  # Add the schedule decorator here
class ClassifierTrainFlow(FlowSpec):
    
    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = (
            train_test_split(X, y, test_size=0.2, random_state=0)
        )
        print("Data loaded successfully")
        self.lambdas = np.arange(0.001, 1, 0.2)
        self.next(self.train_lasso, foreach="lambdas")

    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print("Scores:")
        print("\n".join("%s %f" % res for res in self.results))
        print("Model:", self.model)


if __name__ == "__main__":
    ClassifierTrainFlow()
