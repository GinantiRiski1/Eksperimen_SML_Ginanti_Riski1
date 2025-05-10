import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data():
    X_train = pd.read_csv("car_preprocessing/X_train.csv")
    X_test = pd.read_csv("car_preprocessing/X_test.csv")
    y_train = pd.read_csv("car_preprocessing/y_train.csv").values.ravel()
    y_test = pd.read_csv("car_preprocessing/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def train_model(n_neighbors):
    # Aktifkan autolog sebelum training
    mlflow.sklearn.autolog()

    # Set eksperimen
    mlflow.set_experiment("basic-model_v2")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=3, help="Jumlah tetangga untuk KNN")
    args = parser.parse_args()

    train_model(args.n_neighbors)
