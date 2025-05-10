import mlflow
import mlflow.sklearn
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import numpy as np
import dagshub
warnings.filterwarnings("ignore")

# Inisialisasi DagsHub untuk MLflow
dagshub.init(repo_owner='GinantiRiski1', repo_name='my-first-repo', mlflow=True)  # Ganti dengan username dan repo kamu

# Konfigurasi MLflow ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/GinantiRiski1/my-first-repo.mlflow")  # Ganti dengan URL repo kamu
mlflow.set_experiment("advanced-model_v2")

# Load data
X_train = pd.read_csv("car_preprocessing/X_train.csv")
X_test = pd.read_csv("car_preprocessing/X_test.csv")
y_train = pd.read_csv("car_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("car_preprocessing/y_test.csv").values.ravel()

# Grid search setup
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy')

start_time = time.time()

with mlflow.start_run(run_name="KNN_GridSearch_Advanced"):
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrik evaluasi utama
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Metrik tambahan untuk advance
    cm = confusion_matrix(y_test, y_pred)
    training_time = time.time() - start_time

    # Logging parameter dan metrik
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Logging metrik tambahan
    mlflow.log_metric("training_time_seconds", training_time)
    mlflow.log_metric("confusion_matrix_sum", np.sum(cm))

    # Logging model
    mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:5])  # Contoh 5 data pertama

    print(f"Model logged to DagsHub with accuracy: {acc:.4f}, training time: {training_time:.2f} sec")
