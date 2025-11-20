import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


def load_wine_data():
    """Load and combine red and white wine datasets."""
    white = pd.read_csv("winequality-white.csv", sep=";")
    red = pd.read_csv("winequality-red.csv", sep=";")

    white["type"] = 1  # white wine
    red["type"] = 0    # red wine

    df = pd.concat([white, red], ignore_index=True)
    return df


def train_test_split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["quality"])
    y = df["quality"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def evaluate_regression_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculamos primero el MSE y luego sacamos la raíz.
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    print(f"===== {name} =====")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")
    print()
    return rmse, r2



def main():
    # 1. Load data
    df = load_wine_data()
    print("Dataset shape:", df.shape)
    print(df.head(), "\n")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Baseline: Linear Regression (con escalado)
    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])

    evaluate_regression_model(
        "Baseline - Linear Regression",
        baseline_pipe,
        X_train, X_test, y_train, y_test
    )

    # 4. Random Forest Regressor con GridSearchCV
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("===== Random Forest - Best params (CV) =====")
    print(grid.best_params_)
    print(f"Best CV RMSE: {-grid.best_score_:.4f}\n")

    best_rf = grid.best_estimator_

    # 5. Evaluate best RF on test set
    rmse, r2 = evaluate_regression_model(
        "Best Random Forest Regressor",
        best_rf,
        X_train, X_test, y_train, y_test
    )

    # 6. Feature importances
    importances = best_rf.feature_importances_
    feature_names = X_train.columns

    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), feature_names[sorted_idx], rotation=45, ha="right")
    plt.tight_layout()
    plt.title("Feature importances (Random Forest)")
    plt.show()


if __name__ == "__main__":
    main()
