import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate


def main(X_train, Y_train, cv):
    model = Pipeline([
        ("model", LinearRegression())
    ])
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
    scores = cross_validate(
        model,
        X_train,
        Y_train,
        cv=cv,
        scoring=scoring
    )
    MAE = -np.mean(scores["test_MAE"])
    RMSE = -np.mean(scores["test_RMSE"])
    R2 = np.mean(scores["test_R2"])
    model.fit(X_train, Y_train)
    return model, MAE, RMSE, R2
