"""
NAVRIS Phase 3.0: XGBoost Regression for Local Navigation Velocity Residuals.

Implements three independent XGBoost regressors for East, North, and Up velocity residuals:
[delta_v_E, delta_v_N, delta_v_U]

Key architectural choices:
- Independent regressors: Simple, auditable, distinct dynamics per axis.
- Standard scaler: Fitted strictly on training partition features only.
- Modest hyperparameters: Avoids overfitting to training trajectory artifacts.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from navris.ml.features import FEATURE_NAMES, TARGET_NAMES


class VelocityResidualXGBoost:
    """
    3-Axis XGBoost Regressor for Navigation Velocity Residual Prediction:
    delta_v^n = [delta_v_east, delta_v_north, delta_v_up]^T.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.target_names: List[str] = list(TARGET_NAMES)

        self.model_east: Optional[xgb.XGBRegressor] = None
        self.model_north: Optional[xgb.XGBRegressor] = None
        self.model_up: Optional[xgb.XGBRegressor] = None

        self.is_fitted: bool = False

    def _create_regressor(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="reg:squarederror",
            tree_method="hist",
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
    ) -> "VelocityResidualXGBoost":
        """
        Fits the scaler strictly on X_train, then trains independent regressors for E, N, and U.
        """
        assert list(X_train.columns) == self.feature_names, "Feature names mismatch in X_train"
        assert list(y_train.columns) == self.target_names, "Target names mismatch in y_train"

        # Fit scaler ONLY on training features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        X_val_scaled = None
        if X_val is not None:
            assert list(X_val.columns) == self.feature_names, "Feature names mismatch in X_val"
            X_val_scaled = self.scaler.transform(X_val)

        # 1. East residual regressor
        self.model_east = self._create_regressor()
        eval_set_e = [(X_val_scaled, y_val.iloc[:, 0].values)] if (X_val_scaled is not None and y_val is not None) else None
        self.model_east.fit(
            X_train_scaled,
            y_train.iloc[:, 0].values,
            eval_set=eval_set_e,
            verbose=False,
        )

        # 2. North residual regressor
        self.model_north = self._create_regressor()
        eval_set_n = [(X_val_scaled, y_val.iloc[:, 1].values)] if (X_val_scaled is not None and y_val is not None) else None
        self.model_north.fit(
            X_train_scaled,
            y_train.iloc[:, 1].values,
            eval_set=eval_set_n,
            verbose=False,
        )

        # 3. Up residual regressor
        self.model_up = self._create_regressor()
        eval_set_u = [(X_val_scaled, y_val.iloc[:, 2].values)] if (X_val_scaled is not None and y_val is not None) else None
        self.model_up.fit(
            X_train_scaled,
            y_train.iloc[:, 2].values,
            eval_set=eval_set_u,
            verbose=False,
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts 3-axis velocity residuals [pred_delta_v_E, pred_delta_v_N, pred_delta_v_U].
        Input features are transformed using the scaler fitted on training data.

        Returns:
            np.ndarray of shape (N, 3).
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")

        assert list(X.columns) == self.feature_names, "Feature names mismatch during predict"
        X_scaled = self.scaler.transform(X)

        pred_e = self.model_east.predict(X_scaled)
        pred_n = self.model_north.predict(X_scaled)
        pred_u = self.model_up.predict(X_scaled)

        return np.column_stack([pred_e, pred_n, pred_u])

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns feature importances for all three axes.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance_east": self.model_east.feature_importances_,
            "importance_north": self.model_north.feature_importances_,
            "importance_up": self.model_up.feature_importances_,
        })
        df_imp["importance_mean"] = df_imp[
            ["importance_east", "importance_north", "importance_up"]
        ].mean(axis=1)
        return df_imp.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
