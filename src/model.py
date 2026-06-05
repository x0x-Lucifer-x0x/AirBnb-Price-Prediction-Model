"""
Model
=====
Wraps XGBoost, LightGBM, and a stacked ensemble.
Supports optional Optuna hyperparameter tuning.
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
class PricePredictor:
    """
    Unified interface for XGBoost / LightGBM / Ensemble price predictor.
    """

    DEFAULT_XGB_PARAMS = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.05,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    DEFAULT_LGB_PARAMS = dict(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    def __init__(
        self,
        model_type: str = "ensemble",
        feature_names: Optional[List[str]] = None,
        tune_hyperparams: bool = False,
        n_trials: int = 50,
    ):
        assert model_type in ("xgboost", "lightgbm", "ensemble")
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.tune_hyperparams = tune_hyperparams
        self.n_trials = n_trials
        self.models_ = {}

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ):
        if self.model_type in ("xgboost", "ensemble"):
            self.models_["xgb"] = self._fit_xgb(X_train, y_train, X_val, y_val)
        if self.model_type in ("lightgbm", "ensemble"):
            self.models_["lgb"] = self._fit_lgb(X_train, y_train, X_val, y_val)

        # Meta-learner for ensemble (Ridge on OOF predictions)
        if self.model_type == "ensemble":
            self._fit_meta(X_train, y_train, X_val, y_val)

        logger.info(f"  Training complete. Models: {list(self.models_.keys())}")

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == "xgboost":
            return self.models_["xgb"].predict(X)
        if self.model_type == "lightgbm":
            return self.models_["lgb"].predict(X)
        # ensemble
        return self._predict_ensemble(X)

    # ------------------------------------------------------------------ #
    def feature_importances(self) -> dict:
        """Return averaged feature importances across models."""
        out = {}
        for name, mdl in self.models_.items():
            if name in ("xgb", "lgb") and hasattr(mdl, "feature_importances_"):
                imp = mdl.feature_importances_
                for i, fn in enumerate(self.feature_names):
                    out[fn] = out.get(fn, 0) + imp[i]
        n = max(1, sum(1 for k in self.models_ if k in ("xgb", "lgb")))
        return {k: v / n for k, v in out.items()}

    # ================================================================== #
    #  Private helpers
    # ================================================================== #
    def _fit_xgb(self, X_tr, y_tr, X_val, y_val):
        from xgboost import XGBRegressor
        params = self._tune_xgb(X_tr, y_tr, X_val, y_val) if self.tune_hyperparams \
                 else self.DEFAULT_XGB_PARAMS.copy()

        mdl = XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        mdl.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=200,
        )
        logger.info(f"  XGB best iteration: {mdl.best_iteration}")
        return mdl

    def _fit_lgb(self, X_tr, y_tr, X_val, y_val):
        from lightgbm import LGBMRegressor
        params = self._tune_lgb(X_tr, y_tr, X_val, y_val) if self.tune_hyperparams \
                 else self.DEFAULT_LGB_PARAMS.copy()

        mdl = LGBMRegressor(**params)
        mdl.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(50, verbose=False),
                __import__("lightgbm").log_evaluation(200),
            ],
        )
        logger.info(f"  LGB best iteration: {mdl.best_iteration_}")
        return mdl

    def _fit_meta(self, X_tr, y_tr, X_val, y_val):
        """Stack XGB + LGB with a Ridge meta-learner trained on val set."""
        from sklearn.linear_model import Ridge
        val_preds = np.column_stack([
            self.models_["xgb"].predict(X_val),
            self.models_["lgb"].predict(X_val),
        ])
        self.models_["meta"] = Ridge(alpha=1.0)
        self.models_["meta"].fit(val_preds, y_val)
        logger.info(f"  Meta weights: {self.models_['meta'].coef_}")

    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([
            self.models_["xgb"].predict(X),
            self.models_["lgb"].predict(X),
        ])
        return self.models_["meta"].predict(preds)

    # ── Optuna tuning ─────────────────────────────────────────────────
    def _tune_xgb(self, X_tr, y_tr, X_val, y_val) -> dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed – using default XGB params")
            return self.DEFAULT_XGB_PARAMS.copy()

        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 500, 2000),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=30,
            )
            m = XGBRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return mean_squared_error(y_val, m.predict(X_val), squared=False)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        best = study.best_params
        best.update({"tree_method": "hist", "random_state": 42, "n_jobs": -1})
        logger.info(f"  Best XGB params: {best}")
        return best

    def _tune_lgb(self, X_tr, y_tr, X_val, y_val) -> dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed – using default LGB params")
            return self.DEFAULT_LGB_PARAMS.copy()

        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        from sklearn.metrics import mean_squared_error

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 500, 2000),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 31, 255),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            m = LGBMRegressor(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)],
            )
            return mean_squared_error(y_val, m.predict(X_val), squared=False)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        best = study.best_params
        best.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
        logger.info(f"  Best LGB params: {best}")
        return best