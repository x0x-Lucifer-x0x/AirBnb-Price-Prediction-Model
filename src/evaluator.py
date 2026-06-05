"""
Evaluator
=========
Computes regression metrics, generates diagnostic plots,
and produces SHAP explainability outputs.
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, log_transform_target: bool = True):
        self.log_transform_target = log_transform_target

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        predictor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        output_dir: str,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        y_pred = predictor.predict(X_test)

        # ── metrics in log space ──────────────────────────────────────
        metrics_log = self._compute_metrics(y_test, y_pred, prefix="log_")

        # ── metrics in original price space ───────────────────────────
        if self.log_transform_target:
            y_test_raw = np.expm1(y_test)
            y_pred_raw = np.expm1(y_pred)
        else:
            y_test_raw, y_pred_raw = y_test, y_pred

        metrics_raw = self._compute_metrics(y_test_raw, y_pred_raw, prefix="")
        metrics = {**metrics_log, **metrics_raw}

        # save metrics
        with open(out / "metrics.json", "w") as f:
            json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)

        # ── plots ─────────────────────────────────────────────────────
        self._plot_diagnostics(y_test_raw, y_pred_raw, feature_names, predictor, out)

        # ── SHAP ──────────────────────────────────────────────────────
        self._shap_analysis(predictor, X_test, feature_names, out)

        # ── Error breakdown by price bucket ───────────────────────────
        self._error_by_bucket(y_test_raw, y_pred_raw, out)

        return metrics

    # ================================================================== #
    #  Metrics
    # ================================================================== #
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
        mdae = np.median(np.abs(y_true - y_pred))

        return {
            f"{prefix}rmse": rmse,
            f"{prefix}mae":  mae,
            f"{prefix}r2":   r2,
            f"{prefix}mape": mape,
            f"{prefix}mdae": mdae,
        }

    # ================================================================== #
    #  Diagnostic plots
    # ================================================================== #
    def _plot_diagnostics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        predictor,
        out: Path,
    ):
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

        # 1. Predicted vs Actual
        ax1 = fig.add_subplot(gs[0, 0])
        lim = (0, np.percentile(np.concatenate([y_true, y_pred]), 99))
        ax1.scatter(y_true, y_pred, alpha=0.25, s=8, color="#4C72B0")
        ax1.plot(lim, lim, "r--", lw=1.5, label="Perfect")
        ax1.set_xlim(lim); ax1.set_ylim(lim)
        ax1.set_xlabel("Actual Price ($)"); ax1.set_ylabel("Predicted Price ($)")
        ax1.set_title("Predicted vs Actual"); ax1.legend()

        # 2. Residuals vs Predicted
        ax2 = fig.add_subplot(gs[0, 1])
        resid = y_pred - y_true
        ax2.scatter(y_pred, resid, alpha=0.25, s=8, color="#DD8452")
        ax2.axhline(0, color="red", lw=1.5)
        ax2.set_xlabel("Predicted Price ($)"); ax2.set_ylabel("Residual ($)")
        ax2.set_title("Residuals vs Predicted")

        # 3. Residual distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(resid, bins=60, color="#55A868", edgecolor="white", linewidth=0.4)
        ax3.axvline(0, color="red", lw=1.5)
        ax3.set_xlabel("Residual ($)"); ax3.set_ylabel("Count")
        ax3.set_title("Residual Distribution")

        # 4. Feature importance (top 20)
        ax4 = fig.add_subplot(gs[1, :2])
        importances = predictor.feature_importances()
        if importances:
            imp_s = pd.Series(importances).sort_values(ascending=False).head(20)
            imp_s[::-1].plot.barh(ax=ax4, color="#4C72B0")
            ax4.set_title("Top-20 Feature Importances (averaged)")
            ax4.set_xlabel("Importance")

        # 5. Error by price decile
        ax5 = fig.add_subplot(gs[1, 2])
        df_err = pd.DataFrame({"actual": y_true, "pred": y_pred})
        df_err["decile"] = pd.qcut(df_err["actual"], 10, labels=False, duplicates="drop")
        mae_by_decile = df_err.groupby("decile").apply(
            lambda g: np.mean(np.abs(g["actual"] - g["pred"]))
        )
        mae_by_decile.plot.bar(ax=ax5, color="#C44E52")
        ax5.set_title("MAE by Price Decile")
        ax5.set_xlabel("Decile (0=cheapest)"); ax5.set_ylabel("MAE ($)")
        ax5.tick_params(axis="x", rotation=0)

        plt.suptitle("LA Airbnb Price Prediction – Diagnostic Report", fontsize=14, y=1.01)
        path = out / "diagnostics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Diagnostics plot saved → {path}")

    # ================================================================== #
    #  SHAP
    # ================================================================== #
    def _shap_analysis(self, predictor, X_test: np.ndarray, feature_names: List[str], out: Path):
        try:
            import shap
        except ImportError:
            logger.warning("  shap not installed – skipping SHAP analysis")
            return

        sample = min(500, len(X_test))
        idx = np.random.choice(len(X_test), sample, replace=False)
        X_sample = X_test[idx]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for ax, model_key in zip(axes, ["xgb", "lgb"]):
            if model_key not in predictor.models_:
                ax.axis("off")
                continue
            mdl = predictor.models_[model_key]
            explainer = shap.TreeExplainer(mdl)
            shap_vals = explainer.shap_values(X_sample)

            plt.sca(ax)
            shap.summary_plot(
                shap_vals, X_sample,
                feature_names=feature_names,
                max_display=20,
                show=False,
                plot_type="bar",
            )
            ax.set_title(f"SHAP – {model_key.upper()}", fontsize=12)

        path = out / "shap_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  SHAP plot saved → {path}")

    # ================================================================== #
    #  Error breakdown
    # ================================================================== #
    @staticmethod
    def _error_by_bucket(y_true: np.ndarray, y_pred: np.ndarray, out: Path):
        bins   = [0, 50, 100, 150, 200, 300, 500, 1000, np.inf]
        labels = ["<50","50-100","100-150","150-200","200-300","300-500","500-1k",">1k"]
        df = pd.DataFrame({"actual": y_true, "pred": y_pred})
        df["bucket"] = pd.cut(df["actual"], bins=bins, labels=labels)
        summary = df.groupby("bucket", observed=False).apply(
            lambda g: pd.Series({
                "n": len(g),
                "mae": np.mean(np.abs(g["actual"] - g["pred"])),
                "mape": np.mean(np.abs((g["actual"] - g["pred"]) / np.clip(g["actual"], 1, None))) * 100,
                "median_error": np.median(g["pred"] - g["actual"]),
            })
        ).reset_index()
        path = out / "error_by_price_bucket.csv"
        summary.to_csv(path, index=False)
        logger.info(f"  Error breakdown saved → {path}")