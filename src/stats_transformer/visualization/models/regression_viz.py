import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
from stats_transformer.visualization.models.model_viz import ModelVisualizer
from stats_transformer.visualization.formatters.style import apply_style
from stats_transformer.visualization.defaults.labels import get_readable_label

class RegressionVisualizer(ModelVisualizer):

    def __init__(self, params_path=None, output_dir="reports/visualizations", file_format="png", dpi=300, style="default"):
        if params_path is None:
            ModelVisualizer.__init__(self, params_path=None, output_dir=output_dir, file_format=file_format, dpi=dpi, style=style)
        else:
            super().__init__(params_path)
            if output_dir != "reports/visualizations" or file_format != "png" or dpi != 300 or style != "default":
                self.viz_params.update({"output_dir": output_dir, "file_format": file_format, "dpi": dpi, "style": style})
    
    def visualize_from_json(self, json_path=None, model_summary=None, subdir="regression", **kwargs):
        if json_path is None and model_summary is None:
            return {}
        if model_summary is None:
            model_summary = self.load_model_from_json(json_path)
        visualizations = {}
        try:
            visualizations["coefficient_plot"] = self.create_coefficient_plot_from_json(model_summary, subdir, display_only=kwargs.get('display_only', False))
        except Exception as e:
            self.logger.error(f"Error creating coefficient plot: {e}")
        try:
            visualizations["model_summary"] = self.create_model_summary_plot(model_summary, subdir, display_only=kwargs.get('display_only', False))
        except Exception as e:
            self.logger.error(f"Error creating model summary plot: {e}")
        return visualizations
    
    def create_model_summary_plot(self, model_summary, subdir="regression", display_only=False):
        self.logger.info("Creating model summary plot")
        summary = model_summary.get('summary', {})
        coefficients = model_summary.get('coefficients', {})

        def _fmt(v, decimals=4):
            if v is None or v == "N/A":
                return "N/A"
            try:
                fv = float(v)
                if abs(fv) < 0.001 and fv != 0:
                    return f"{fv:.3e}"
                return f"{fv:.{decimals}f}"
            except (TypeError, ValueError):
                return str(v)

        def _fmt_pval(v):
            if v is None or v == "N/A":
                return "N/A"
            try:
                fv = float(v)
                if fv < 0.001:
                    return f"{fv:.3e}"
                return f"{fv:.4f}"
            except (TypeError, ValueError):
                return str(v)

        r2 = summary.get("rsquared", summary.get("r_squared", "N/A"))
        adj_r2 = summary.get("rsquared_adj", summary.get("adj_r_squared", "N/A"))
        fstat = summary.get("fvalue", summary.get("f_statistic", "N/A"))
        fpval = summary.get("f_pvalue", "N/A")
        llf = summary.get("llf", summary.get("log_likelihood", "N/A"))
        aic = summary.get("aic", "N/A")
        bic = summary.get("bic", "N/A")
        nobs = summary.get("nobs", summary.get("num_observations", "N/A"))
        df_resid = summary.get("df_resid", summary.get("df_residuals", "N/A"))

        fit_rows = [
            ["R\u00b2", _fmt(r2)],
            ["Adj. R\u00b2", _fmt(adj_r2)],
            ["F-statistic", _fmt(fstat, 3)],
            ["Prob(F-statistic)", _fmt_pval(fpval)],
            ["Log-Likelihood", _fmt(llf, 3)],
            ["AIC", _fmt(aic, 3)],
            ["BIC", _fmt(bic, 3)],
            ["Observations", str(int(float(nobs))) if nobs != "N/A" else "N/A"],
            ["Df Residuals", str(int(float(df_resid))) if df_resid != "N/A" else "N/A"],
        ]

        coef_headers = ["Variable", "Coef.", "Std Err", "t", "P>|t|", "[0.025", "0.975]"]
        coef_data = []
        for var_name, stats in coefficients.items():
            coef_data.append([
                get_readable_label(var_name),
                _fmt(stats.get("value"), 6),
                _fmt(stats.get("std_err"), 6),
                _fmt(stats.get("t_value"), 3),
                _fmt_pval(stats.get("p_value")),
                _fmt(stats.get("ci_lower"), 6),
                _fmt(stats.get("ci_upper"), 6),
            ])

        n_fit = len(fit_rows)
        n_coef = len(coef_data) + 1
        fig_height = 1.4 + n_fit * 0.38 + 0.7 + n_coef * 0.40
        fig, axes = plt.subplots(2, 1, figsize=(13, fig_height),
                                 gridspec_kw={"height_ratios": [n_fit, n_coef]})
        for ax in axes:
            ax.axis("off")

        header_color = "#0a0a6e"
        alt_color = "#e8e8f0"

        fit_table = axes[0].table(
            cellText=fit_rows,
            colLabels=["Statistic", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.5, 0.35],
        )
        fit_table.auto_set_font_size(False)
        fit_table.set_fontsize(11)
        fit_table.scale(1, 1.65)
        for (i, j), cell in fit_table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if i == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor(header_color)
            elif i % 2 == 1:
                cell.set_facecolor(alt_color)
            else:
                cell.set_facecolor("white")
        axes[0].set_title("Model Fit Statistics", fontsize=11, fontweight="bold", pad=6, loc="left")

        coef_col_widths = [0.22, 0.12, 0.12, 0.10, 0.12, 0.12, 0.12]
        coef_table = axes[1].table(
            cellText=coef_data,
            colLabels=coef_headers,
            cellLoc="left",
            loc="center",
            colWidths=coef_col_widths,
        )
        coef_table.auto_set_font_size(False)
        coef_table.set_fontsize(11)
        coef_table.scale(1, 1.65)
        for (i, j), cell in coef_table.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if i == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor(header_color)
            elif i % 2 == 1:
                cell.set_facecolor(alt_color)
            else:
                cell.set_facecolor("white")
        axes[1].set_title("Coefficient Estimates (95% Confidence Interval)", fontsize=11, fontweight="bold", pad=6, loc="left")

        dep_var = summary.get("dependent_variable", "")
        if dep_var:
            dep_var = get_readable_label(dep_var)
        model_type = summary.get("model_type", "")
        subtitle_parts = []
        if dep_var:
            subtitle_parts.append(f"Dependent Variable: {dep_var}")
        if model_type:
            subtitle_parts.append(f"Model: {model_type}")

        top_margin = 0.88 if subtitle_parts else 0.92
        plt.subplots_adjust(top=top_margin, hspace=0.55)

        fig.suptitle("Regression Model Summary", fontsize=15, fontweight="bold")
        if subtitle_parts:
            fig.text(0.5, top_margin + 0.01, "   |   ".join(subtitle_parts), ha="center", fontsize=10, fontstyle="italic", color="#444444")


        return self.save_figure(fig, "model_summary_table", subdir, display_only=display_only)
    
    def create_variance_decomposition(self, model, X, subdir="regression"):
        self.logger.info("Creating variance decomposition plot")
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        if hasattr(model, 'params') and hasattr(model, 'model'):
            params = model.params
            y = model.model.endog
            tss = np.sum((y - np.mean(y)) ** 2)
            var_explained = {}
            for var in X.columns:
                if var in params.index:
                    var_explained[var] = (params[var] ** 2) * np.var(X[var]) * len(X[var]) / tss
            if hasattr(model, 'resid'):
                var_explained['Unexplained'] = np.sum(model.resid ** 2) / tss
            var_df = pd.DataFrame({'Variable': list(var_explained.keys()), 'Explained Variance': list(var_explained.values())})
            var_df = var_df.sort_values('Explained Variance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(var_df['Variable'], var_df['Explained Variance'])
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
            ax.set_title("Variance Decomposition", fontsize=12, fontweight='bold')
            ax.set_xlabel("Variable")
            ax.set_ylabel("Proportion of Variance Explained")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, min(1, max(var_df['Explained Variance']) * 1.2))
            plt.tight_layout()
            return self.save_figure(fig, "variance_decomposition", subdir)
        return None
    
    def create_partial_regression_plots(self, model, X, y, subdir="regression"):
        self.logger.info("Creating partial regression plots")
        saved_files = []
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        for i, col in enumerate(X.columns):
            fig, ax = plt.subplots(figsize=(8, 6))
            exog_others = X.drop(columns=[col])
            sm.graphics.plot_partregress(endog=y, exog_i=X[col], exog_others=exog_others, ax=ax)
            readable_col = get_readable_label(col)
            ax.set_xlabel(readable_col)
            ax.set_title(f"Partial Regression Plot: {readable_col}", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            saved_files.append(self.save_figure(fig, f"partial_regress_{col}", subdir))
        return saved_files
    
    def create_component_plus_residual_plots(self, model, X, subdir="regression"):
        self.logger.info("Creating component-plus-residual plots")
        saved_files = []
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        for i, col in enumerate(X.columns):
            fig, ax = plt.subplots(figsize=(8, 6))
            sm.graphics.plot_ccpr(model, i, ax=ax)
            ax.set_title(f"Component-Plus-Residual Plot: {get_readable_label(col)}", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            saved_files.append(self.save_figure(fig, f"ccpr_{col}", subdir))
        return saved_files
    
    def create_prediction_intervals(self, model, X, y, alpha=0.05, subdir="regression"):
        if not hasattr(model, 'get_prediction'):
            return None
        try:
            predictions = model.get_prediction(X)
            prediction_ci = predictions.conf_int(alpha=alpha)
            prediction_pi = predictions.conf_int(alpha=alpha, obs=True)
            mean_prediction = predictions.predicted_mean
            fig, ax = plt.subplots(figsize=(10, 6))
            n_obs = len(mean_prediction)
            x_values = np.arange(n_obs)
            y_values = y.values if hasattr(y, 'values') else y
            mean_pred_values = mean_prediction.values if hasattr(mean_prediction, 'values') else mean_prediction
            ci_lower = prediction_ci.iloc[:, 0].values if hasattr(prediction_ci, 'iloc') else prediction_ci[:, 0]
            ci_upper = prediction_ci.iloc[:, 1].values if hasattr(prediction_ci, 'iloc') else prediction_ci[:, 1]
            pi_lower = prediction_pi.iloc[:, 0].values if hasattr(prediction_pi, 'iloc') else prediction_pi[:, 0]
            pi_upper = prediction_pi.iloc[:, 1].values if hasattr(prediction_pi, 'iloc') else prediction_pi[:, 1]
            sort_idx = np.argsort(mean_pred_values)
            ax.scatter(x_values, y_values[sort_idx], alpha=0.6, color='blue', label='Actual')
            ax.plot(x_values, mean_pred_values[sort_idx], 'r-', label='Predicted')
            ax.fill_between(x_values, ci_lower[sort_idx], ci_upper[sort_idx], color='r', alpha=0.1, label=f'{int((1-alpha)*100)}% CI')
            ax.fill_between(x_values, pi_lower[sort_idx], pi_upper[sort_idx], color='g', alpha=0.1, label=f'{int((1-alpha)*100)}% PI')
            r2 = r2_score(y_values, mean_pred_values)
            ax.text(0.05, 0.95, f"R² = {r2:.3f}\nN = {n_obs}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.set_title("Predictions with Intervals", fontsize=12, fontweight='bold')
            ax.set_xlabel("Obs Index")
            ax.set_ylabel("Value")
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            return self.save_figure(fig, "prediction_intervals", subdir)
        except Exception as e:
            self.logger.error(f"Error creating prediction intervals plot: {e}")
            return None

    def create_residuals_vs_fitted(self, model, subdir="regression", display_only=False):
        self.logger.info("Creating residuals vs fitted scatter plot")
        if not hasattr(model, 'fittedvalues') or not hasattr(model, 'resid'):
            self.logger.warning("Model does not have fittedvalues/resid attributes")
            return None
        fitted = model.fittedvalues
        residuals = model.resid
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(fitted, residuals, alpha=0.35, s=18, color='#4C72B0', edgecolors='white', linewidth=0.3)
        ax.axhline(y=0, color='#c0392b', linestyle='--', linewidth=1.2, alpha=0.8)
        try:
            # Optimization: LOWESS is O(n^2). For large datasets, downsample for the trend line.
            max_points = 800
            if len(fitted) > max_points:
                self.logger.info(f"Downsampling for LOWESS calculation ({len(fitted)} -> {max_points})")
                indices = np.random.choice(len(fitted), max_points, replace=False)
                lowess_fitted = fitted.iloc[indices] if hasattr(fitted, 'iloc') else fitted[indices]
                lowess_resid = residuals.iloc[indices] if hasattr(residuals, 'iloc') else residuals[indices]
                lowess = sm.nonparametric.lowess(lowess_resid, lowess_fitted, frac=0.3)
            else:
                lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.3)
            
            # Sort for plotting
            lowess_sorted = lowess[lowess[:, 0].argsort()]
            ax.plot(lowess_sorted[:, 0], lowess_sorted[:, 1], color='#e67e22', linewidth=2.5, label='LOWESS trend')
            ax.legend(loc='best', fontsize=9)
        except Exception as e:
            self.logger.warning(f"Could not calculate LOWESS: {e}")
        ax.set_title("Residuals vs Fitted Values", fontsize=12, fontweight='bold')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.grid(True, linestyle='--', alpha=0.7)
        stats_text = f"Mean resid: {residuals.mean():.4f}\nStd resid: {residuals.std():.4f}"
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        plt.tight_layout()
        return self.save_figure(fig, "residuals_vs_fitted", subdir, display_only=display_only)

    def create_actual_vs_predicted(self, model, entity_labels=None, subdir="regression", display_only=False):
        self.logger.info("Creating actual vs predicted scatter plot")
        if not hasattr(model, 'fittedvalues') or not hasattr(model, 'model'):
            self.logger.warning("Model does not have fittedvalues/model attributes")
            return None
        actual = model.model.endog
        predicted = model.fittedvalues.values if hasattr(model.fittedvalues, 'values') else model.fittedvalues
        fig, ax = plt.subplots(figsize=(10, 7))
        if entity_labels is not None and len(entity_labels) == len(actual):
            unique_entities = sorted(set(entity_labels))
            from stats_transformer.visualization.defaults.colors import get_color_palette
            colors = get_color_palette("colorblind", len(unique_entities))
            color_map = {e: colors[i] for i, e in enumerate(unique_entities)}
            for entity in unique_entities:
                mask = [e == entity for e in entity_labels]
                ax.scatter(np.array(actual)[mask], np.array(predicted)[mask], alpha=0.4, s=20, color=color_map[entity], label=entity, edgecolors='white', linewidth=0.3)
            ax.legend(loc='best', fontsize=8, ncol=2)
        else:
            ax.scatter(actual, predicted, alpha=0.35, s=18, color='#4C72B0', edgecolors='white', linewidth=0.3)
        all_vals = np.concatenate([actual, predicted])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'r--', linewidth=1.5, alpha=0.7, label='45° line')
        r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        n = len(actual)
        ax.text(0.02, 0.97, f"R² = {r2:.4f}\nN = {n}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_title("Actual vs Predicted", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return self.save_figure(fig, "actual_vs_predicted", subdir, display_only=display_only)

    def run(self):
        pass

