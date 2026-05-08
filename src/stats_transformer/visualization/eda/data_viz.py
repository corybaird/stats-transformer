import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from stats_transformer.visualization.base import BaseVisualizer
from stats_transformer.visualization.utils.viz_utils import configure_plot_aesthetics, get_color_palette
def get_readable_label(x): return str(x)

class DataVisualizer(BaseVisualizer):
    
    def __init__(self, params_path=None, output_dir="reports/visualizations", file_format="png", dpi=300, style="default"):
        super().__init__(params_path, output_dir, file_format, dpi, style)
        
    def create_visualization(self, data, feature_list=None, viz_type="histogram", display_only=False, **kwargs):
        kwargs['display_only'] = display_only
        if feature_list is None:
            feature_list = data.select_dtypes(include=[np.number]).columns.tolist()
        
        missing_features = [f for f in feature_list if f not in data.columns]
        if missing_features:
            raise ValueError(f"Features not in dataframe: {missing_features}")
        
        if viz_type == "histogram":
            return self.create_histograms(data, feature_list, **kwargs)
        elif viz_type == "scatter":
            return self.create_scatter_plots(data, feature_list, **kwargs)
        elif viz_type == "distribution":
            return self.create_distribution_plots(data, feature_list, **kwargs)
        elif viz_type == "correlation":
            return self.create_correlation_matrix(data, feature_list, **kwargs)
        elif viz_type == "boxplot":
            return self.create_box_plots(data, feature_list, **kwargs)
        elif viz_type == "normality":
            return self.test_normality(data, feature_list, **kwargs)
        else:
            raise ValueError(f"Visualization type '{viz_type}' not supported")
    
    def create_time_series_plot(self, data, entity_column=None, date_column="date", display_only=False, subdir="time_series"):
        if date_column not in data.columns:
            return None
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if entity_column and entity_column in df.columns:
            entities = df[entity_column].unique()
            colors = get_color_palette("default", len(entities))
            for i, entity in enumerate(entities):
                entity_data = df[df[entity_column] == entity]
                numeric_cols = entity_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    ax.plot(entity_data[date_column], entity_data[col], label=f"{entity}-{col}", color=colors[i], alpha=0.7)
            ax.legend(loc='best', fontsize=8)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                ax.plot(df[date_column], df[col], label=col, alpha=0.7)
            ax.legend(loc='best')
        
        configure_plot_aesthetics(ax, title="Time Series Plot", xlabel=date_column, ylabel="Value", legend=True, grid=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self.save_figure(fig, "time_series", subdir, display_only=display_only)
    
    def create_histograms(self, data, feature_list, bins=30, kde=True, subdir="histograms", display_only=False):
        saved_files = []
        for feature in feature_list:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data[feature].dropna(), bins=bins, kde=kde, ax=ax)
            configure_plot_aesthetics(ax, title=f"Dist of {get_readable_label(feature)}", xlabel=get_readable_label(feature), ylabel="Freq", grid=True)
            stats_text = f"Mean: {data[feature].mean():.2f}\nMedian: {data[feature].median():.2f}\nSD: {data[feature].std():.2f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            filepath = self.save_figure(fig, f"histogram_{feature}", subdir, display_only=display_only)
            if not display_only:
                saved_files.append(filepath)
        return None if display_only else saved_files
    
    def create_scatter_plots(self, data, feature_list, target=None, hue=None, subdir="scatter_plots", display_only=False):
        saved_files = []
        if target is not None:
            for feature in feature_list:
                if feature == target: continue
                fig, ax = plt.subplots(figsize=(8, 6))
                if hue and hue in data.columns:
                    sns.scatterplot(x=feature, y=target, data=data, hue=hue, alpha=0.7, ax=ax)
                else:
                    sns.scatterplot(x=feature, y=target, data=data, alpha=0.7, ax=ax)
                corr = data[[feature, target]].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                sns.regplot(x=feature, y=target, data=data, scatter=False, ax=ax, line_kws={"color":"red", "alpha":0.7, "lw":2})
                feature_label = get_readable_label(feature)
                target_label = get_readable_label(target)
                configure_plot_aesthetics(ax, title=f"{feature_label} vs {target_label}", xlabel=feature_label, ylabel=target_label, grid=True)
                filepath = self.save_figure(fig, f"scatter_{feature}_vs_{target}", subdir, display_only=display_only)
                if not display_only:
                    saved_files.append(filepath)
        return None if display_only else saved_files
    
    def create_distribution_plots(self, data, feature_list, by_group=None, subdir="distributions", display_only=False):
        saved_files = []
        for feature in feature_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            if by_group and by_group in data.columns:
                groups = data[by_group].unique()[:10]
                colors = get_color_palette("default", len(groups))
                for i, group in enumerate(groups):
                    group_data = data[data[by_group] == group][feature].dropna()
                    if len(group_data) > 1:
                        sns.kdeplot(group_data, ax=ax, label=str(group), color=colors[i])
                ax.legend(title=by_group)
            else:
                sns.histplot(data[feature].dropna(), kde=True, ax=ax)
            configure_plot_aesthetics(ax, title=f"Dist of {get_readable_label(feature)}", xlabel=get_readable_label(feature), ylabel="Density", legend=by_group is not None, grid=True)
            filepath = self.save_figure(fig, f"distribution_{feature}", subdir, display_only=display_only)
            if not display_only:
                saved_files.append(filepath)
        return None if display_only else saved_files
    
    def create_correlation_matrix(self, data, feature_list, method="pearson", subdir="correlations", display_only=False):
        corr_matrix = data[feature_list].corr(method=method)
        fig, ax = plt.subplots(figsize=(max(8, len(feature_list)), max(6, len(feature_list))))
        readable_features = [get_readable_label(f) for f in feature_list]
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, linewidths=0.5, ax=ax, fmt=".2f", xticklabels=readable_features, yticklabels=readable_features)
        plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_figure(fig, f"correlation_matrix_{method}", subdir, display_only=display_only)
    
    def create_box_plots(self, data, feature_list, by_group=None, subdir="boxplots", display_only=False):
        saved_files = []
        for feature in feature_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            if by_group and by_group in data.columns and data[by_group].nunique() <= 15:
                sns.boxplot(x=by_group, y=feature, data=data, ax=ax)
                plt.xticks(rotation=45)
            else:
                sns.boxplot(x=data[feature].dropna(), ax=ax)
            plt.title(f"Box Plot of {get_readable_label(feature)}", fontsize=12, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            filepath = self.save_figure(fig, f"boxplot_{feature}", subdir, display_only=display_only)
            if not display_only:
                saved_files.append(filepath)
        return None if display_only else saved_files
    
    def test_normality(self, data, feature_list, test_type="shapiro", subdir="normality_tests", display_only=False):
        results = {}
        for feature in feature_list:
            feature_data = data[feature].dropna()
            if len(feature_data) < 3: continue
            if test_type == "shapiro":
                stat, p_value = stats.shapiro(feature_data)
                test_name = "Shapiro-Wilk"
            elif test_type == "ks":
                stat, p_value = stats.kstest(feature_data, 'norm', args=(feature_data.mean(), feature_data.std()))
                test_name = "Kolmogorov-Smirnov"
            else:
                raise ValueError(f"Unknown test: {test_type}")
            is_normal = p_value > 0.05
            results[feature] = {"test": test_name, "statistic": stat, "p_value": p_value, "is_normal": is_normal}
            fig, ax = plt.subplots(figsize=(8, 6))
            sm.qqplot(feature_data, line='45', ax=ax)
            ax.set_title(f"Q-Q Plot for {get_readable_label(feature)}\np={p_value:.4f}", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            results[feature]["qq_plot"] = self.save_figure(fig, f"qqplot_{feature}_{test_type}", subdir, display_only=display_only)
        return results

    def run(self):
        pass

