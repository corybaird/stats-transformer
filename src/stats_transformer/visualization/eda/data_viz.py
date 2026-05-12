import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from stats_transformer.visualization.base import BaseVisualizer
from stats_transformer.visualization.defaults.labels import get_readable_label
from stats_transformer.visualization.formatters.style import apply_style

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
        
        from stats_transformer.visualization.charts.time_series import TimeSeriesPlot
        
        if entity_column and entity_column in df.columns:
            # We want one line per entity, so we pivot the data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c != entity_column]
            
            # Use the first numeric col if there are multiple, to keep it simple, or pivot
            if not numeric_cols:
                return None
                
            y_col = numeric_cols[0]
            df_pivot = df.pivot(index=date_column, columns=entity_column, values=y_col).reset_index()
            y_cols = [c for c in df_pivot.columns if c != date_column]
            
            chart = TimeSeriesPlot(style_path=self.viz_params.get("style", "timeseries"))
            fig, ax = chart.plot(
                df=df_pivot, 
                x_col=date_column, 
                y_cols=y_cols, 
                title=f"Time Series Plot ({y_col})", 
                ylabel="Value"
            )
            return self.save_figure(fig, "time_series", subdir, display_only=display_only)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c != date_column]
            
            chart = TimeSeriesPlot(style_path=self.viz_params.get("style", "timeseries"))
            fig, ax = chart.plot(
                df=df, 
                x_col=date_column, 
                y_cols=numeric_cols, 
                title="Time Series Plot", 
                ylabel="Value"
            )
            return self.save_figure(fig, "time_series", subdir, display_only=display_only)
    
    def create_histograms(self, data, feature_list, bins=30, kde=True, subdir="histograms", display_only=False):
        saved_files = []
        apply_style(self.viz_params.get("style", "default"))
        for feature in feature_list:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data[feature].dropna(), bins=bins, kde=kde, ax=ax)
            ax.set_title(f"Dist of {get_readable_label(feature)}", fontsize=12, fontweight='bold')
            ax.set_xlabel(get_readable_label(feature), fontsize=10)
            ax.set_ylabel("Freq", fontsize=10)
            stats_text = f"Mean: {data[feature].mean():.2f}\nMedian: {data[feature].median():.2f}\nSD: {data[feature].std():.2f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            filepath = self.save_figure(fig, f"histogram_{feature}", subdir, display_only=display_only)
            if not display_only:
                saved_files.append(filepath)
        return None if display_only else saved_files
    
    def create_scatter_plots(self, data, feature_list, target=None, hue=None, subdir="scatter_plots", display_only=False):
        saved_files = []
        if target is not None:
            from stats_transformer.visualization.charts.scatter import ScatterWithRegression
            chart = ScatterWithRegression(style_path=self.viz_params.get("style", "scatter"))
            for feature in feature_list:
                if feature == target: continue
                
                fig, ax = chart.plot(
                    df=data,
                    x_col=feature,
                    y_col=target,
                    hue_col=hue,
                    title=f"{get_readable_label(feature)} vs {get_readable_label(target)}"
                )
                
                filepath = self.save_figure(fig, f"scatter_{feature}_vs_{target}", subdir, display_only=display_only)
                if not display_only:
                    saved_files.append(filepath)
        return None if display_only else saved_files
    
    def create_distribution_plots(self, data, feature_list, by_group=None, subdir="distributions", display_only=False):
        saved_files = []
        apply_style(self.viz_params.get("style", "default"))
        for feature in feature_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            if by_group and by_group in data.columns:
                groups = data[by_group].unique()[:10]
                from stats_transformer.visualization.defaults.colors import get_color_palette
                colors = get_color_palette("default", len(groups))
                for i, group in enumerate(groups):
                    group_data = data[data[by_group] == group][feature].dropna()
                    if len(group_data) > 1:
                        sns.kdeplot(group_data, ax=ax, label=str(group), color=colors[i])
                ax.legend(title=by_group)
            else:
                sns.histplot(data[feature].dropna(), kde=True, ax=ax)
                
            ax.set_title(f"Dist of {get_readable_label(feature)}", fontsize=12, fontweight='bold')
            ax.set_xlabel(get_readable_label(feature), fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            
            filepath = self.save_figure(fig, f"distribution_{feature}", subdir, display_only=display_only)
            if not display_only:
                saved_files.append(filepath)
        return None if display_only else saved_files
    
    def create_correlation_matrix(self, data, feature_list, method="pearson", subdir="correlations", display_only=False):
        from stats_transformer.visualization.charts.heatmap import CorrelationHeatmap
        chart = CorrelationHeatmap(style_path=self.viz_params.get("style", "default"))
        fig, ax = chart.plot(df=data, columns=feature_list, method=method)
        return self.save_figure(fig, f"correlation_matrix_{method}", subdir, display_only=display_only)
    
    def create_box_plots(self, data, feature_list, by_group=None, subdir="boxplots", display_only=False):
        saved_files = []
        apply_style(self.viz_params.get("style", "default"))
        for feature in feature_list:
            fig, ax = plt.subplots(figsize=(10, 6))
            if by_group and by_group in data.columns and data[by_group].nunique() <= 15:
                sns.boxplot(x=by_group, y=feature, data=data, ax=ax)
                plt.xticks(rotation=45)
            else:
                sns.boxplot(x=data[feature].dropna(), ax=ax)
            plt.title(f"Box Plot of {get_readable_label(feature)}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            filepath = self.save_figure(fig, f"boxplot_{feature}", subdir, display_only=display_only)
            if not display_only:
                saved_files.append(filepath)
        return None if display_only else saved_files
    
    def test_normality(self, data, feature_list, test_type="shapiro", subdir="normality_tests", display_only=False):
        apply_style(self.viz_params.get("style", "default"))
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
            results[feature]["qq_plot"] = self.save_figure(fig, f"qqplot_{feature}_{test_type}", subdir, display_only=display_only)
        return results

    def run(self):
        pass
