import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from stats_transformer.reporting.defaults.labels import SIGNIFICANCE_THRESHOLDS
from stats_transformer.reporting.formatters.significance import format_number, star
from stats_transformer.reporting.tables.table_generator import TableGenerator

_CONSTANT_NAMES = ("const", "Intercept")
_CONSTANT_LABEL = "Constant"

class RegressionTable:
    """
    Booktabs LaTeX coefficient table: one column per model, coefficient with
    significance stars and (SE) below, N and R² at the bottom. Numbers come
    from the model metadata produced by ModelBase.get_model_metadata() (the
    same dict written to model_summary.json). The constant is listed last.
    """

    def __init__(self, table_generator: TableGenerator):
        self.table_gen = table_generator
        self.logger = logging.getLogger(self.__class__.__name__)

    def _column(self, model_results: Dict[str, Any], index: int) -> str:
        """Column header for one model: dependent variable, else "Model i"."""
        summary = model_results.get("summary", {})
        return summary.get("dependent_variable") or f"Model {index + 1}"

    def _flat_coefficients(self, model_results: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return {var: stats} for regression-family metadata, or None for
        system models (VAR/VECM/SVAR metadata nests {equation: {term: stats}}),
        for models without extractable coefficients, and for extraction errors."""
        coefficients = model_results.get("coefficients") or {}
        if not isinstance(coefficients, dict) or "error" in coefficients or not coefficients:
            return None
        if not all(isinstance(entry, dict) and "value" in entry for entry in coefficients.values()):
            return None
        return coefficients

    # (var, value, p, se) with the constant last; var is "Constant" when labeled
    @staticmethod
    def _ordered(coefficients: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        rest = [(var, entry) for var, entry in coefficients.items() if var not in _CONSTANT_NAMES]
        constants = [(f"{_CONSTANT_LABEL}", entry) for var, entry in coefficients.items() if var in _CONSTANT_NAMES]
        return rest + constants

    def build(self, model_results: Union[Dict[str, Any], List[Dict[str, Any]]], model_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Builds the table for one model dict or a list (one column per model).
        Returns None when no model in the list is regression-family (skipped via
        a log info, not an exception).
        """
        results_list = model_results if isinstance(model_results, list) else [model_results]
        columns = [model_names[i] if model_names and i < len(model_names) else self._column(results, i) for i, results in enumerate(results_list)]

        parsed: List[Optional[Dict[str, Tuple[str, str]]]] = []
        variable_order: List[str] = []
        for i, results in enumerate(results_list):
            coefficients = self._flat_coefficients(results)
            if coefficients is None:
                # Non-regression models (VAR, PCA, ...) get no column; logged, not raised.
                self.logger.info(f"Skipping regression table column '{columns[i]}': coefficients are not regression-family (no parsable coefficients).")
                parsed.append(None)
                continue
            cells: Dict[str, Tuple[str, str]] = {}
            for var, entry in self._ordered(coefficients):
                if var in cells:
                    continue  # duplicate var across models; first wins
                cells[var] = (
                    f"{format_number(entry['value'])}{star(entry.get('p_value'))}",
                    f"({format_number(entry['std_err'])})" if entry.get("std_err") is not None else "",
                )
                if var not in variable_order:
                    variable_order.append(var)
            parsed.append(cells)

        if all(p is None for p in parsed):
            return None

        rows: List[Dict[str, Any]] = []
        for var in variable_order:
            coef_row: Dict[str, Any] = {"Variable": var}
            se_row: Dict[str, Any] = {"Variable": ""}
            for i, cells in enumerate(parsed):
                coef_str, se_str = cells.get(var, ("", "")) if cells else ("", "")
                coef_row[columns[i]] = coef_str
                se_row[columns[i]] = se_str
            rows.append(coef_row)
            rows.append(se_row)

        # Bottom block (separated by a \midrule by export): N and R² per model
        for label, key in [("N", "nobs"), ("R$^2$", "rsquared"), ("Adj. R$^2$", "rsquared_adj")]:
            row: Dict[str, Any] = {"Variable": label}
            has_value = False
            for i, results in enumerate(results_list):
                value = results.get("summary", {}).get(key)
                if key == "nobs":
                    row[columns[i]] = f"{int(value):,}" if value is not None else ""
                else:
                    row[columns[i]] = format_number(value) if value is not None else ""
                has_value = has_value or value is not None
            if has_value:
                rows.append(row)

        return pd.DataFrame(rows, columns=["Variable"] + columns)

    def export(self, model_results: Union[Dict[str, Any], List[Dict[str, Any]]], filename: str, model_names: Optional[List[str]] = None) -> Optional[str]:
        """Builds the table and exports booktabs LaTeX with the star-threshold note; returns the path or None."""
        table = self.build(model_results, model_names)
        if table is None:
            return None
        thresholds = sorted(SIGNIFICANCE_THRESHOLDS, reverse=True)
        note = ", ".join(f"{SIGNIFICANCE_THRESHOLDS[t]} p$<${t:g}" for t in thresholds)
        # Bottom block starts at the N (or R^2) row; skip the break when that label
        # also appears as a coefficient variable (it would split the coefficient block).
        labels: List[str] = list(table["Variable"])
        break_label = next((label for label in ("N", "R$^2$") if labels.count(label) == 1 and label in labels), None)
        path = self.table_gen.export_latex(table, filename, midrule_before=[break_label] if break_label else None, note=note)
        self.logger.info(f"Regression table exported to {path}")
        return path