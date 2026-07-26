import pandas as pd


class TimeSeriesReportData:

    def __init__(self, specification=None, coefficients=None, irfs=None, fevd=None, historical_decomposition=None, structural_shocks=None):
        self.specification = specification if specification is not None else pd.DataFrame()
        self.coefficients = coefficients if coefficients is not None else pd.DataFrame()
        self.irfs = irfs if irfs is not None else pd.DataFrame()
        self.fevd = fevd if fevd is not None else pd.DataFrame()
        self.historical_decomposition = historical_decomposition if historical_decomposition is not None else pd.DataFrame()
        self.structural_shocks = structural_shocks if structural_shocks is not None else pd.DataFrame()

    def tables(self):
        table_items = [
            ("specification", self.specification),
            ("coefficients", self.coefficients),
            ("irfs", self.irfs),
            ("fevd", self.fevd),
            ("historical_decomposition", self.historical_decomposition),
            ("structural_shocks", self.structural_shocks),
        ]
        return [(name, table) for name, table in table_items if not table.empty]


class TimeSeriesReportRun:

    def __init__(self, data, figures=None, tables=None):
        self.data = data
        self.figures = figures or {}
        self.tables = tables or {}
