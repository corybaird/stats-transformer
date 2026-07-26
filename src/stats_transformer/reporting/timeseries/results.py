import pandas as pd


import xarray as xr

class TimeSeriesReportData:

    def __init__(self, specification=None, coefficients=None, irfs=None, fevd=None, historical_decomposition=None, structural_shocks=None):
        self.specification = specification if specification is not None else pd.DataFrame()
        self.coefficients = coefficients if coefficients is not None else pd.DataFrame()
        self.irfs = irfs if irfs is not None else xr.Dataset()
        self.fevd = fevd if fevd is not None else xr.Dataset()
        self.historical_decomposition = historical_decomposition if historical_decomposition is not None else xr.Dataset()
        self.structural_shocks = structural_shocks if structural_shocks is not None else xr.Dataset()

    def tables(self):
        table_items = [
            ("specification", self.specification),
            ("coefficients", self.coefficients),
        ]
        
        for name, ds in [
            ("irfs", self.irfs),
            ("fevd", self.fevd),
            ("historical_decomposition", self.historical_decomposition),
            ("structural_shocks", self.structural_shocks),
        ]:
            if len(ds.data_vars) > 0:
                table_items.append((name, ds.to_dataframe().reset_index()))

        return [(name, table) for name, table in table_items if not table.empty]


class TimeSeriesReportRun:

    def __init__(self, data, figures=None, tables=None):
        self.data = data
        self.figures = figures or {}
        self.tables = tables or {}
