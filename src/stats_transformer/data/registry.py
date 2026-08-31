def _file(path, **kwargs):
    spec = {"path": path}
    if kwargs:
        spec["read_kwargs"] = kwargs
    return spec


def _single(path, description, time_period, source, **kwargs):
    return {
        "description": description,
        "time_period": time_period,
        "source": source,
        "files": {"data": _file(path, **kwargs)},
    }


EXAMPLE_DATASETS = {
    "macrodb_gdp_inflation": {
        "description": "Macroeconomic dataset with GDP and inflation data for multiple countries",
        "columns": {
            "country": "Country code (ISO 3-letter)",
            "date": "Year of observation",
            "inflation": "Inflation rate (annual %)",
            "gdp": "GDP per capita (current US$)",
        },
        "shape": (11490, 4),
        "time_period": "Annual",
        "source": "Derived from World Bank and IMF data",
        "notes": "Includes data for multiple countries across different time periods",
        "files": {
            "data": {
                "path": "macrodb_gdp_inflation.parquet",
                "package_path": "macrodb_gdp_inflation.parquet",
                "repository_path": "src/stats_transformer/data/macrodb_gdp_inflation.parquet",
            }
        },
    },
    "sofr_surprises": _single(
        "academic/acosta_brennan_jacobson_2024/sofr_surprises.parquet.gzip",
        "High-frequency SOFR futures monetary surprises",
        "Daily",
        "Acosta, Brennan, and Jacobson (2024)",
    ),
    "bauer_swanson_2023": {
        "description": "Orthogonalized monetary-surprise macroeconomic inputs",
        "time_period": "Monthly",
        "source": "Bauer and Swanson (2023)",
        "files": {
            "nonfarm_payrolls": _file("academic/bauer_swanson_2023/NonfarmPayrolls.txt"),
            "unemployment": _file("academic/bauer_swanson_2023/Unemployment.txt"),
            "cpi": _file("academic/bauer_swanson_2023/cpi.txt"),
        },
    },
    "bbm_2023": _single(
        "academic/bbm_2023/feds_subset.csv",
        "High-frequency yield-curve data",
        "Daily",
        "Bauer, Bernanke, and Milstein (2023)",
        skiprows=9,
    ),
    "pmu_data": _single(
        "academic/cieslak_hansen_mcmahon_xiao_2024/pmu_data.parquet.gzip",
        "Policymakers' Uncertainty text shocks",
        "Monthly",
        "Cieslak, Hansen, McMahon, and Xiao (2024)",
    ),
    "greenbook_forecast_errors": _single(
        "academic/coibion_gorodnichenko_2012/greenbook_forecast_errors.parquet.gzip",
        "FOMC Greenbook CPI forecast errors",
        "Quarterly",
        "Coibion and Gorodnichenko (2012)",
    ),
    "fomc_surprises": _single(
        "academic/jarocinski_karadi_2020/fomc_surprises.parquet.gzip",
        "FOMC event monetary and information surprises",
        "Daily / Event",
        "Jarocinski and Karadi (2020)",
    ),
    "news_sentiment": _single(
        "academic/shapiro_sudhof_wilson_2022/news_sentiment.parquet.gzip",
        "High-frequency news sentiment series",
        "Daily",
        "Shapiro, Sudhof, and Wilson (2022)",
    ),
    "policy_loans": _single(
        "academic/lane_2025/policy_loans.parquet.gzip",
        "Targeted industrial-policy lending panel",
        "Annual",
        "Lane (2025)",
    ),
    "tariffs": _single(
        "academic/lane_2025/tariffs.parquet.gzip",
        "Tariff liberalization panel",
        "Annual",
        "Lane (2025)",
    ),
    "global_factor": _single(
        "academic/miranda_agrippino_rey_2020/global_factor.parquet.gzip",
        "Published global financial factor series",
        "Monthly",
        "Miranda-Agrippino and Rey (2020)",
    ),
    "var_panel": _single(
        "academic/miranda_agrippino_rey_2020/var_panel.parquet.gzip",
        "Dynamic-factor model input panel",
        "Monthly",
        "Miranda-Agrippino and Rey (2020)",
    ),
    "nakamura_steinsson_2018": {
        "description": "High-frequency monetary shocks and macroeconomic controls",
        "time_period": "Event / Mixed",
        "source": "Nakamura and Steinsson (2018)",
        "files": {
            "nominal_yields": _file("academic/nakamura_steinsson_2018/NominalYields.csv"),
            "cpi": _file("academic/nakamura_steinsson_2018/cpi.csv"),
            "gdp": _file("academic/nakamura_steinsson_2018/gdp.csv"),
            "master": _file("academic/nakamura_steinsson_2018/master.dta"),
            "transformed_master": _file("academic/nakamura_steinsson_2018/transformed_master.csv"),
        },
    },
    "synthetic_nakamura": _single(
        "academic/synthetic_nakamura.csv",
        "Synthetic monetary-surprise PCA input",
        "Synthetic",
        "Nakamura and Steinsson-inspired synthetic data",
    ),
    "ghysels_ch1": {
        "description": "Regression and forecasting textbook exercises",
        "time_period": "Mixed / Synthetic",
        "source": "Ghysels and Marcellino (2018), Chapter 1",
        "files": {
            "gdp": _file("regression/ghysels_ch1/ex2_regress_gdp.csv"),
            "gdp_us": _file("regression/ghysels_ch1/ex2_regress_gdp_us.csv"),
            "oas": _file("regression/ghysels_ch1/ex3_regress_oas.csv"),
            "simulated": _file("regression/ghysels_ch1/simulated_data.csv"),
        },
    },
    "ghysels_ch2": {
        "description": "Regression misspecification and diagnostics textbook exercises",
        "time_period": "Mixed / Synthetic",
        "source": "Ghysels and Marcellino (2018), Chapter 2",
        "files": {
            "data": _file("regression/ghysels_ch2/Data.csv"),
            "default_risk": _file("regression/ghysels_ch2/default_risk.csv"),
            "gdp": _file("regression/ghysels_ch2/ex2_misspecification_gdp.csv"),
            "gdp_us": _file("regression/ghysels_ch2/ex2_misspecification_gdp_us.csv"),
            "simulated": _file("regression/ghysels_ch2/simulated_datac2.csv"),
        },
    },
    "longley": _single(
        "regression/longley.csv",
        "Highly collinear OLS benchmark",
        "Annual",
        "Longley (1967)",
    ),
    "grunfeld": _single(
        "regression/grunfeld.csv",
        "Corporate investment panel",
        "Annual",
        "Grunfeld (1958)",
    ),
    "mincer_wage": _single(
        "regression/mincer_wage.csv",
        "Cross-sectional Mincer wage equation data",
        "Cross-Section",
        "Mincer (1974) / Wooldridge wage data",
    ),
    "mroz": _single(
        "regression/mroz.csv",
        "Female labor-supply instrumental-variables data",
        "Cross-Section",
        "Mroz (1987)",
    ),
    "okuns_law": _single(
        "regression/okuns_law.csv",
        "GDP growth and unemployment changes",
        "Quarterly",
        "Okun's Law / FRED",
    ),
    "spector_logit": _single(
        "discrete/spector.csv",
        "Educational performance binary-choice data",
        "Cross-Section",
        "Spector and Mazzeo (1980)",
    ),
    "macrodata": _single(
        "timeseries/macrodata.csv",
        "US macroeconomic time-series benchmark",
        "Quarterly",
        "statsmodels macrodata",
    ),
    "ghysels_ch6": {
        "description": "VAR forecasting and impulse-response textbook exercises",
        "time_period": "Quarterly",
        "source": "Ghysels and Marcellino (2018), Chapter 6",
        "files": {
            "eu_growth": _file("timeseries/ghysels_ch6/eu_growth_ch6_sec10.csv"),
            "us_monetary": _file("timeseries/ghysels_ch6/usmonetary_ch6_sec11.csv"),
            "var_simulated": _file("timeseries/ghysels_ch6/var_simulated_ch6_sec9.csv"),
        },
    },
    "ghysels_ch7": {
        "description": "Cointegration and vector error-correction textbook exercises",
        "time_period": "Monthly",
        "source": "Ghysels and Marcellino (2018), Chapter 7",
        "files": {
            "uk_term_structure": _file("timeseries/ghysels_ch7/dataonly_uktermstructure.csv"),
            "us_lei": _file("timeseries/ghysels_ch7/dataonly_uslei.csv"),
            "simulated_cointegration": _file("timeseries/ghysels_ch7/simulated_cointegration.csv"),
        },
    },
    "adrr_2018": _single(
        "matlab_examples/ADRR2018_Data.xlsx",
        "Narrative sign-restriction benchmark workbook",
        "Workbook-defined",
        "Antolin-Diaz and Rubio-Ramirez (2018)",
    ),
    "bq_1989": _single(
        "matlab_examples/BQ1989_Data.xlsx",
        "Blanchard-Quah long-run SVAR benchmark workbook",
        "Quarterly",
        "Blanchard and Quah (1989)",
    ),
    "gk_2015": _single(
        "matlab_examples/GK2015_Data.xlsx",
        "Proxy SVAR monetary-policy benchmark workbook",
        "Monthly",
        "Gertler and Karadi (2015)",
    ),
    "jt_2025": _single(
        "matlab_examples/JT2025_Data.xlsx",
        "LP-IV local-projections benchmark workbook",
        "Workbook-defined",
        "Jorda and Taylor (2025)",
    ),
    "sw_2001": _single(
        "matlab_examples/SW2001_Data.xlsx",
        "Reduced-form VAR and SVEC benchmark workbook",
        "Quarterly",
        "Stock and Watson (2001)",
    ),
    "uhlig_2005": _single(
        "matlab_examples/Uhlig2005_Data.xlsx",
        "Sign-restricted monetary-policy SVAR benchmark workbook",
        "Monthly",
        "Uhlig (2005)",
    ),
}


__all__ = ["EXAMPLE_DATASETS"]
