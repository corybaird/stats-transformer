import os, sys, json, time
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

from src.stats_transformer.models.regression.robust_ols import RobustOLSModel
from src.stats_transformer.models.regression.iv import IV2SLSModel

def load_manual_env(path=".env"):
    env_vars = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    env_vars[key.strip()] = val.strip()
    return env_vars

ENV = load_manual_env()
FRED_KEY = ENV.get("API_FRED")

def fred_fetch(series_id, start="2000-01-01", end="2024-12-31"):
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={FRED_KEY}&file_type=json"
           f"&observation_start={start}&observation_end={end}")
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read())
        obs = [(o['date'], float(o['value'])) for o in data['observations'] if o['value'] != '.']
        dates, vals = zip(*obs)
        return pd.Series(vals, index=pd.to_datetime(dates), name=series_id)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return None

class Apep0238RealReplication:

    def __init__(self):
        # Using a representative subset of states
        self.states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        self.horizons = [0, 12, 24, 36, 48]
        self.gr_peak = "2007-12-01"
        self.saiz_data = {'CA': 1.15, 'TX': 2.50, 'FL': 1.60, 'NY': 0.90, 'IL': 1.70, 'PA': 1.90, 'OH': 2.70, 'GA': 2.40, 'NC': 2.35, 'MI': 2.60}

    def fetch_real_data(self):
        print(f"Fetching real FRED data for {len(self.states)} states...")
        all_data = []
        for st in self.states:
            emp = fred_fetch(f"{st}NA")
            hpi = fred_fetch(f"{st}STHPI")
            
            if emp is not None and hpi is not None:
                hpi_02 = hpi.get("2002-01-01", hpi.iloc[0])
                hpi_06 = hpi.get("2006-01-01", hpi.iloc[0])
                hpi_boom = np.log(hpi_06 / hpi_02)
                
                peak_val = emp.get(self.gr_peak, emp.iloc[0])
                
                for h in self.horizons:
                    target_date = pd.to_datetime(self.gr_peak) + pd.DateOffset(months=h)
                    if target_date in emp.index:
                        outcome = np.log(emp.loc[target_date] / peak_val)
                        all_data.append({
                            "state": st,
                            "horizon": h,
                            "hpi_boom": hpi_boom,
                            "gr_outcome": outcome,
                            "saiz": self.saiz_data.get(st)
                        })
        return pd.DataFrame(all_data)

    def run_comparison(self, df):
        print("\n" + "="*50)
        print("REAL DATA ESTIMATION: APEP_0238 (10 States)")
        print("="*50)
        
        for h in self.horizons:
            sub = df[df["horizon"] == h].copy()
            if len(sub) < 3: continue
            
            # Manual HC1
            y = sub["gr_outcome"].values
            X = np.column_stack([np.ones(len(sub)), sub["hpi_boom"].values])
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta_hat
            n, k = X.shape
            bread = np.linalg.inv(X.T @ X)
            meat = (resid**2).reshape(-1, 1, 1) * (X.reshape(n, k, 1) @ X.reshape(n, 1, k))
            V_hc1 = (n/(n-k)) * bread @ meat.sum(axis=0) @ bread
            se_hc1 = np.sqrt(np.diag(V_hc1))[1]
            
            # Stats-Transformer HC3
            model = RobustOLSModel(target="gr_outcome", independent_variables=["hpi_boom"], cov_type="HC3")
            model.fit(sub)
            se_hc3 = model.model.bse["hpi_boom"]
            
            diff = (se_hc3 - se_hc1) / se_hc1 * 100
            print(f"Horizon {h:2d} | Beta: {beta_hat[1]:.4f} | HC1 SE: {se_hc1:.4f} | HC3 SE: {se_hc3:.4f} | Diff: {diff:+.2f}%")

    def run(self):
        if not FRED_KEY:
            print("Error: API_FRED not found in .env")
            return
        
        df = self.fetch_real_data()
        if df.empty:
            print("Error: No data fetched.")
            return
            
        self.run_comparison(df)

if __name__ == "__main__":
    rep = Apep0238RealReplication()
    rep.run()
