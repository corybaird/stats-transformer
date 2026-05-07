import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.iv import IV2SLSModel

class Apep0238Replication:

    def __init__(self):
        self.n_states = 50
        self.horizons = [0, 6, 12, 24, 36, 48]
        self.states = [f"ST_{i:02d}" for i in range(self.n_states)]

    def generate_data(self):
        data_rows = []
        np.random.seed(42)
        
        # Cross-sectional instruments (Z)
        hpi_boom = np.random.normal(0.2, 0.1, self.n_states)
        bartik_covid = np.random.normal(-0.05, 0.02, self.n_states)
        saiz = -hpi_boom + np.random.normal(0, 0.05, self.n_states)
        is_south = (np.random.rand(self.n_states) > 0.5).astype(int)

        for i, st in enumerate(self.states):
            for h in self.horizons:
                # GR path: persistent scarring from HPI boom
                gr_beta_h = -0.3 if h >= 24 else -0.15 * (h/12)
                gr_outcome = gr_beta_h * hpi_boom[i] + 0.01 * is_south[i] + np.random.normal(0, 0.01)
                
                # COVID path: quick bounce back
                covid_beta_h = -0.5 * np.exp(-h/12)
                covid_outcome = covid_beta_h * bartik_covid[i] + np.random.normal(0, 0.01)
                
                data_rows.append({
                    "state": st,
                    "horizon": h,
                    "hpi_boom": hpi_boom[i],
                    "bartik_covid": bartik_covid[i],
                    "saiz": saiz[i],
                    "is_south": is_south[i],
                    "gr_outcome": gr_outcome,
                    "covid_outcome": covid_outcome,
                    "const": 1.0
                })
        
        return pd.DataFrame(data_rows)

    def manual_estimation(self, df, target, instrument, controls):
        results = []
        for h in self.horizons:
            sub = df[df["horizon"] == h]
            y = sub[target].values
            x = sub[instrument].values
            c = sub[controls].values
            X = np.column_stack([np.ones(len(x)), x, c])
            
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta_hat
            n, k = X.shape
            
            # Manual HC1 SEs (Common in ad-hoc paper code)
            meat = np.zeros((k, k))
            for i in range(n):
                xi = X[i:i+1, :]
                meat += (resid[i]**2) * (xi.T @ xi)
            bread = np.linalg.inv(X.T @ X)
            V_hc1 = (n / (n - k)) * bread @ meat @ bread
            se = np.sqrt(np.diag(V_hc1))
            
            results.append({
                "h": h,
                "beta": float(beta_hat[1]),
                "se": float(se[1]),
            })
        return pd.DataFrame(results)

    def transformer_estimation(self, df, target, instrument, controls):
        results = []
        for h in self.horizons:
            sub = df[df["horizon"] == h].copy()
            # RobustOLSModel uses HC3 by default, which is superior for N=50
            model = RobustOLSModel(target=target, independent_variables=[instrument] + controls, cov_type="HC3")
            model.fit(sub)
            res = model.results
            
            results.append({
                "h": h,
                "beta": float(res.params[instrument]),
                "se": float(model.model.bse[instrument]),
            })
        return pd.DataFrame(results)

    def run(self):
        df = self.generate_data()
        controls = ["is_south"]
        
        print("\n" + "="*50)
        print("REPLICATING APEP_0238 ESTIMATION LOGIC")
        print("="*50)
        
        # 1. Great Recession Analysis
        print("\n[1] Great Recession: Recovery from Housing Boom")
        manual_gr = self.manual_estimation(df, "gr_outcome", "hpi_boom", controls)
        trans_gr = self.transformer_estimation(df, "gr_outcome", "hpi_boom", controls)
        
        print("\nComparison: Manual (HC1) vs Stats-Transformer (HC3)")
        print("Horizon | Manual Beta (SE) | Trans Beta (SE) | SE Difference %")
        print("-" * 65)
        for i in range(len(manual_gr)):
            h = manual_gr.loc[i, "h"]
            m_b, m_s = manual_gr.loc[i, "beta"], manual_gr.loc[i, "se"]
            t_b, t_s = trans_gr.loc[i, "beta"], trans_gr.loc[i, "se"]
            diff = (t_s - m_s) / m_s * 100
            print(f"{h:7d} | {m_b:7.3f} ({m_s:.4f}) | {t_b:7.3f} ({t_s:.4f}) | {diff:+.2f}%")

        # 2. 2SLS Demonstration
        print("\n[2] 2SLS: Using Saiz (2010) Geography for HPI Exposure")
        h = 48
        sub = df[df["horizon"] == h].copy()
        iv_model = IV2SLSModel(target="gr_outcome", independent_variables=["is_south"], endogenous=["hpi_boom"], instruments=["saiz"])
        iv_model.fit(sub)
        iv_res = iv_model.model
        
        print(f"\n2SLS Results at Horizon {h}:")
        print(f"  Instrumented Beta: {iv_res.params['hpi_boom']:.4f}")
        print(f"  Standard Error:    {iv_res.std_errors['hpi_boom']:.4f}")
        
        try:
            f_stat = iv_res.first_stage.diagnostics['f.stat']
            val = f_stat.iloc[0] if isinstance(f_stat, pd.Series) else f_stat
            print(f"  First-stage F:     {float(val):.2f}")
        except:
            pass

        print("\n" + "="*50)
        print("AGENT REVIEW COMPLETE")
        print("="*50)

if __name__ == "__main__":
    rep = Apep0238Replication()
    rep.run()
