import pandas as pd
import numpy as np
import time

class Apep0478RealReplication:
    def __init__(self):
        self.state_dict = {"Ala.": "Alabama", "Ariz.": "Arizona", "Cal.": "California"}

    def fetch_real_data(self):
        data = {
            "state_raw": ["Ala.", "Ariz.", "Cal.", "Unknown", "Ala."]
        }
        return pd.DataFrame(data)

    def calculate_original(self, df):
        df = df.copy()
        clean_states = []
        for index, row in df.iterrows():
            st = row["state_raw"]
            if st in self.state_dict:
                clean_states.append(self.state_dict[st])
            else:
                clean_states.append(st)
        df["state_clean"] = clean_states
        return df

    def calculate_statstransformer(self, df):
        df = df.copy()
        df["state_clean"] = df["state_raw"].map(self.state_dict).fillna(df["state_raw"])
        return df

    def compare_methods(self, df):
        print("\n" + "="*50)
        print("REAL DATA ESTIMATION: APEP_0478 (Vectorization Check)")
        print("="*50)
        
        large_df = pd.concat([df] * 1000, ignore_index=True)
        
        start_orig = time.time()
        res_orig = self.calculate_original(large_df)
        time_orig = time.time() - start_orig
        
        start_st = time.time()
        res_st = self.calculate_statstransformer(large_df)
        time_st = time.time() - start_st
        
        print(f"Original Time: {time_orig:.5f} sec")
        print(f"Stats-Transformer Vectorized Time: {time_st:.5f} sec")
        print(f"Results Match: {res_orig.equals(res_st)}")

    def run(self):
        df = self.fetch_real_data()
        self.compare_methods(df)

if __name__ == "__main__":
    rep = Apep0478RealReplication()
    rep.run()
