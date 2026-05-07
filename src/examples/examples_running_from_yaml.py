import os
import yaml
from pathlib import Path
from src.stats_transformer.pipeline import Pipeline

class RunExamplesFromYaml:
    def __init__(self):
        self.config_dir = Path("references/configs/examples")
        self.configs = [
            "mincer_wage.yaml",
            "grunfeld_panel.yaml",
            "okuns_law.yaml",
            "mroz_iv.yaml",
            "nakamura_steinsson_pca.yaml",
            "longley.yaml"
        ]

    def run(self):
        for config_file in self.configs:
            config_path = self.config_dir / config_file
            print(f"\n{'='*50}")
            print(f"Running pipeline with config: {config_path}")
            print(f"{'='*50}")
            
            if not config_path.exists():
                print(f"Config file {config_path} does not exist. Skipping.")
                continue
                
            with open(config_path) as f:
                config_content = yaml.safe_load(f)
                print("YAML Configuration:")
                print(yaml.dump(config_content, default_flow_style=False))
                
            try:
                # Initialize pipeline as shown in README.md
                pipeline = Pipeline(params_path=str(config_path))
                
                print("--- Running regression stage ---")
                # Using regression stage as an example of pipeline execution
                model_results = pipeline.run(stage="regression")
                print("Stage completed successfully.")
                
            except FileNotFoundError as e:
                print(f"Note: Could not run example. The required data file might be missing: {e}")
                print("You may need to run the corresponding sanity script in src/examples/regression/ first to fetch and prepare the data.")
            except Exception as e:
                print(f"Error running pipeline for {config_file}: {e}")

if __name__ == "__main__":
    runner = RunExamplesFromYaml()
    runner.run()
