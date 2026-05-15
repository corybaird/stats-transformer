import logging
import pandas as pd

class SpecificationRunner:
    """
    Standardizes the process of running multiple regression specifications 
    (e.g., baseline, year FE, purged models, country-by-country) over a loop 
    of key variables, and compiles the results into a tidy format.
    """

    def __init__(self, base_model_cls, entity_col='country', time_col='date'):
        self.base_model_cls = base_model_cls
        self.entity_col = entity_col
        self.time_col = time_col
        self.specs = []
        self.results = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_spec(self, name, target, key_variables, controls=None, subset_mask_func=None, **model_kwargs):
        """
        Register a named specification to run.
        
        Args:
            name (str): The name of the specification (e.g., 'Baseline').
            target (str): Dependent variable.
            key_variables (list): List of variables of interest to loop over one by one.
                                  For each variable, a separate model is fit.
            controls (list, optional): Variables to include in every model as controls.
            subset_mask_func (callable, optional): A function taking the panel and returning a boolean Series.
            **model_kwargs: Additional kwargs to pass to the model class (e.g., entity_effects=True).
        """
        self.specs.append({
            'name': name,
            'target': target,
            'key_variables': key_variables,
            'controls': controls or [],
            'subset_mask_func': subset_mask_func,
            'model_kwargs': model_kwargs
        })

    def run(self, panel):
        """
        Execute all registered specifications against the provided panel.
        
        Args:
            panel (pd.DataFrame): The data panel.
            
        Returns:
            pd.DataFrame: A tidy long-form DataFrame of all results.
        """
        self.results = []
        self.logger.info(f"Running {len(self.specs)} registered specifications.")

        for spec in self.specs:
            self.logger.info(f"Running specification: {spec['name']}")
            target = spec['target']
            controls = spec['controls']
            kwargs = spec['model_kwargs']
            
            # Apply subsetting if specified
            df_spec = panel.copy()
            if spec['subset_mask_func']:
                mask = spec['subset_mask_func'](df_spec)
                df_spec = df_spec[mask]

            # Ensure entity and time cols are columns, not just in index
            if isinstance(df_spec.index, pd.MultiIndex):
                df_spec = df_spec.reset_index()
            elif self.entity_col in df_spec.index.names or self.time_col in df_spec.index.names:
                df_spec = df_spec.reset_index()

            for kv in spec['key_variables']:
                # The independent variables for this run are the key variable + controls
                indep_vars = [kv] + [c for c in controls if c != kv]
                
                # Check coverage
                missing_cols = [c for c in [target] + indep_vars if c not in df_spec.columns]
                if missing_cols:
                    self.logger.warning(f"Skipping {kv} in {spec['name']}: missing columns {missing_cols}")
                    continue
                
                try:
                    # Instantiate and fit the model
                    model = self.base_model_cls(
                        target=target,
                        independent_variables=indep_vars,
                        entity_column=self.entity_col,
                        time_column=self.time_col,
                        **kwargs
                    )
                    
                    # Instead of running the full pipeline fit which might save metadata, 
                    # we just load data and build the model to extract coefficients
                    model.load_data(df_spec)
                    model.build_model()
                    
                    metadata = model.get_model_metadata()
                    metrics = metadata.get('metrics', {})
                    coefs = metadata.get('coefficients', {})
                    
                    # Extract the coefficient of interest
                    kv_coef_data = coefs.get(kv, {})
                    
                    res_row = {
                        'spec_name': spec['name'],
                        'target': target,
                        'key_variable': kv,
                        'coef': kv_coef_data.get('value'),
                        'se': kv_coef_data.get('std_err'),
                        'pval': kv_coef_data.get('p_value'),
                        'n_obs': metrics.get('nobs'),
                        'r2_within': metrics.get('rsquared_within', metrics.get('rsquared'))
                    }
                    
                    # Extract control coefficients to store dynamically
                    for c in controls:
                        c_data = coefs.get(c, {})
                        res_row[f'{c}_coef'] = c_data.get('value')
                        res_row[f'{c}_se'] = c_data.get('std_err')
                        res_row[f'{c}_pval'] = c_data.get('p_value')
                        
                    self.results.append(res_row)
                    
                except Exception as e:
                    self.logger.error(f"Error running model for {kv} in {spec['name']}: {str(e)}")
                    
        return self.to_dataframe()

    def to_dataframe(self):
        """
        Returns a tidy DataFrame of the collected results.
        """
        return pd.DataFrame(self.results)
