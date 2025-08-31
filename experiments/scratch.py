import pandas as pd

df = pd.read_csv("experiments/correlation/20250817_185145/correlation_experiment_results_synthetic_linear.csv")

# convert to long format
id_cols = ["regime_name", "starting_width", "init_id"]

h_cols = [c for c in df.columns if c.startswith("delta_test_loss_at_h")]

metric_cols = [c for c in df.columns if c not in id_cols+h_cols]

df_long = df.melt(id_vars=id_cols+h_cols, value_vars=metric_cols, var_name="metric", value_name="value")

tmp_agg_names = [c for c in metric_cols if c.startswith("tmp")]
atomic_names = [c for c in metric_cols if c not in tmp_agg_names]

tmp_df_long = df_long[df_long['metric'].isin(tmp_agg_names)]
atomic_df_long = df_long[df_long['metric'].isin(atomic_names)]

tmp_df_long['tmp_agg_metric'] = tmp_df_long['metric'].str.split('__').str[1]
tmp_df_long['atomic_metric'] = tmp_df_long['metric'].str.split('__').str[2]

# save to csv
tmp_df_long_nans = tmp_df_long[tmp_df_long['value'].isna()]
atomic_df_long_nans = atomic_df_long[atomic_df_long['value'].isna()]

tmp_agg_metric_nans = tmp_df_long_nans['tmp_agg_metric'].unique()
tmp_atomic_metric_nans = tmp_df_long_nans['atomic_metric'].unique()
atomic_metric_nans = atomic_df_long_nans['metric'].unique()

tmp_agg_metric_nans
# make a joint dataframe and print a 


