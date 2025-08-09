# TODOs for Correlation Experiment Improvements

- Robust correlation analysis
  - Add Spearman correlation alongside Pearson; report n, unique counts, and bootstrap CIs.
  - Within-initialization rank correlation (per init Spearman between metric rank and delta; aggregate via Fisher z).

- Temporal features (windowed/EMA)
  - Maintain rolling window (K=20) in `experiments/stats_wrapper.py` for per-neuron:
    - post_activation mean/std, gradient magnitude mean/std
    - EMA of |activation|, |grad|, and their product
  - Compute at split time and store with stats.

- Composite features
  - L2/L1 norms for incoming/outgoing weights and their gradients (works for single-output too).
  - Fraction active (for ReLU/LeakyReLU) or average magnitude (for GELU) across batch.
  - Fisher proxy: mean squared gradient of post-activation.

- Scheduler/optimizer consistency
  - Consider re-instantiating both optimizer and scheduler for control and treatment in a single helper to avoid divergence in state handling.

- Dataset coverage and action timing
  - Run at least one classification and one regression dataset per sweep.
  - Sweep action timing (e.g., 30%, 50%, 70%) and include as a covariate.

- Result logging
  - Save `control_best_test_loss`, `delta_best_test_loss` to CSV (done).
  - Persist normalized features (`*_z`, `*_pct`) alongside raw (done).
  - Add per-column NA counts and n_unique to the printed correlation table.

- Degenerate metrics handling
  - For single-output regression, drop variance metrics on 1-length vectors entirely in favor of norms. Currently set to 0.0; consider removing from correlation to avoid misleading features.

References
- Normalization and within-layer ranking implemented in `experiments/test_correlation_split_and_train_step.py`.
- Safe variance for 1-D vectors implemented in `experiments/stats_wrapper.py`.
