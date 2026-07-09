# Utilities API

## Seeding

The single source of seeding policy.

::: asset_embeddings.utils.seeding
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - seed_basic
        - seed_hf_global
        - seed_accelerate

## Accumulators

::: asset_embeddings.utils.stats_utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - MeanVarianceTracker
        - MinMaxTracker
        - EarlyStopTracker

## Error handling

::: asset_embeddings.utils.log_utils.log_exceptions_inclass
