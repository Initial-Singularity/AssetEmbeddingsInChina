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

## Resources

Every `Run` owns one probe; the trainers ask it for a line only when the VRAM high-water mark moved.

::: asset_embeddings.utils.memory_utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ResourceProbe

## Logging

See [Logging system](../../explanation/logging-system.md).

::: asset_embeddings.logger.configure_logging

::: asset_embeddings.logger.failure_report

::: asset_embeddings.logger.job_log_paths

::: asset_embeddings.logger.error_dir
