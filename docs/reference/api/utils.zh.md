# 工具 API

## 随机种子

种子策略的唯一来源。

::: asset_embeddings.utils.seeding
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - seed_basic
        - seed_hf_global
        - seed_accelerate

## 累加器

::: asset_embeddings.utils.stats_utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - MeanVarianceTracker
        - MinMaxTracker
        - EarlyStopTracker

## 资源

每个 `Run` 持有一个探针；训练器只在 VRAM 高水位变化时向它索取一行记录。

::: asset_embeddings.utils.memory_utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ResourceProbe

## 日志

见 [日志系统](../../explanation/logging-system.md)。

::: asset_embeddings.logger.configure_logging

::: asset_embeddings.logger.failure_report

::: asset_embeddings.logger.job_log_paths

::: asset_embeddings.logger.error_dir
