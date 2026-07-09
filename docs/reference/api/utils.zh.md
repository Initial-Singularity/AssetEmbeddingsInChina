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

## 错误处理

::: asset_embeddings.utils.log_utils.log_exceptions_inclass
