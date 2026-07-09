# 配置 API

配置框架。叙述性指南见 [配置系统](../../explanation/config-system.md)。

## 核心

::: asset_embeddings.configs.Config

::: asset_embeddings.configs.ConfigField

::: asset_embeddings.configs.ConfigContainer

## 约束

::: asset_embeddings.configs.constraints
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - BaseConstraint
        - Constraint
        - TypeConstraint
        - RangeConstraint
        - ChoiceConstraint
        - ChoicesConstraint
        - EqualityConstraint
        - LengthConstraint
        - LambdaConstraint
        - NoneConstraint
        - NoConstraint
        - OptionalConstraint
        - InverseConstraint
        - IntersectionConstraints
        - UnionConstraints
        - CrossConstraint
        - LambdaCrossConstraint
        - MutualExclusionConstraint
        - MethodCrossConstraint
        - CrossIntersectionConstraints
        - CrossUnionConstraints
        - cross_validate
        - ConstraintError
