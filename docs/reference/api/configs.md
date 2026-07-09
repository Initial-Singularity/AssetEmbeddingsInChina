# Configuration API

The configuration framework. For the narrative guide see
[Configuration system](../../explanation/config-system.md).

## Core

::: asset_embeddings.configs.Config

::: asset_embeddings.configs.ConfigField

::: asset_embeddings.configs.ConfigContainer

## Constraints

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
