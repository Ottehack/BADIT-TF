"""BADIT-TF experiment implementation.

The public API is deliberately small so every experiment uses the same
primitive, assignment, and routing implementation.
"""

from .assignment import (
    AssignmentResult,
    build_natural_modulation_profiles,
    labels_to_matrix,
    solve_balanced_tf_assignment,
    validate_assignment,
)
from .core import (
    AbilityInjectionResult,
    TFAbilityLinear,
    advance_router_step,
    inject_ability_layers,
    iter_ability_layers,
)

__all__ = [
    "AbilityInjectionResult",
    "AssignmentResult",
    "TFAbilityLinear",
    "advance_router_step",
    "build_natural_modulation_profiles",
    "inject_ability_layers",
    "iter_ability_layers",
    "labels_to_matrix",
    "solve_balanced_tf_assignment",
    "validate_assignment",
]

