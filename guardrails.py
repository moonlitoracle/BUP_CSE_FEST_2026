"""
Deterministic Guardrails & Validation Layer for GridWise.

Processes and sanitizes raw LLM outputs before any mathematical
optimization occurs. Ensures all interpreted directives conform
to the Problem Statement specification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from models import (
    Battery,
    DirectiveInterpretation,
    DirectiveType,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    SolarReductionAdjustment,
    WindowAdjustment,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Guardrail Errors
# ──────────────────────────────────────────────

class GuardrailError(Exception):
    """Raised when LLM output fails deterministic validation."""
    pass


# ──────────────────────────────────────────────
# Validation Functions
# ──────────────────────────────────────────────

VALID_DIRECTIVE_TYPES = {dt.value for dt in DirectiveType}


def _validate_note_mapping(
    raw_directives: List[Dict[str, Any]],
    num_notes: int,
) -> None:
    """
    Validate that note_index maps correctly to every original note
    without gaps or duplicates.
    """
    if len(raw_directives) != num_notes:
        raise GuardrailError(
            f"Expected {num_notes} directive(s), got {len(raw_directives)}. "
            f"Each operator note must produce exactly one interpretation entry."
        )

    indices = [d.get("note_index") for d in raw_directives]

    # Check for non-integer indices
    for i, idx in enumerate(indices):
        if not isinstance(idx, int):
            raise GuardrailError(
                f"directive[{i}].note_index must be an integer, got {type(idx).__name__}"
            )

    expected = list(range(num_notes))
    if sorted(indices) != expected:
        raise GuardrailError(
            f"note_index values must be {expected} (no gaps/duplicates), got {sorted(indices)}"
        )

    # Check sequential order
    if indices != expected:
        raise GuardrailError(
            f"Directives must be in note_index order {expected}, got {indices}"
        )


def _validate_directive_type(directive: Dict[str, Any], index: int) -> str:
    """Validate that directive_type is one of the supported values."""
    dtype = directive.get("directive_type")
    if dtype not in VALID_DIRECTIVE_TYPES:
        raise GuardrailError(
            f"directive[{index}].directive_type '{dtype}' is not supported. "
            f"Must be one of: {', '.join(sorted(VALID_DIRECTIVE_TYPES))}"
        )
    return dtype


def _validate_applies_semantics(directive: Dict[str, Any], dtype: str, index: int) -> bool:
    """
    Validate that applies is false only for no_op (with structured_adjustment=null)
    and true for all other directives.
    """
    applies = directive.get("applies")
    adjustment = directive.get("structured_adjustment")

    if dtype == "no_op":
        if applies is not False:
            # Auto-correct: force applies=False for no_op
            logger.warning(
                "directive[%d] is no_op but applies=%s; correcting to false", index, applies
            )
            directive["applies"] = False
            applies = False
        if adjustment is not None:
            # Auto-correct: force null adjustment for no_op
            logger.warning(
                "directive[%d] is no_op but structured_adjustment is not null; correcting",
                index,
            )
            directive["structured_adjustment"] = None
    else:
        if applies is not True:
            # Auto-correct: force applies=True for non-no_op
            logger.warning(
                "directive[%d] type=%s but applies=%s; correcting to true",
                index, dtype, applies,
            )
            directive["applies"] = True
            applies = True
        if adjustment is None:
            raise GuardrailError(
                f"directive[{index}] type={dtype} requires a structured_adjustment, got null"
            )

    return directive["applies"]


def _validate_hours_array(hours: Any, index: int) -> List[int]:
    """
    Validate that time windows contain unique integer hours from 0 to 23,
    sorted in ascending order.
    """
    if not isinstance(hours, list):
        raise GuardrailError(
            f"directive[{index}].structured_adjustment.hours must be a list"
        )

    if len(hours) == 0:
        raise GuardrailError(
            f"directive[{index}].structured_adjustment.hours must not be empty"
        )

    for h in hours:
        if not isinstance(h, int):
            raise GuardrailError(
                f"directive[{index}].structured_adjustment.hours contains non-integer: {h}"
            )
        if h < 0 or h > 23:
            raise GuardrailError(
                f"directive[{index}].structured_adjustment.hours contains invalid hour: {h}"
            )

    if len(hours) != len(set(hours)):
        raise GuardrailError(
            f"directive[{index}].structured_adjustment.hours contains duplicates"
        )

    sorted_hours = sorted(hours)
    if hours != sorted_hours:
        # Auto-correct: sort hours
        logger.warning(
            "directive[%d].hours not in ascending order; auto-sorting", index
        )
        hours = sorted_hours

    return hours


def _validate_structured_adjustment(
    directive: Dict[str, Any],
    dtype: str,
    index: int,
    battery: Battery,
) -> Optional[Any]:
    """
    Validate the structured_adjustment for a non-no_op directive.
    Returns the validated Pydantic model.
    """
    adj = directive.get("structured_adjustment")
    if adj is None:
        return None  # no_op case handled elsewhere

    if not isinstance(adj, dict):
        raise GuardrailError(
            f"directive[{index}].structured_adjustment must be an object, "
            f"got {type(adj).__name__}"
        )

    # Validate hours array (present in all non-no_op adjustments)
    if "hours" not in adj:
        raise GuardrailError(
            f"directive[{index}].structured_adjustment missing required field 'hours'"
        )
    adj["hours"] = _validate_hours_array(adj["hours"], index)

    if dtype == "solar_reduction":
        factor = adj.get("factor")
        if factor is None:
            raise GuardrailError(
                f"directive[{index}] solar_reduction missing 'factor'"
            )
        if not isinstance(factor, (int, float)):
            raise GuardrailError(
                f"directive[{index}] solar_reduction factor must be a number"
            )
        if factor < 0 or factor > 1:
            raise GuardrailError(
                f"directive[{index}] solar_reduction factor must be between 0 and 1, "
                f"got {factor}"
            )
        return SolarReductionAdjustment(hours=adj["hours"], factor=float(factor))

    elif dtype == "minimum_battery_reserve":
        min_kwh = adj.get("minimum_energy_kwh")
        if min_kwh is None:
            raise GuardrailError(
                f"directive[{index}] minimum_battery_reserve missing 'minimum_energy_kwh'"
            )
        if not isinstance(min_kwh, (int, float)):
            raise GuardrailError(
                f"directive[{index}] minimum_battery_reserve minimum_energy_kwh must be a number"
            )
        if min_kwh < 0:
            raise GuardrailError(
                f"directive[{index}] minimum_battery_reserve minimum_energy_kwh must be >= 0"
            )
        if min_kwh > battery.capacity_kwh:
            raise GuardrailError(
                f"directive[{index}] minimum_battery_reserve minimum_energy_kwh ({min_kwh}) "
                f"exceeds battery capacity ({battery.capacity_kwh})"
            )
        return MinimumBatteryReserveAdjustment(
            hours=adj["hours"], minimum_energy_kwh=float(min_kwh)
        )

    elif dtype in ("no_charge_window", "no_discharge_window"):
        return WindowAdjustment(hours=adj["hours"])

    elif dtype == "max_grid_window":
        max_grid = adj.get("max_grid_kwh")
        if max_grid is None:
            raise GuardrailError(
                f"directive[{index}] max_grid_window missing 'max_grid_kwh'"
            )
        if not isinstance(max_grid, (int, float)):
            raise GuardrailError(
                f"directive[{index}] max_grid_window max_grid_kwh must be a number"
            )
        if max_grid < 0:
            raise GuardrailError(
                f"directive[{index}] max_grid_window max_grid_kwh must be >= 0"
            )
        return MaxGridWindowAdjustment(hours=adj["hours"], max_grid_kwh=float(max_grid))

    else:
        raise GuardrailError(f"directive[{index}] unhandled type: {dtype}")


# ──────────────────────────────────────────────
# Main Guardrail Entry Point
# ──────────────────────────────────────────────

def validate_directives(
    raw_directives: List[Dict[str, Any]],
    num_notes: int,
    battery: Battery,
) -> List[DirectiveInterpretation]:
    """
    Validate and sanitize raw LLM-produced directive interpretations.

    Args:
        raw_directives: List of dicts from the LLM.
        num_notes: Number of original operator notes.
        battery: Battery configuration for bound checks.

    Returns:
        List of validated DirectiveInterpretation objects.

    Raises:
        GuardrailError: If validation fails and cannot be auto-corrected.
    """
    logger.info("Validating %d raw directive(s) against %d note(s)", len(raw_directives), num_notes)

    # Step 1: Validate note mapping
    _validate_note_mapping(raw_directives, num_notes)

    validated: List[DirectiveInterpretation] = []

    for i, directive in enumerate(raw_directives):
        # Step 2: Validate directive_type
        dtype = _validate_directive_type(directive, i)

        # Step 3: Validate applies semantics
        _validate_applies_semantics(directive, dtype, i)

        # Step 4: Validate structured_adjustment
        if dtype == "no_op":
            structured = None
        else:
            structured = _validate_structured_adjustment(directive, dtype, i, battery)

        # Step 5: Build validated interpretation
        explanation = directive.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)

        validated.append(DirectiveInterpretation(
            note_index=directive["note_index"],
            applies=directive["applies"],
            directive_type=DirectiveType(dtype),
            structured_adjustment=structured,
            explanation=explanation,
        ))

    logger.info("All %d directive(s) passed guardrail validation", len(validated))
    return validated
