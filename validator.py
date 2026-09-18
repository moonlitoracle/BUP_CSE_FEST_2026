"""
Post-Optimization Verification for GridWise.

Independently replays the hourly schedule to guarantee all constraints
and operator directives are strictly met. Computes response totals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from models import (
    Battery,
    BatteryAction,
    DirectiveInterpretation,
    DirectiveType,
    HourEntry,
    HourlyPlanEntry,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    SolarReductionAdjustment,
    WindowAdjustment,
)

logger = logging.getLogger(__name__)

TOLERANCE = 0.01  # kWh / BDT tolerance per the problem statement


class ValidationError(Exception):
    """Raised when the post-optimization replay detects a constraint violation."""
    pass


def _compute_effective_solar(
    hours: List[HourEntry],
    interpretations: List[DirectiveInterpretation],
) -> Dict[int, float]:
    """Compute effective solar per hour after solar_reduction directives."""
    effective: Dict[int, float] = {}
    for h in hours:
        effective[h.hour] = h.solar_kwh

    for interp in interpretations:
        if interp.directive_type == DirectiveType.solar_reduction and interp.applies:
            adj: SolarReductionAdjustment = interp.structured_adjustment
            for hour in adj.hours:
                effective[hour] = hours[hour].solar_kwh * adj.factor

    return effective


def _collect_directive_constraints(
    interpretations: List[DirectiveInterpretation],
    battery: Battery,
) -> Tuple[set, set, Dict[int, float], Dict[int, float]]:
    """
    Extract constraint sets from directives.

    Returns:
        (no_charge_hours, no_discharge_hours, max_grid_caps, min_reserves)
    """
    no_charge_hours = set()
    no_discharge_hours = set()
    max_grid_caps: Dict[int, float] = {}
    min_reserves: Dict[int, float] = {h: battery.minimum_energy_kwh for h in range(24)}

    for interp in interpretations:
        if not interp.applies:
            continue

        if interp.directive_type == DirectiveType.no_charge_window:
            adj: WindowAdjustment = interp.structured_adjustment
            no_charge_hours.update(adj.hours)

        elif interp.directive_type == DirectiveType.no_discharge_window:
            adj: WindowAdjustment = interp.structured_adjustment
            no_discharge_hours.update(adj.hours)

        elif interp.directive_type == DirectiveType.max_grid_window:
            adj: MaxGridWindowAdjustment = interp.structured_adjustment
            for hour in adj.hours:
                if hour in max_grid_caps:
                    max_grid_caps[hour] = min(max_grid_caps[hour], adj.max_grid_kwh)
                else:
                    max_grid_caps[hour] = adj.max_grid_kwh

        elif interp.directive_type == DirectiveType.minimum_battery_reserve:
            adj: MinimumBatteryReserveAdjustment = interp.structured_adjustment
            for hour in adj.hours:
                min_reserves[hour] = max(min_reserves[hour], adj.minimum_energy_kwh)

    return no_charge_hours, no_discharge_hours, max_grid_caps, min_reserves


def verify_schedule(
    hourly_plan: List[HourlyPlanEntry],
    hours: List[HourEntry],
    battery: Battery,
    interpretations: List[DirectiveInterpretation],
) -> None:
    """
    Deterministic replay of the hourly schedule.

    Checks all energy balance, battery, solar, and directive constraints.

    Raises:
        ValidationError: If any constraint is violated.
    """
    effective_solar = _compute_effective_solar(hours, interpretations)
    no_charge, no_discharge, max_grid, min_reserves = _collect_directive_constraints(
        interpretations, battery
    )

    # Verify hourly_plan covers hours 0-23 exactly
    plan_hours = sorted(entry.hour for entry in hourly_plan)
    if plan_hours != list(range(24)):
        raise ValidationError(
            f"hourly_plan must contain exactly hours 0-23, got {plan_hours}"
        )

    # Sort plan by hour
    plan_by_hour = {entry.hour: entry for entry in hourly_plan}

    prev_energy = battery.initial_energy_kwh
    errors = []

    for h in range(24):
        entry = plan_by_hour[h]
        hour_data = hours[h]

        # ── Non-negative values ──
        if entry.grid_kwh < -TOLERANCE:
            errors.append(f"Hour {h}: grid_kwh ({entry.grid_kwh}) is negative")
        if entry.solar_used_kwh < -TOLERANCE:
            errors.append(f"Hour {h}: solar_used_kwh ({entry.solar_used_kwh}) is negative")
        if entry.battery_kwh < -TOLERANCE:
            errors.append(f"Hour {h}: battery_kwh ({entry.battery_kwh}) is negative")

        # ── Solar usage <= effective solar ──
        eff_solar = effective_solar[h]
        if entry.solar_used_kwh > eff_solar + TOLERANCE:
            errors.append(
                f"Hour {h}: solar_used_kwh ({entry.solar_used_kwh}) exceeds "
                f"effective solar ({eff_solar})"
            )

        # ── Energy balance ──
        # grid + solar_used + discharge == demand + charge
        if entry.battery_action == BatteryAction.discharge:
            lhs = entry.grid_kwh + entry.solar_used_kwh + entry.battery_kwh
            rhs = hour_data.demand_kwh
        elif entry.battery_action == BatteryAction.charge:
            lhs = entry.grid_kwh + entry.solar_used_kwh
            rhs = hour_data.demand_kwh + entry.battery_kwh
        else:  # idle
            lhs = entry.grid_kwh + entry.solar_used_kwh
            rhs = hour_data.demand_kwh

        if abs(lhs - rhs) > TOLERANCE:
            errors.append(
                f"Hour {h}: energy balance failed. "
                f"LHS={lhs:.4f}, RHS={rhs:.4f}, diff={abs(lhs - rhs):.4f}"
            )

        # ── Battery state transition ──
        if entry.battery_action == BatteryAction.charge:
            expected_energy = prev_energy + entry.battery_kwh
        elif entry.battery_action == BatteryAction.discharge:
            expected_energy = prev_energy - entry.battery_kwh
        else:
            expected_energy = prev_energy
            if entry.battery_kwh > TOLERANCE:
                errors.append(
                    f"Hour {h}: battery_action is idle but battery_kwh={entry.battery_kwh}"
                )

        if abs(entry.battery_energy_after_kwh - expected_energy) > TOLERANCE:
            errors.append(
                f"Hour {h}: battery_energy_after_kwh ({entry.battery_energy_after_kwh}) "
                f"!= expected ({expected_energy})"
            )

        # ── Battery capacity bounds ──
        if entry.battery_energy_after_kwh > battery.capacity_kwh + TOLERANCE:
            errors.append(
                f"Hour {h}: battery_energy_after_kwh ({entry.battery_energy_after_kwh}) "
                f"exceeds capacity ({battery.capacity_kwh})"
            )

        if entry.battery_energy_after_kwh < min_reserves[h] - TOLERANCE:
            errors.append(
                f"Hour {h}: battery_energy_after_kwh ({entry.battery_energy_after_kwh}) "
                f"below minimum reserve ({min_reserves[h]})"
            )

        # ── Charge/discharge rate limits ──
        if entry.battery_action == BatteryAction.charge:
            if entry.battery_kwh > battery.max_charge_kwh_per_hour + TOLERANCE:
                errors.append(
                    f"Hour {h}: charge ({entry.battery_kwh}) exceeds max rate "
                    f"({battery.max_charge_kwh_per_hour})"
                )
        elif entry.battery_action == BatteryAction.discharge:
            if entry.battery_kwh > battery.max_discharge_kwh_per_hour + TOLERANCE:
                errors.append(
                    f"Hour {h}: discharge ({entry.battery_kwh}) exceeds max rate "
                    f"({battery.max_discharge_kwh_per_hour})"
                )

        # ── No-charge window ──
        if h in no_charge and entry.battery_action == BatteryAction.charge:
            if entry.battery_kwh > TOLERANCE:
                errors.append(
                    f"Hour {h}: no_charge_window violated, "
                    f"charge={entry.battery_kwh}"
                )

        # ── No-discharge window ──
        if h in no_discharge and entry.battery_action == BatteryAction.discharge:
            if entry.battery_kwh > TOLERANCE:
                errors.append(
                    f"Hour {h}: no_discharge_window violated, "
                    f"discharge={entry.battery_kwh}"
                )

        # ── Max grid window ──
        if h in max_grid:
            if entry.grid_kwh > max_grid[h] + TOLERANCE:
                errors.append(
                    f"Hour {h}: grid_kwh ({entry.grid_kwh}) exceeds "
                    f"max_grid_kwh ({max_grid[h]})"
                )

        prev_energy = entry.battery_energy_after_kwh

    # ── End-of-day neutrality ──
    final_energy = plan_by_hour[23].battery_energy_after_kwh
    if abs(final_energy - battery.initial_energy_kwh) > TOLERANCE:
        errors.append(
            f"End-of-day neutrality: final energy ({final_energy}) "
            f"!= initial energy ({battery.initial_energy_kwh})"
        )

    if errors:
        error_msg = "Post-optimization validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValidationError(error_msg)

    logger.info("Post-optimization verification passed all checks")


def compute_totals(
    hourly_plan: List[HourlyPlanEntry],
    hours: List[HourEntry],
) -> Tuple[float, float, float]:
    """
    Compute response totals directly from the hourly_plan.

    Returns:
        (total_grid_kwh, total_cost_bdt, peak_grid_kwh)
    """
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    plan_by_hour = {entry.hour: entry for entry in hourly_plan}

    for h in range(24):
        entry = plan_by_hour[h]
        g = entry.grid_kwh

        total_grid += g
        total_cost += g * hours[h].tariff_bdt_per_kwh
        peak_grid = max(peak_grid, g)

    return round(total_grid, 2), round(total_cost, 2), round(peak_grid, 2)


def generate_plan_summary(
    total_grid: float,
    total_cost: float,
    peak_grid: float,
    interpretations: List[DirectiveInterpretation],
) -> str:
    """Generate a brief human-readable plan summary."""
    active = [i for i in interpretations if i.applies]
    ignored = [i for i in interpretations if not i.applies]

    parts = []

    if active:
        directive_descs = []
        for d in active:
            directive_descs.append(d.directive_type.value.replace("_", " "))
        parts.append(f"Applied {len(active)} directive(s): {', '.join(directive_descs)}.")

    if ignored:
        parts.append(f"Ignored {len(ignored)} irrelevant note(s).")

    parts.append(
        f"Optimized 24-hour schedule achieves total grid usage of {total_grid:.1f} kWh "
        f"at a cost of {total_cost:.1f} BDT with peak hourly grid import of {peak_grid:.1f} kWh. "
        f"Battery returns to initial energy level at end of day."
    )

    return " ".join(parts)
