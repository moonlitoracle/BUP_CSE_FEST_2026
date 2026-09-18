"""
Mathematical Optimization Engine for GridWise.

Uses PuLP to schedule a 24-hour energy horizon minimizing total grid cost
while respecting all physical constraints and validated operator directives.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pulp

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


# ──────────────────────────────────────────────
# Directive Extraction Helpers
# ──────────────────────────────────────────────

def _extract_directives(
    interpretations: List[DirectiveInterpretation],
) -> Dict[str, list]:
    """
    Group validated directives by type for easy constraint application.

    Returns a dict with keys for each directive type, containing the
    list of applicable DirectiveInterpretation objects.
    """
    grouped: Dict[str, list] = {
        "solar_reduction": [],
        "minimum_battery_reserve": [],
        "no_charge_window": [],
        "no_discharge_window": [],
        "max_grid_window": [],
    }

    for interp in interpretations:
        if interp.applies and interp.directive_type != DirectiveType.no_op:
            grouped[interp.directive_type.value].append(interp)

    return grouped


def _compute_effective_solar(
    hours: List[HourEntry],
    solar_directives: List[DirectiveInterpretation],
) -> Dict[int, float]:
    """
    Compute effective solar for each hour after applying solar_reduction directives.
    """
    effective: Dict[int, float] = {}
    for h in hours:
        effective[h.hour] = h.solar_kwh

    for directive in solar_directives:
        adj: SolarReductionAdjustment = directive.structured_adjustment
        for hour in adj.hours:
            effective[hour] = hours[hour].solar_kwh * adj.factor

    return effective


# ──────────────────────────────────────────────
# PuLP Optimizer
# ──────────────────────────────────────────────

def optimize(
    hours: List[HourEntry],
    battery: Battery,
    interpretations: List[DirectiveInterpretation],
) -> Tuple[List[HourlyPlanEntry], float]:
    """
    Solve the 24-hour energy optimization problem.

    Args:
        hours: List of 24 HourEntry objects (sorted by hour).
        battery: Battery configuration.
        interpretations: Validated directive interpretations.

    Returns:
        Tuple of (hourly_plan, total_cost_bdt).

    Raises:
        RuntimeError: If the optimization problem is infeasible.
    """
    directives = _extract_directives(interpretations)
    effective_solar = _compute_effective_solar(hours, directives["solar_reduction"])

    # ──────────────────────────────────────────
    # Create the LP problem
    # ──────────────────────────────────────────
    prob = pulp.LpProblem("GridWise_EnergyOptimization", pulp.LpMinimize)

    # ──────────────────────────────────────────
    # Decision variables
    # ──────────────────────────────────────────
    H = range(24)

    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in H]
    solar_used = [pulp.LpVariable(f"solar_{h}", lowBound=0) for h in H]
    charge = [pulp.LpVariable(f"charge_{h}", lowBound=0) for h in H]
    discharge = [pulp.LpVariable(f"discharge_{h}", lowBound=0) for h in H]
    energy_after = [pulp.LpVariable(f"energy_{h}", lowBound=0) for h in H]

    # ──────────────────────────────────────────
    # Objective: minimize total grid cost
    # ──────────────────────────────────────────
    prob += pulp.lpSum(
        grid[h] * hours[h].tariff_bdt_per_kwh for h in H
    ), "TotalGridCost"

    # ──────────────────────────────────────────
    # Base constraints for each hour
    # ──────────────────────────────────────────

    # Collect directive hours into sets for fast lookup
    no_charge_hours = set()
    for d in directives["no_charge_window"]:
        adj: WindowAdjustment = d.structured_adjustment
        no_charge_hours.update(adj.hours)

    no_discharge_hours = set()
    for d in directives["no_discharge_window"]:
        adj: WindowAdjustment = d.structured_adjustment
        no_discharge_hours.update(adj.hours)

    max_grid_caps: Dict[int, float] = {}
    for d in directives["max_grid_window"]:
        adj: MaxGridWindowAdjustment = d.structured_adjustment
        for hour in adj.hours:
            if hour in max_grid_caps:
                # If multiple directives affect the same hour, use the tighter cap
                max_grid_caps[hour] = min(max_grid_caps[hour], adj.max_grid_kwh)
            else:
                max_grid_caps[hour] = adj.max_grid_kwh

    # Compute effective minimum battery reserve per hour
    min_reserve: Dict[int, float] = {}
    for h in H:
        min_reserve[h] = battery.minimum_energy_kwh
    for d in directives["minimum_battery_reserve"]:
        adj: MinimumBatteryReserveAdjustment = d.structured_adjustment
        for hour in adj.hours:
            min_reserve[hour] = max(min_reserve[hour], adj.minimum_energy_kwh)

    for h in H:
        # ── Energy balance ──
        # grid + solar_used + discharge == demand + charge
        prob += (
            grid[h] + solar_used[h] + discharge[h]
            == hours[h].demand_kwh + charge[h]
        ), f"EnergyBalance_{h}"

        # ── Solar usage cap ──
        prob += (
            solar_used[h] <= effective_solar[h]
        ), f"SolarCap_{h}"

        # ── Charge rate limit ──
        prob += (
            charge[h] <= battery.max_charge_kwh_per_hour
        ), f"ChargeRate_{h}"

        # ── Discharge rate limit ──
        prob += (
            discharge[h] <= battery.max_discharge_kwh_per_hour
        ), f"DischargeRate_{h}"

        # ── Battery state transition ──
        if h == 0:
            prob += (
                energy_after[h] == battery.initial_energy_kwh + charge[h] - discharge[h]
            ), f"BatteryTransition_{h}"
        else:
            prob += (
                energy_after[h] == energy_after[h - 1] + charge[h] - discharge[h]
            ), f"BatteryTransition_{h}"

        # ── Battery capacity upper bound ──
        prob += (
            energy_after[h] <= battery.capacity_kwh
        ), f"BatteryCapacity_{h}"

        # ── Battery minimum reserve ──
        prob += (
            energy_after[h] >= min_reserve[h]
        ), f"BatteryMinReserve_{h}"

        # ── No-charge window ──
        if h in no_charge_hours:
            prob += charge[h] == 0, f"NoCharge_{h}"

        # ── No-discharge window ──
        if h in no_discharge_hours:
            prob += discharge[h] == 0, f"NoDischarge_{h}"

        # ── Max grid window ──
        if h in max_grid_caps:
            prob += grid[h] <= max_grid_caps[h], f"MaxGrid_{h}"

    # ── End-of-day battery neutrality ──
    prob += (
        energy_after[23] == battery.initial_energy_kwh
    ), "EndOfDayNeutrality"

    # ──────────────────────────────────────────
    # Solve
    # ──────────────────────────────────────────
    logger.info("Solving optimization problem...")
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"Optimization failed with status: {pulp.LpStatus[status]}. "
            f"The problem may be infeasible given the constraints and directives."
        )

    total_cost = pulp.value(prob.objective)
    logger.info("Optimization complete. Total cost: %.2f BDT", total_cost)

    # ──────────────────────────────────────────
    # Extract hourly plan
    # ──────────────────────────────────────────
    hourly_plan: List[HourlyPlanEntry] = []

    for h in H:
        g = round(pulp.value(grid[h]), 4)
        s = round(pulp.value(solar_used[h]), 4)
        c = round(pulp.value(charge[h]), 4)
        d = round(pulp.value(discharge[h]), 4)
        e = round(pulp.value(energy_after[h]), 4)

        # Determine battery action
        # Use a small epsilon to handle floating point
        eps = 1e-6
        if c > eps and d > eps:
            # Simultaneous charge and discharge: net to one action
            if c > d:
                c = c - d
                d = 0.0
            else:
                d = d - c
                c = 0.0

        if c > eps:
            action = BatteryAction.charge
            batt_kwh = c
        elif d > eps:
            action = BatteryAction.discharge
            batt_kwh = d
        else:
            action = BatteryAction.idle
            batt_kwh = 0.0

        hourly_plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=round(g, 2),
            solar_used_kwh=round(s, 2),
            battery_action=action,
            battery_kwh=round(batt_kwh, 2),
            battery_energy_after_kwh=round(e, 2),
        ))

    return hourly_plan, round(total_cost, 2)
