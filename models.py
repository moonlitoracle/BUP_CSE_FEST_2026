"""
Pydantic models for the GridWise Energy Optimization API.

Defines exact request/response schemas matching the BUP CSE Fest 2026
Problem Statement specification.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class DirectiveType(str, Enum):
    solar_reduction = "solar_reduction"
    minimum_battery_reserve = "minimum_battery_reserve"
    no_charge_window = "no_charge_window"
    no_discharge_window = "no_discharge_window"
    max_grid_window = "max_grid_window"
    no_op = "no_op"


class BatteryAction(str, Enum):
    charge = "charge"
    discharge = "discharge"
    idle = "idle"


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class HourEntry(BaseModel):
    """One hourly data point: demand, solar forecast, and tariff."""
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class Battery(BaseModel):
    """Battery system parameters for the scenario."""
    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_battery_consistency(self) -> "Battery":
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        return self


class OptimizationRequest(BaseModel):
    """
    POST /optimize-energy request body.

    Contains the scenario ID, 1-3 operator notes, 24 hourly entries,
    and battery configuration.
    """
    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourEntry] = Field(..., min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def validate_notes_non_empty(cls, v: List[str]) -> List[str]:
        for i, note in enumerate(v):
            if not note.strip():
                raise ValueError(f"operator_notes[{i}] must be a non-empty string")
        return v

    @field_validator("hours")
    @classmethod
    def validate_hours_unique_and_sorted(cls, v: List[HourEntry]) -> List[HourEntry]:
        hour_vals = [h.hour for h in v]
        if sorted(hour_vals) != list(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0 through 23")
        return sorted(v, key=lambda h: h.hour)


# ──────────────────────────────────────────────
# Structured Adjustment Models
# ──────────────────────────────────────────────

class SolarReductionAdjustment(BaseModel):
    """structured_adjustment for solar_reduction directives."""
    hours: List[int]
    factor: float = Field(..., ge=0.0, le=1.0)


class MinimumBatteryReserveAdjustment(BaseModel):
    """structured_adjustment for minimum_battery_reserve directives."""
    hours: List[int]
    minimum_energy_kwh: float = Field(..., ge=0)


class WindowAdjustment(BaseModel):
    """structured_adjustment for no_charge_window / no_discharge_window."""
    hours: List[int]


class MaxGridWindowAdjustment(BaseModel):
    """structured_adjustment for max_grid_window directives."""
    hours: List[int]
    max_grid_kwh: float = Field(..., ge=0)


# Union of all structured adjustment types
StructuredAdjustment = Union[
    SolarReductionAdjustment,
    MinimumBatteryReserveAdjustment,
    WindowAdjustment,
    MaxGridWindowAdjustment,
]


# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────

class DirectiveInterpretation(BaseModel):
    """
    One machine-checkable interpretation entry for a single operator note.
    Returned in note_index order (0, 1, ... N-1).
    """
    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[StructuredAdjustment] = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    """One hour of the final 24-hour energy schedule."""
    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0)
    solar_used_kwh: float = Field(..., ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0)
    battery_energy_after_kwh: float = Field(..., ge=0)


class OptimizationResponse(BaseModel):
    """
    POST /optimize-energy response body.

    Contains the scenario ID, directive interpretations, 24-hour plan,
    aggregate totals, and a human-readable plan summary.
    """
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry] = Field(..., min_length=24, max_length=24)
    total_grid_kwh: float = Field(..., ge=0)
    total_cost_bdt: float = Field(..., ge=0)
    peak_grid_kwh: float = Field(..., ge=0)
    plan_summary: str


# ──────────────────────────────────────────────
# Health Response
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health response."""
    status: Literal["ok"] = "ok"
