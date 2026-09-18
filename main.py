"""
GridWise Energy Optimization API — Main Application.

BUP CSE Fest 2026 Hackathon Preliminary.
LLM-Assisted Operator Directive Interpretation & 24-Hour Energy Scheduling.
"""

from __future__ import annotations

import logging
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from guardrails import GuardrailError, validate_directives
from interpreter import interpret_notes
from models import HealthResponse, OptimizationRequest, OptimizationResponse
from optimizer import optimize
from validator import ValidationError, compute_totals, generate_plan_summary, verify_schedule

# Load .env file if present
load_dotenv()

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="GridWise Energy Optimization API",
    description="LLM-assisted 24-hour campus energy scheduling and optimization.",
    version="1.0.0",
)


# ──────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return HTTP 200 with {"status": "ok"} when the service is ready."""
    return HealthResponse(status="ok")


# ──────────────────────────────────────────────
# Optimize Energy Endpoint
# ──────────────────────────────────────────────

@app.post("/optimize-energy", response_model=OptimizationResponse)
async def optimize_energy(request: OptimizationRequest) -> OptimizationResponse:
    """
    Accept one scenario JSON and return an interpretation + optimization-plan.

    Pipeline:
      1. LLM interprets operator notes → raw directives
      2. Guardrails validate and sanitize the raw directives
      3. Optimizer solves the 24-hour schedule with PuLP
      4. Validator replays the schedule to confirm all constraints
      5. Compute totals and assemble response
    """
    logger.info(
        "Processing scenario '%s' with %d operator note(s)",
        request.scenario_id,
        len(request.operator_notes),
    )

    try:
        # ── Step 1: LLM Interpretation ──
        logger.info("Step 1: Interpreting operator notes via LLM...")
        raw_directives = await interpret_notes(
            operator_notes=request.operator_notes,
            battery_capacity_kwh=request.battery.capacity_kwh,
        )

        # ── Step 2: Guardrail Validation ──
        logger.info("Step 2: Applying deterministic guardrails...")
        validated_interpretations = validate_directives(
            raw_directives=raw_directives,
            num_notes=len(request.operator_notes),
            battery=request.battery,
        )

        # ── Step 3: Mathematical Optimization ──
        logger.info("Step 3: Running PuLP optimization...")
        hourly_plan, _ = optimize(
            hours=request.hours,
            battery=request.battery,
            interpretations=validated_interpretations,
        )

        # ── Step 4: Post-Optimization Verification ──
        logger.info("Step 4: Running post-optimization verification...")
        verify_schedule(
            hourly_plan=hourly_plan,
            hours=request.hours,
            battery=request.battery,
            interpretations=validated_interpretations,
        )

        # ── Step 5: Compute Totals & Assemble Response ──
        logger.info("Step 5: Computing totals and assembling response...")
        total_grid, total_cost, peak_grid = compute_totals(hourly_plan, request.hours)
        summary = generate_plan_summary(
            total_grid, total_cost, peak_grid, validated_interpretations
        )

        response = OptimizationResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=validated_interpretations,
            hourly_plan=hourly_plan,
            total_grid_kwh=total_grid,
            total_cost_bdt=total_cost,
            peak_grid_kwh=peak_grid,
            plan_summary=summary,
        )

        logger.info(
            "Scenario '%s' completed: cost=%.2f BDT, grid=%.2f kWh, peak=%.2f kWh",
            request.scenario_id, total_cost, total_grid, peak_grid,
        )
        return response

    except GuardrailError as e:
        logger.error("Guardrail validation failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Guardrail validation failed: {e}")

    except ValidationError as e:
        logger.error("Post-optimization validation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Schedule validation failed: {e}")

    except ValueError as e:
        logger.error("Value error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    except RuntimeError as e:
        logger.error("Optimization error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error("Unexpected error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please check the server logs.",
        )


# ──────────────────────────────────────────────
# Exception Handler (prevents raw stack traces)
# ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch unhandled exceptions and return a controlled 500 response."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
    )


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
