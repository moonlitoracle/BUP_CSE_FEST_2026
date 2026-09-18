"""
LLM Interpretation Module for GridWise.

Uses Google Gemini (via google-genai) to interpret natural-language
operator notes into structured directive objects.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a structured-data extraction engine for the GridWise campus energy system.
You will receive 1 to 3 natural-language operator notes about a 24-hour energy schedule.

## Your task
For EACH note, produce exactly ONE JSON interpretation object. Return a JSON array with one entry per note, in the same order (note_index 0, 1, ... N-1).

## Supported directive types and their structured_adjustment shapes

1. **solar_reduction** — Reduce usable solar during specific hours.
   structured_adjustment: {"hours": [<int>, ...], "factor": <float 0-1>}
   - "factor" is the REMAINING usable fraction. Example: "80% reduction" → factor = 0.2; "reduced to 25%" → factor = 0.25.

2. **minimum_battery_reserve** — Keep battery energy at or above a required level during specific hours.
   structured_adjustment: {"hours": [<int>, ...], "minimum_energy_kwh": <float>}

3. **no_charge_window** — Battery charging is unavailable during specific hours.
   structured_adjustment: {"hours": [<int>, ...]}

4. **no_discharge_window** — Battery discharging is unavailable during specific hours.
   structured_adjustment: {"hours": [<int>, ...]}

5. **max_grid_window** — Grid import capped at a stated amount during specific hours.
   structured_adjustment: {"hours": [<int>, ...], "max_grid_kwh": <float>}

6. **no_op** — The note does NOT affect the current 24-hour energy schedule (irrelevant, about the future, about non-energy topics, etc.).
   structured_adjustment: null

## Time conversion rules
- Hours are integers 0–23 (0 = midnight, 12 = noon, 13 = 1 PM, etc.).
- Time windows are START-INCLUSIVE, END-EXCLUSIVE: "1 PM to 3 PM" → hours [13, 14].
- "noon until 2 PM" → hours [12, 13].
- "2 AM until 5 AM" → hours [2, 3, 4].
- "6 PM until 9 PM" → hours [18, 19, 20].
- Hours arrays must contain unique integers in ascending order.

## Output rules
- For no_op: set applies=false, directive_type="no_op", structured_adjustment=null.
- For ALL other directives: set applies=true.
- Do NOT invent demand, solar, tariff, battery parameters, or unsupported directive types.
- Do NOT combine multiple directives into one entry. Each note maps to exactly one directive.
- Do NOT add any extra fields beyond note_index, applies, directive_type, structured_adjustment, explanation.
- The "explanation" field should be a brief, clear sentence explaining the interpretation.

## CRITICAL RULES about solar_reduction factor
- "reduced to X%" means factor = X/100. E.g., "reduced to 25%" → factor = 0.25
- "X% reduction" means factor = (100-X)/100. E.g., "80% reduction" → factor = 0.20
- "roughly X% of the forecast" means factor = X/100. E.g., "roughly 25% of the forecast" → factor = 0.25
- "drop to about X%" means factor = X/100. E.g., "drop to about 20%" → factor = 0.20
- "one-fifth of normal" → factor = 0.2

## Output format
Return ONLY a valid JSON array. No markdown, no code fences, no commentary. Example:
[
  {
    "note_index": 0,
    "applies": true,
    "directive_type": "solar_reduction",
    "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
    "explanation": "Solar availability is reduced to 20% during maintenance."
  },
  {
    "note_index": 1,
    "applies": false,
    "directive_type": "no_op",
    "structured_adjustment": null,
    "explanation": "This note does not affect today's energy schedule."
  }
]
"""

# ──────────────────────────────────────────────
# Reserve percentage handling prompt addition
# ──────────────────────────────────────────────

PERCENTAGE_PROMPT = """
## CRITICAL RULES about minimum_battery_reserve with percentages
- If the note says "keep at least X% reserve" or "maintain X% battery", compute:
  minimum_energy_kwh = capacity_kwh * X / 100
- You will be told the battery capacity so you can compute this.
"""


def _build_user_prompt(operator_notes: List[str], battery_capacity_kwh: float) -> str:
    """Build the user message containing the operator notes to interpret."""
    notes_text = "\n".join(
        f"Note {i}: \"{note}\"" for i, note in enumerate(operator_notes)
    )
    return (
        f"Battery capacity is {battery_capacity_kwh} kWh.\n\n"
        f"Interpret the following {len(operator_notes)} operator note(s):\n\n"
        f"{notes_text}\n\n"
        f"Return a JSON array with exactly {len(operator_notes)} interpretation object(s), "
        f"one per note, in note_index order."
    )


def _parse_llm_response(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM's raw text response into a list of directive dicts.
    Handles potential markdown code fences around the JSON.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    result = json.loads(text)

    if not isinstance(result, list):
        raise ValueError("LLM output must be a JSON array")

    return result


async def interpret_notes(
    operator_notes: List[str],
    battery_capacity_kwh: float,
) -> List[Dict[str, Any]]:
    """
    Send operator notes to the Gemini LLM and return raw interpreted directives.

    Returns a list of dicts, each with keys:
      note_index, applies, directive_type, structured_adjustment, explanation

    Raises ValueError if the LLM output cannot be parsed.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free key from https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    full_system = SYSTEM_PROMPT + PERCENTAGE_PROMPT
    user_message = _build_user_prompt(operator_notes, battery_capacity_kwh)

    logger.info("Sending %d operator notes to Gemini for interpretation", len(operator_notes))

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_system,
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )

        raw_text = response.text
        if not raw_text:
            raise ValueError("LLM returned empty response")

        logger.info("Raw LLM response: %s", raw_text[:500])

        directives = _parse_llm_response(raw_text)
        return directives

    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}") from e
    except Exception as e:
        if "GEMINI_API_KEY" in str(e) or "API key" in str(e):
            raise
        logger.error("LLM interpretation failed: %s", e, exc_info=True)
        raise ValueError(f"LLM interpretation failed: {e}") from e
