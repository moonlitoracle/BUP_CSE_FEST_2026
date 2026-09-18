# GridWise — LLM-Assisted Energy Optimization API

**BUP CSE Fest 2026 · Hackathon · Online Preliminary**

An HTTP API service that interprets natural-language operator notes using a Gemini LLM, validates the interpretations through deterministic guardrails, and produces an optimal 24-hour campus energy schedule using PuLP linear programming.

---

## Architecture

```
Operator Notes → [Gemini LLM Interpreter] → [Deterministic Guardrails] → [PuLP Optimizer] → [Post-Optimization Validator] → Response
```

| Component | File | Role |
|---|---|---|
| API Layer | `main.py` | FastAPI endpoints, pipeline orchestration |
| Data Models | `models.py` | Pydantic request/response schemas |
| LLM Interpreter | `interpreter.py` | Gemini-based operator note interpretation |
| Guardrails | `guardrails.py` | Deterministic validation of LLM outputs |
| Optimizer | `optimizer.py` | PuLP linear programming solver |
| Validator | `validator.py` | Post-optimization schedule replay & totals |

---

## Required Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Google Gemini API key. Get one free at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `PORT` | No | Server port (default: `8000`) |

---

## Local Setup

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone / navigate to the project directory
cd BUP_CSE_FEST_2026_Participant_Docs

# Install dependencies
pip install -r requirements.txt

# Set up your API key (create a .env file or export directly)
echo GEMINI_API_KEY=your-api-key-here > .env

# Run the server
python main.py
```

The server starts at `http://localhost:8000`.

---

## Docker Setup

```bash
# Build the image
docker build -t gridwise-api .

# Run the container (pass API key at runtime, never bake it in)
docker run -p 8000:8000 -e GEMINI_API_KEY=your-api-key-here gridwise-api
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | ≥0.115.0 | Web framework for HTTP API |
| `uvicorn[standard]` | ≥0.30.0 | ASGI server to run FastAPI |
| `pydantic` | ≥2.0.0 | Data validation and schemas |
| `pulp` | ≥2.8.0 | Linear programming solver |
| `google-genai` | ≥1.0.0 | Google Gemini LLM client |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |

---

## Model / Provider Declaration

- **LLM Provider**: Google Gemini (via `google-genai` SDK)
- **Model**: `gemini-2.5-flash`
- **Role**: Interprets 1–3 natural-language operator notes into structured directive objects (the sole LLM-dependent step in the pipeline)
- **Solver**: PuLP with CBC (Coin-or Branch and Cut) solver for deterministic 24-hour energy scheduling

---

## API Endpoints

### `GET /health`

Returns service readiness status.

```bash
curl http://localhost:8000/health
```

**Response** (HTTP 200):
```json
{"status": "ok"}
```

### `POST /optimize-energy`

Accepts a scenario and returns directive interpretations + optimized 24-hour plan.

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "TEST-01",
    "operator_notes": [
      "Solar output will drop to about 20% from 1 PM to 3 PM.",
      "The cafeteria menu changes tomorrow."
    ],
    "hours": [
      {"hour": 0, "demand_kwh": 100, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 1, "demand_kwh": 95, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 2, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 4},
      {"hour": 3, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 4},
      {"hour": 4, "demand_kwh": 95, "solar_kwh": 0, "tariff_bdt_per_kwh": 4},
      {"hour": 5, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 6, "demand_kwh": 120, "solar_kwh": 5, "tariff_bdt_per_kwh": 7},
      {"hour": 7, "demand_kwh": 135, "solar_kwh": 20, "tariff_bdt_per_kwh": 9},
      {"hour": 8, "demand_kwh": 145, "solar_kwh": 50, "tariff_bdt_per_kwh": 11},
      {"hour": 9, "demand_kwh": 155, "solar_kwh": 80, "tariff_bdt_per_kwh": 13},
      {"hour": 10, "demand_kwh": 165, "solar_kwh": 120, "tariff_bdt_per_kwh": 15},
      {"hour": 11, "demand_kwh": 175, "solar_kwh": 150, "tariff_bdt_per_kwh": 16},
      {"hour": 12, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
      {"hour": 13, "demand_kwh": 175, "solar_kwh": 150, "tariff_bdt_per_kwh": 15},
      {"hour": 14, "demand_kwh": 165, "solar_kwh": 120, "tariff_bdt_per_kwh": 14},
      {"hour": 15, "demand_kwh": 160, "solar_kwh": 80, "tariff_bdt_per_kwh": 15},
      {"hour": 16, "demand_kwh": 170, "solar_kwh": 40, "tariff_bdt_per_kwh": 19},
      {"hour": 17, "demand_kwh": 190, "solar_kwh": 10, "tariff_bdt_per_kwh": 24},
      {"hour": 18, "demand_kwh": 210, "solar_kwh": 0, "tariff_bdt_per_kwh": 30},
      {"hour": 19, "demand_kwh": 220, "solar_kwh": 0, "tariff_bdt_per_kwh": 32},
      {"hour": 20, "demand_kwh": 210, "solar_kwh": 0, "tariff_bdt_per_kwh": 28},
      {"hour": 21, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 20},
      {"hour": 22, "demand_kwh": 145, "solar_kwh": 0, "tariff_bdt_per_kwh": 11},
      {"hour": 23, "demand_kwh": 115, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
    ],
    "battery": {
      "capacity_kwh": 200,
      "initial_energy_kwh": 100,
      "minimum_energy_kwh": 30,
      "max_charge_kwh_per_hour": 50,
      "max_discharge_kwh_per_hour": 50
    }
  }'
```

---

## LLM Role in the Pipeline

The Gemini LLM is used **exclusively** for interpreting natural-language operator notes into structured directive objects. It is **not** used for optimization, scheduling, or cosmetic text generation. The pipeline is:

1. **LLM Interpretation** — Gemini reads operator notes and produces structured JSON directives
2. **Deterministic Guardrails** — Code validates every LLM output (types, bounds, hours, applies semantics)
3. **PuLP Optimization** — Mathematical solver produces the schedule (no LLM involvement)
4. **Post-Optimization Replay** — Independent verification that the schedule meets all constraints

---

## Known Limitations

- Requires a valid `GEMINI_API_KEY` for operator note interpretation
- The service does not support grid export (excess solar is curtailed)
- Hidden notes with highly ambiguous wording may require prompt tuning
