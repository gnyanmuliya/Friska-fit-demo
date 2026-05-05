# Friska AI Fitness Demo

Streamlit-based fitness planning app that can generate weekly workout plans from:

- A structured intake form.
- Expert/doctor notes (text or PDF).
- SOAP notes (service and test flow present in codebase).

## What Is In This Project

- `app.py`: Main Streamlit entry point and top menu.
- `ui/`: Frontend views and shared plan rendering.
- `services/`: Feature-level orchestration services.
- `core/`: Planning engines, parsers, legacy-compatible fitness logic, models.
- `dataset/`: CSV datasets used for exercise selection and video enrichment.

## Main Product Flows

### 1) Fitness Plan Generator Flow (Form-based)

Used from menu option: `Fitness Plan Generator`.

1. User fills intake form in `ui/workout_generator_view.py`.
2. UI builds a normalized `profile` dict (age, goals, days, equipment, restrictions, etc.).
3. `services/workout_service.py` calls `core/fitness_engine.py -> FitnessEngine.generate_plan`.
4. Engine path:
- Normalizes profile values.
- Loads fitness dataset.
- Applies filters (`ExerciseFilter.apply_filters`).
- Applies safety/quality filters (hypertension guardrail, user avoid-list, equipment boosting).
- Runs legacy planner (`run_old_engine`) with deterministic seed.
- Removes avoided exercises from generated output.
- Ensures enough strength work when goal requests strength/muscle.
- Enriches each exercise with video/thumbnail metadata via `core/video_mapper.py`.
5. UI renders plan + JSON download via `ui/shared_components.py`.

### 2) Wellness Expert Note Based Flow (AI + Fallback)

Used from menu option: `Wellness Experts Note based Plan`.

1. User either:
- Pastes note text, or
- Uploads PDF (text extracted with `pdfplumber`), or
- Loads sample PDFs from `Experts note samples/`.
2. `services/experts_note_service.py -> ExpertsNoteService.generate_plan_from_notes`.
3. Parser stage:
- Primary: `AzureAIPrescriptionParser.parse_notes` (`core/azure_ai_parser.py`).
- Fallback: local rule parser when Azure blocks content (`AzureAIContentFilterError`).
4. Service ensures weekly schedule is usable (fills/expands sparse schedules from note frequency hints).
5. Profile + clinical context are derived from parsed output.
6. Dataset is filtered and safety-pruned:
- `ExerciseFilter.apply_filters`
- `_hard_medical_exclusion`
- explicit avoided exercise removal
7. Day-by-day plan is generated from parsed weekly schedule and session types.
8. Result returned as:
- `plan`
- `profile`
- `ai_prescription`
- `clinical_context`
- `used_local_fallback`
9. UI shows plan and the applied profile.

### 3) SOAP Note Flow (Implemented in Service, Not Wired in Main Menu)

Core flow exists and can be used via service/tests:

1. `core/soap_engine.py -> SoapEngine.parse_text`:
- Extracts SOAP sections/vitals using `ClinicalExtractionTool`.
- Extracts restrictions, exercise mentions, and frequency.
- Builds inferred profile (days, focus, duration, equipment, restrictions).
2. `generate_plan_from_text` builds plan by passing inferred profile to `generate_plan_local`.
3. Additional SOAP restriction filtering is applied to resulting exercises.
4. Titles/categories are normalized for readable output.

Notes:
- `ui/soap_testing_view.py` exists but is not currently connected in `app.py`.
- `SoapEngine.USE_PARSER_TOOL = False` by default; fallback path is local planning.

## Setup

### Prerequisites

- Python 3.10+ recommended.
- Windows PowerShell commands shown below.

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment Variables

Create `.env` in project root:

```env
AZURE_AI_KEY=your_azure_openai_key
```

Important:
- `ExpertsNoteService` initializes Azure parser in `__init__`, so this key is required before opening the Expert Note flow.
- Do not commit real keys.

## Run App

```powershell
streamlit run app.py
```

Main menu routes:
- `Fitness Plan Generator`
- `Wellness Experts Note based Plan`

## Datasets Used

- `dataset/fitness.csv`: Primary exercise dataset for plan generation.
- `dataset/Exercise videos.csv`: Video links and media metadata mapped by `GuidId`.
- `dataset/soap_data.csv`: SOAP-specific dataset used by SOAP flow helper functions.

Expected dataset content includes exercise name, category, body region, equipment, sets/reps, RPE, safety cues, tags, and medical suitability fields.

## Output Plan Structure

Typical output is a JSON object keyed by day:

- `Monday`, `Tuesday`, ...
  - `main_workout_category`
  - `warmup`: list of exercise dicts
  - `main_workout`: list of exercise dicts
  - `cooldown`: list of exercise dicts

Exercise entries commonly include:
- `name`, `sets`, `reps`, `rest`, `intensity_rpe`
- `benefit`, `safety_cue`, `steps`
- `equipment`, `body_region`, `primary_category`
- `video_url`, `thumbnail_url` (when available)

## Safety And Guardrails

Built-in protections include:

- Medical suitability filtering (`is_not_suitable_for`, limitations, conditions).
- Restriction-aware exclusion (knee-sensitive, high-impact, floor-work, overhead/spinal constraints).
- Hypertension-specific filtering in engine wrapper.
- User-entered explicit avoidance filtering.

These rules reduce risk but do not replace clinical supervision.

## Useful Test/Debug Scripts

- `test_complete_exercises.py`: Smoke-style experts-note plan generation with mocked parser.
- `tmp_validate_soap.py`: Quick SOAP parse/build sanity script.
- `SOAPnotetest.py`: Legacy SOAP demo script.

Run example:

```powershell
python test_complete_exercises.py
python tmp_validate_soap.py
```

## Known Implementation Notes

- Main app currently exposes only two UI modes; SOAP view is present but not wired in the menu.
- `services/experts_note_service.py` is large and contains legacy/duplicate helper sections; behavior follows the latest method definitions in the class.
- Azure endpoint/model names are currently hardcoded in parser modules.

## Suggested Next Cleanup (Optional)

1. Add SOAP mode into `app.py` menu if that workflow should be user-facing.
2. Split `experts_note_service.py` into parser, planner, and safety modules for maintainability.
3. Add formal tests for:
- weekly schedule expansion
- clinical restriction enforcement
- avoid-list filtering

