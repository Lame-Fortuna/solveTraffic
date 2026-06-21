from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from .predictor import TrafficPredictor, parse_bool
except ImportError:
    from predictor import TrafficPredictor, parse_bool


app = FastAPI(title="Traffic Impact and Duration Predictor")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

predictor = TrafficPredictor()


@app.on_event("startup")
def load_model() -> None:
    predictor.load()


def impact_label_name(label: int) -> str:
    return {0: "Low", 1: "Moderate", 2: "Critical"}.get(int(label), "Unknown")


def build_event_payload(
    latitude: float,
    longitude: float,
    start_datetime: str,
    event_type: str,
    event_cause: str,
    priority: str,
    requires_road_closure: bool,
    corridor: str,
    police_station: str,
    zone: str,
    junction: str,
    direction: str,
    veh_type: str,
    cargo_material: str,
    reason_breakdown: str,
    age_of_truck: str,
) -> dict[str, Any]:
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_datetime": start_datetime,
        "event_type": event_type or "unplanned",
        "event_cause": event_cause or "unknown",
        "priority": priority or "Low",
        "requires_road_closure": bool(requires_road_closure),
        "corridor": corridor or "Unknown",
        "police_station": police_station or "Unknown",
        "zone": zone or "Unknown",
        "junction": junction or "Unknown",
        "direction": direction or "Unknown",
        "veh_type": veh_type or None,
        "cargo_material": cargo_material or None,
        "reason_breakdown": reason_breakdown or None,
        "age_of_truck": float(age_of_truck) if str(age_of_truck).strip() else 0.0,
    }


def enrich_result(result: dict[str, Any]) -> dict[str, Any]:
    probs = result.get("impact_probabilities", {})
    result["impact_label_name"] = impact_label_name(result.get("impact_label", 0))
    result["impact_percentages"] = {
        key: round(float(value) * 100, 1) for key, value in probs.items()
    }
    return result


@app.get("/", response_class=HTMLResponse)
def form_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "model_status": predictor.status,
            "options": predictor.category_options(),
            "default_datetime": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    latitude: float = Form(...),
    longitude: float = Form(...),
    start_datetime: str = Form(...),
    event_type: str = Form("unplanned"),
    event_cause: str = Form("unknown"),
    priority: str = Form("Low"),
    requires_road_closure: str | None = Form(None),
    corridor: str = Form("Unknown"),
    police_station: str = Form("Unknown"),
    zone: str = Form("Unknown"),
    junction: str = Form("Unknown"),
    direction: str = Form("Unknown"),
    veh_type: str = Form(""),
    cargo_material: str = Form(""),
    reason_breakdown: str = Form(""),
    age_of_truck: str = Form(""),
) -> HTMLResponse:
    event = build_event_payload(
        latitude,
        longitude,
        start_datetime,
        event_type,
        event_cause,
        priority,
        parse_bool(requires_road_closure),
        corridor,
        police_station,
        zone,
        junction,
        direction,
        veh_type,
        cargo_material,
        reason_breakdown,
        age_of_truck,
    )
    try:
        result = enrich_result(predictor.predict(event))
    except Exception as exc:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": str(exc), "event": event, "model_status": predictor.status},
            status_code=503,
        )
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "result": result, "event": event, "model_status": predictor.status},
    )


@app.post("/api/predict")
async def predict_api(request: Request) -> JSONResponse:
    payload = await request.json()
    try:
        result = enrich_result(predictor.predict(payload))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(result)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": predictor.status.loaded,
        "bundle_path": predictor.status.bundle_path,
        "error": predictor.status.error,
    }
