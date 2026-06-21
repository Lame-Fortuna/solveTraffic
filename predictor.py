from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
MODELER_DIR = APP_DIR / "modeler"
ARTIFACT_DIR = MODELER_DIR / "traffic_cascade_artifacts"
DEFAULT_BUNDLE_PATH = ARTIFACT_DIR / "traffic_cascade_bundle.pkl"
HIDDEN_UI_VALUES = {"__missing__", "NULL", "None", "nan", "", "test_demo"}


@dataclass
class ModelStatus:
    loaded: bool
    bundle_path: str
    error: str | None = None


class TrafficPredictor:
    def __init__(self, bundle_path: Path = DEFAULT_BUNDLE_PATH) -> None:
        self.bundle_path = Path(bundle_path)
        self.bundle: dict[str, Any] | None = None
        self.saver: Any | None = None
        self.status = ModelStatus(False, str(self.bundle_path), "Model not loaded yet")

    def load(self) -> ModelStatus:
        try:
            if not self.bundle_path.exists():
                raise FileNotFoundError(
                    f"Model bundle missing at {self.bundle_path}. Run training first or copy artifacts into app/modeler/traffic_cascade_artifacts."
                )
            sys.path.insert(0, str(MODELER_DIR)) if str(MODELER_DIR) not in sys.path else None
            self.saver = importlib.import_module("saver")
            self.bundle = self.saver.load_cascade_bundle(self.bundle_path)
            self.status = ModelStatus(True, str(self.bundle_path), None)
        except Exception as exc:
            self.bundle = None
            self.saver = None
            self.status = ModelStatus(False, str(self.bundle_path), str(exc))
        return self.status

    def ensure_loaded(self) -> None:
        if not self.status.loaded or self.bundle is None or self.saver is None:
            self.load()
        if not self.status.loaded or self.bundle is None or self.saver is None:
            raise RuntimeError(self.status.error or "Model bundle is unavailable")

    def predict(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        assert self.bundle is not None
        assert self.saver is not None

        prediction = self.saver.predict_traffic_impact_and_duration(raw_event, self.bundle)
        return {
            **prediction,
            "historical_info": self.historical_info(raw_event),
            "recommendations": self.recommendations(prediction, raw_event),
        }

    def historical_info(self, raw_event: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        assert self.bundle is not None

        corridor = str(raw_event.get("corridor") or "__missing__")
        lookup = self.bundle.get("historical_lookup", {})
        by_key = lookup.get("by_key", {})
        global_stats = lookup.get("global", {})
        corridor_lookup = by_key.get("corridor", {})

        closure_risk = corridor_lookup.get("closure_risk", {}).get(
            corridor, global_stats.get("closure_risk", 0.0)
        )
        impact_mean = corridor_lookup.get("impact_mean", {}).get(
            corridor, global_stats.get("impact_mean", 0.0)
        )
        frequency = corridor_lookup.get("freq", {}).get(corridor, 0)

        duration_summary = self.bundle.get("duration_training_summary", {})
        avg_duration = duration_summary.get("target_mean_minutes")
        median_duration = duration_summary.get("target_median_minutes")

        return {
            "corridor": corridor if corridor != "__missing__" else "Unknown",
            "corridor_event_count": int(frequency or 0),
            "corridor_closure_frequency_percent": round(float(closure_risk or 0.0) * 100, 1),
            "historical_impact_mean": round(float(impact_mean or 0.0), 2),
            "historical_average_duration_minutes": round(float(avg_duration), 1) if avg_duration is not None else None,
            "historical_median_duration_minutes": round(float(median_duration), 1) if median_duration is not None else None,
            "duration_scope": "global training history",
        }

    def category_options(self) -> dict[str, list[str]]:
        fixed = {
            "event_type": ["unplanned", "planned"],
            "priority": ["Low", "Medium", "High"],
            "event_cause": [
                "vehicle_breakdown",
                "accident",
                "water_logging",
                "tree_fall",
                "pot_holes",
                "debris",
                "vip_movement",
                "public_event",
                "procession",
                "protest",
                "construction",
                "congestion",
                "road_conditions",
                "others",
            ],
            "direction": ["north", "south", "east", "west", "north_east", "north_west", "south_east", "south_west"],
            "veh_type": ["heavy_vehicle", "truck", "lcv", "bmtc_bus", "private_bus", "private_car", "taxi", "auto", "others"],
        }
        if not self.bundle:
            return {**fixed, "corridor": [], "police_station": [], "zone": [], "junction": [], "cargo_material": [], "reason_breakdown": []}

        maps = self.bundle.get("category_maps", {})
        options = dict(fixed)
        for key in ["event_cause", "event_type", "corridor", "police_station", "zone", "junction", "direction", "veh_type", "cargo_material", "reason_breakdown"]:
            deduped: dict[str, str] = {}
            for value in maps.get(key, {}).keys():
                text = str(value).strip()
                if text in HIDDEN_UI_VALUES:
                    continue
                normalized = text.lower()
                deduped.setdefault(normalized, text)
            values = list(deduped.values())
            if values:
                options[key] = sorted(values, key=str.lower)
        options.setdefault("priority", fixed["priority"])
        return options

    def recommendations(self, prediction: dict[str, Any], raw_event: dict[str, Any]) -> dict[str, Any]:
        label = int(prediction.get("impact_label", 0))
        probs = prediction.get("impact_probabilities", {})
        prob_2 = float(probs.get("2", 0.0))
        duration = prediction.get("duration_range", {})
        upper = duration.get("upper_minutes")
        lower = duration.get("lower_minutes", 0)
        closure_requested = str(raw_event.get("requires_road_closure", "")).lower() in {"true", "1", "yes", "on"}

        long_event = upper is None or (upper and upper >= 240) or lower >= 240
        medium_event = (upper and upper >= 120) or lower >= 60
        critical = label == 2 or prob_2 >= 0.5 or closure_requested

        if critical or long_event:
            personnel = "High"
            barricades = "Full" if critical else "Partial"
            primary_action = "Diversion recommended"
            notes = [
                "Set up advance warning points near the incident corridor.",
                "Keep a supervisor looped in until duration bucket drops.",
            ]
        elif label == 1 or medium_event:
            personnel = "Moderate"
            barricades = "Partial"
            primary_action = "Temporary barricades"
            notes = [
                "Use channelization only around the conflict point.",
                "Monitor for escalation if queue length increases.",
            ]
        else:
            personnel = "Low"
            barricades = "None"
            primary_action = "Routine patrol"
            notes = [
                "No diversion needed unless field conditions worsen.",
                "A patrol check and status update should be enough.",
            ]

        return {
            "personnel": personnel,
            "barricades": barricades,
            "primary_action": primary_action,
            "notes": notes,
        }


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}
