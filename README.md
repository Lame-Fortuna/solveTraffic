# Traffic Predictor App

Self-contained FastAPI app for serving the traffic impact classifier and duration range model.

## Run

From `D:/traffic`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

From `D:/traffic/app`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Model Files

The app expects the bundle at:

```text
app/modeler/traffic_cascade_artifacts/traffic_cascade_bundle.pkl
```

The app only performs inference. It does not train models during web requests.
