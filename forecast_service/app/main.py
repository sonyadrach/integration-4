"""Entry point for the forecasting service (forecast-api)."""
import logging
import json
import sys
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI

from .schemas import PredictRequest, PredictResponse

# Used for local runs outside Docker. In the container, orchestrator
# environment variables are already present, and load_dotenv() does not
# override existing environment variables.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("forecast-api")

MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and stats from MLflow registry once at startup."""
    client = MlflowClient()
    try:
        logger.info(f"Resolving model alias '{MODEL_ALIAS}' for '{MODEL_NAME}'...")
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        
        logger.info(f"Loading model version {mv.version}...")
        app.state.model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        app.state.model_version = mv.version
        
        if mv.run_id:
            try:
                logger.info("Downloading data_stats.json from artifacts...")
                stats_path = mlflow.artifacts.download_artifacts(
                    run_id=mv.run_id, artifact_path="data_stats.json"
                )
                app.state.data_stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
            except Exception as stats_err:
                logger.warning(f"Could not load data_stats.json for run {mv.run_id}: {stats_err}")
                app.state.data_stats = {"detail": "data_stats.json not found for this model version"}
        else:
            logger.warning(f"Model version {mv.version} has no linked run_id; skipping data_stats.json.")
            app.state.data_stats = {"detail": "No run_id linked to this model version"}

        logger.info("Model loaded successfully.")
        
    except mlflow.exceptions.RestException as e:
        logger.error(
            f"CRITICAL ERROR: Failed to resolve alias '{MODEL_ALIAS}' "
            f"for model '{MODEL_NAME}'. Reason: {e}"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Unexpected error loading model: {e}")
        sys.exit(1)

    yield

    app.state.model = None
    app.state.data_stats = None


app = FastAPI(
    title="Forecast API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check endpoint used by the orchestrator."""
    return {
        "status": "ok",
        "model_version": app.state.model_version,
    }


@app.get("/model/data-stats")
def get_data_stats():
    """Return descriptive statistics loaded from MLflow artifacts"""
    return app.state.data_stats


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict a house sale price from the provided Ames Housing features."""
    t0 = time.perf_counter()

    df = pd.DataFrame([{
        "GrLivArea": int(req.gr_liv_area),
        "TotalBsmtSF": int(req.total_bsmt_sf),
        "GarageArea": int(req.garage_area),
        "YearBuilt": int(req.year_built),
        "OverallQual": int(req.overall_qual),
        "MSZoning": str(req.ms_zoning),
    }])

    price = float(app.state.model.predict(df)[0])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        "predict: %.2f USD (%.2f ms) [version: %s]",
        price,
        latency_ms,
        app.state.model_version
    )

    return PredictResponse(
        predicted_price=price,
        model_version=app.state.model_version,
        latency_ms=latency_ms,
    )