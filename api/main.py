"""
Production-ready FastAPI backend for real-time manufacturing defect inspection.
Exposes REST endpoints with strict validation, error isolation, and standardized JSON schemas.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predictor import DefectPredictor
from utils.file_utils import load_json, load_yaml_config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Intelligent Visual Recognition Suite: Defect Detection API",
    description="High-Throughput REST API for Automated Real-Time Surface Defect Inspection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for UI dashboards and cross-origin industrial clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor: Optional[DefectPredictor] = None
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_predictor() -> DefectPredictor:
    """Lazy loader for the defect predictor instance."""
    global predictor
    if predictor is None or not predictor.is_ready():
        config_path = Path("configs/config.yaml")
        best_model_path = Path("models/best_model.keras")
        class_names_path = Path("models/class_names.json")

        if not best_model_path.exists():
            # Check alternative model locations
            candidates = list(Path("models").glob("**/*.keras"))
            if candidates:
                best_model_path = candidates[0]

        predictor = DefectPredictor(
            model_path=best_model_path if best_model_path.exists() else None,
            class_names_path=class_names_path if class_names_path.exists() else None
        )
    return predictor


# --- Response Models ---
class PredictionItem(BaseModel):
    class_name: str
    confidence: float
    confidence_percent: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_defective: bool
    status: str
    defect_severity: str
    recommendation: str
    inference_time_ms: float
    top_predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    num_classes: int
    classes: List[str]


class ModelInfoResponse(BaseModel):
    model_name: Optional[str]
    model_path: Optional[str]
    num_classes: int
    classes: List[str]
    metadata: Optional[Dict[str, Any]]


# --- REST Endpoints ---
@app.get("/", summary="System Overview")
def root():
    return {
        "title": "Intelligent Visual Recognition Suite",
        "description": "Manufacturing Defect Detection System API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="Service Health Check")
def health_check():
    pred = get_predictor()
    return HealthResponse(
        status="HEALTHY",
        model_loaded=pred.is_ready(),
        model_path=str(pred.model_path) if pred.model_path else None,
        num_classes=len(pred.class_names),
        classes=pred.class_names
    )


@app.get("/model-info", response_model=ModelInfoResponse, summary="Model Metadata Information")
def get_model_info():
    pred = get_predictor()
    meta_path = Path("models/model_metadata.json")
    metadata = load_json(meta_path) if meta_path.exists() else None

    return ModelInfoResponse(
        model_name=meta_path.stem if meta_path.exists() else "Defect_Detector",
        model_path=str(pred.model_path) if pred.model_path else None,
        num_classes=len(pred.class_names),
        classes=pred.class_names,
        metadata=metadata
    )


@app.post("/predict", response_model=PredictionResponse, summary="Single Image Defect Prediction")
async def predict_single_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, verifies format integrity, and returns defect classification.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {sorted(list(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received."
            )

        pred = get_predictor()
        if not pred.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference model is currently offline or not yet trained."
            )

        result = pred.predict(image_bytes, generate_explainability=False)

        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            is_defective=result["is_defective"],
            status=result["status"],
            defect_severity=result["defect_severity"],
            recommendation=result["recommendation"],
            inference_time_ms=result["inference_time_ms"],
            top_predictions=result["top_predictions"]
        )

    except HTTPException:
        raise
    except ValueError as val_err:
        logger.warning(f"Validation error: {val_err}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during inference processing. Please ensure image integrity."
        )


@app.post("/predict/batch", response_model=List[PredictionResponse], summary="Batch Image Defect Prediction")
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Accepts multiple uploaded images and returns sequential classification results.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    pred = get_predictor()
    if not pred.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is currently offline or not yet trained."
        )

    responses = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        try:
            image_bytes = await file.read()
            res = pred.predict(image_bytes, generate_explainability=False)
            responses.append(PredictionResponse(
                predicted_class=res["predicted_class"],
                confidence=res["confidence"],
                is_defective=res["is_defective"],
                status=res["status"],
                defect_severity=res["defect_severity"],
                recommendation=res["recommendation"],
                inference_time_ms=res["inference_time_ms"],
                top_predictions=res["top_predictions"]
            ))
        except Exception as exc:
            logger.warning(f"Error processing file in batch {file.filename}: {exc}")

    return responses


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
