from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas, auth
from app.ml.predictor import predict as run_prediction

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=schemas.PredictResponse)
def predict_message(
    payload: schemas.PredictRequest,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(auth.get_current_user_optional),
):
    """
    Works for anyone — no login required. If the request DOES include a
    valid token, the result is also saved to that user's history. This is
    what "optional auth" looks like in practice: same endpoint, different
    behavior depending on whether the caller is identified.
    """
    result = run_prediction(payload.message)

    if current_user is not None:
        entry = models.CheckHistory(
            message=result["message"],
            is_spam=result["is_spam"],
            confidence=result["confidence"],
            owner_id=current_user.id,
        )
        db.add(entry)
        db.commit()

    return result


@router.get("/history", response_model=List[schemas.HistoryItem])
def get_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Requires login (strict dependency) — history is per-user."""
    return (
        db.query(models.CheckHistory)
        .filter(models.CheckHistory.owner_id == current_user.id)
        .order_by(models.CheckHistory.checked_at.desc())
        .all()
    )
