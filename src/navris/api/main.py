"""
NAVRIS REST API — Minimal bridge between NAVRIS research data and TBA frontend.

Endpoints follow the exact contract in Step 2:
- GET  /api/v1/nav/recordings
- GET  /api/v1/nav/session
- GET  /api/v1/nav/frame
- GET  /api/v1/nav/history
- GET  /api/v1/nav/events
- POST /api/v1/nav/reset
- POST /api/v1/nav/control
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from navris.api.models import (
    ControlRequest,
    ControlResponse,
    EventsResponse,
    Frame,
    HistoryResponse,
    Origin,
    RecordingsResponse,
    RecordingSummary,
    SessionResponse,
)
from navris.api.replay import ReplayEngine, ALL_RECORDINGS

logger = logging.getLogger("navris.api")

app = FastAPI(
    title="NAVRIS Research API",
    description="Serves verified IO-VNBD benchmark replay datasets to the TBA frontend.",
    version="0.1.0",
)

# CORS: Allow TBA development and preview servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

engine = ReplayEngine()


# ── Step 2 Endpoints ──────────────────────────────────────────────────

@app.get("/api/v1/nav/recordings", response_model=RecordingsResponse)
def list_recordings():
    """Returns all available IO-VNBD benchmark recordings."""
    details = []
    for r in ALL_RECORDINGS:
        if r in engine.available_recordings:
            try:
                rec_data = engine.get_recording(r)
                details.append(RecordingSummary(
                    id=r,
                    dataset="IO-VNBD",
                    status="SUCCESS",
                    samples=rec_data.n_frames,
                    duration_s=rec_data.duration_s,
                    description="Real IO-VNBD driving sequence",
                ))
            except Exception as e:
                details.append(RecordingSummary(
                    id=r,
                    dataset="IO-VNBD",
                    status="ERROR",
                    samples=0,
                    duration_s=0.0,
                    description=str(e),
                ))
        else:
            status = "UNOBSERVABLE" if r == "M" else "NOT_FOUND"
            details.append(RecordingSummary(
                id=r,
                dataset="IO-VNBD",
                status=status,
                samples=0,
                duration_s=0.0,
                description="GNSS unobservable in Gate 2.3B baseline" if r == "M" else "Not processed",
            ))

    return RecordingsResponse(
        recordings=engine.available_recordings,
        details=details,
    )


@app.get("/api/v1/nav/session", response_model=SessionResponse)
def get_session(recording: Optional[str] = Query(default=None)):
    """
    Returns session metadata for the requested recording:
    sample rate, sample count, duration, origin, and exact field availability.
    """
    rec_id = (recording or engine.current_recording).upper()
    try:
        rec_data = engine.get_recording(rec_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return SessionResponse(
        recording=rec_id,
        dataset="IO-VNBD",
        mode="RESEARCH / REPLAY",
        solution="ESKF + ZUPT + NHC",
        configuration="B",
        sample_rate_hz=10.0,
        samples=rec_data.n_frames,
        duration_s=rec_data.duration_s,
        reference_source="VBOX CAN Ground Truth",
        origin=Origin(
            lat=rec_data.origin_lat,
            lon=rec_data.origin_lon,
            alt=rec_data.origin_alt,
        ),
        available_fields=[
            "t", "state", "trueE", "trueN", "trueSpeed", "trueHeading",
            "solE", "solN", "solSpeed", "solHeading", "alt",
            "errE", "errN", "horizErr", "velErr", "hdgErr",
            "gnssAvail", "gnssE", "gnssN", "lastFixT", "fixType",
            "satsUsed", "satsVisible", "accelMag", "gyroZ"
        ],
        unavailable_fields=[
            "confidence", "sigmaE", "sigmaN", "sigmaH",
            "hdop", "cn0", "nis", "accelBiasMg", "gyroBiasDph", "imuTemp",
            "sigmaV", "sigmaPsi", "aiPredictedErr", "aiInferenceMs"
        ],
    )


@app.get("/api/v1/nav/frame", response_model=Frame)
def get_frame(
    recording: Optional[str] = Query(default=None),
    index: Optional[int] = Query(default=None, ge=0),
):
    """Returns one real frame from the benchmark dataset."""
    rec_id = (recording or engine.current_recording).upper()
    try:
        rec_data = engine.get_recording(rec_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = index if index is not None else engine.current_idx
    if idx >= rec_data.n_frames:
        raise HTTPException(status_code=400, detail=f"Index {idx} out of range [0, {rec_data.n_frames - 1}]")

    return rec_data.frame_at(idx)


@app.get("/api/v1/nav/history", response_model=HistoryResponse)
def get_history(
    recording: Optional[str] = Query(default=None),
    start: int = Query(default=0, ge=0),
    end: Optional[int] = Query(default=None, ge=0),
    step: int = Query(default=10, ge=1, le=100),
):
    """
    Returns replay history samples (default step=10 converts 10 Hz recording
    into 1 Hz samples for analytics charts).
    """
    rec_id = (recording or engine.current_recording).upper()
    try:
        rec_data = engine.get_recording(rec_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    samples = rec_data.get_history(start_idx=start, end_idx=end, step=step)
    return HistoryResponse(
        recording=rec_id,
        step=step,
        count=len(samples),
        samples=samples,
    )


@app.get("/api/v1/nav/events", response_model=EventsResponse)
def get_events(recording: Optional[str] = Query(default=None)):
    """
    Returns timeline events derived safely from genuine data.
    Does NOT invent fake AI or outage events.
    """
    rec_id = (recording or engine.current_recording).upper()
    try:
        rec_data = engine.get_recording(rec_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    events = [
        {"t": float(rec_data.times[0]), "kind": "action", "text": f"Loaded {rec_id} · IO-VNBD benchmark replay"},
        {"t": float(rec_data.times[0]), "kind": "note", "text": "Solution: ESKF + ZUPT + NHC (Config B)"},
    ]

    # Detect genuine GNSS dropout transitions from sync parquet
    if "phone_gps_is_valid" in rec_data.df.columns:
        valid_series = rec_data.df["phone_gps_is_valid"].values
        times = rec_data.df["time_s"].values
        prev = bool(valid_series[0])
        # Sample every 10 frames (1s) to find true transitions
        for i in range(10, rec_data.n_frames, 10):
            curr = bool(valid_series[i])
            if curr != prev:
                t_val = float(times[i])
                if not curr:
                    events.append({"t": t_val, "kind": "state", "text": f"GNSS signal lost at T+{t_val:.1f}s — INS dead reckoning"})
                else:
                    events.append({"t": t_val, "kind": "state", "text": f"GNSS signal reacquired at T+{t_val:.1f}s"})
                prev = curr
                if len(events) >= 30:
                    break

    return EventsResponse(
        recording=rec_id,
        events=events,
    )


@app.post("/api/v1/nav/reset")
def post_reset():
    """Reset replay position to index 0."""
    engine.current_idx = 0
    engine.is_playing = False
    return {"ok": True, "recording": engine.current_recording, "index": 0}


@app.post("/api/v1/nav/control", response_model=ControlResponse)
def post_control(req: ControlRequest):
    """Replay control: select recording, play, pause, seek."""
    if req.recording:
        try:
            rec_data = engine.select(req.recording)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        rec_data = engine.current_data()

    if req.action == "play":
        engine.is_playing = True
    elif req.action == "pause":
        engine.is_playing = False
    elif req.action == "reset":
        engine.current_idx = 0
        engine.is_playing = False
    elif req.action == "seek" and req.time is not None:
        idx = int(np.searchsorted(rec_data.times, req.time))
        engine.current_idx = min(idx, rec_data.n_frames - 1)

    if req.speed is not None:
        engine.speed = req.speed

    cur_time = float(rec_data.times[engine.current_idx]) if engine.current_idx < rec_data.n_frames else 0.0

    return ControlResponse(
        ok=True,
        recording=engine.current_recording,
        action=req.action,
        index=engine.current_idx,
        time_s=cur_time,
    )


@app.get("/api/v1/nav/health")
def health():
    return {
        "status": "ok",
        "current_recording": engine.current_recording,
        "available_recordings": engine.available_recordings,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("navris.api.main:app", host="0.0.0.0", port=8000, reload=True)
