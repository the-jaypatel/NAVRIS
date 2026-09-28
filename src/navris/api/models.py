"""
Pydantic response models for the NAVRIS REST API.

Matches the TBA frontend contract. Any field not measured or computed
in the NAVRIS research data is explicitly typed as Optional and returned
as null to preserve scientific integrity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Origin(BaseModel):
    lat: float
    lon: float
    alt: float


class Frame(BaseModel):
    """
    Single coherent navigation state snapshot matching TBA Frame contract.
    All unavailable/unmeasured values are explicitly None/null.
    """
    t: float
    state: str  # 'init' | 'fused' | 'degraded' | 'lost' | 'ins' | 'ai' | 'refusion' | 'stabilized'

    # Ground truth (VBOX reference trajectory)
    trueE: float
    trueN: float
    trueSpeed: float
    trueHeading: float

    # Navigation solution (NAVRIS Config B: ESKF + ZUPT + NHC)
    solE: float
    solN: float
    solSpeed: float
    solHeading: float
    alt: float

    # Errors
    errE: float
    errN: float
    horizErr: float
    sigmaE: Optional[float] = None
    sigmaN: Optional[float] = None
    sigmaH: Optional[float] = None
    confidence: Optional[float] = None
    velErr: float
    hdgErr: float

    # GNSS status (smartphone receiver)
    gnssAvail: bool
    gnssE: float
    gnssN: float
    lastFixT: float
    fixType: str
    satsUsed: int
    satsVisible: int
    hdop: Optional[float] = None
    cn0: Optional[float] = None

    # IMU status (smartphone sensor)
    accelMag: float
    gyroZ: float
    roll: float = 0.0
    pitch: float = 0.0
    accelBiasMg: Optional[float] = None
    gyroBiasDph: Optional[float] = None
    imuTemp: Optional[float] = None

    # ESKF filter status
    nis: Optional[float] = None
    accepted: int = 0
    rejected: int = 0
    inertialOnly: float = 0.0
    sigmaV: Optional[float] = None
    sigmaPsi: Optional[float] = None

    # AI status (unverified/untrained AI model is not fabricated)
    aiStage: str = "idle"
    aiPredictedErr: Optional[float] = None
    aiCorrections: int = 0
    aiInferenceMs: Optional[float] = None


class HistoryRow(BaseModel):
    """1 Hz time-series sample for analytics charts."""
    t: float
    state: str
    errE: float
    errN: float
    horizErr: float
    sigE: Optional[float] = None
    sigN: Optional[float] = None
    sigH: Optional[float] = None
    conf: Optional[float] = None
    velErr: float
    hdgErr: float
    nis: Optional[float] = None
    rej: int = 0
    gnss: int = 1
    accel: float
    gyro: float


class TimelineEvent(BaseModel):
    t: float
    kind: str  # 'state' | 'action' | 'note'
    text: str


class RecordingSummary(BaseModel):
    id: str
    dataset: str = "IO-VNBD"
    status: str
    samples: int
    duration_s: float
    description: str


class RecordingsResponse(BaseModel):
    recordings: List[str]
    details: List[RecordingSummary]


class SessionResponse(BaseModel):
    recording: str
    dataset: str = "IO-VNBD"
    mode: str = "RESEARCH / REPLAY"
    solution: str = "ESKF + ZUPT + NHC"
    configuration: str = "B"
    sample_rate_hz: float = 10.0
    samples: int
    duration_s: float
    reference_source: str = "VBOX CAN Ground Truth"
    origin: Origin
    available_fields: List[str]
    unavailable_fields: List[str]


class HistoryResponse(BaseModel):
    recording: str
    step: int
    count: int
    samples: List[HistoryRow]


class EventsResponse(BaseModel):
    recording: str
    events: List[TimelineEvent]


class ControlRequest(BaseModel):
    recording: Optional[str] = None
    action: Optional[str] = None  # 'play' | 'pause' | 'reset' | 'seek'
    speed: Optional[float] = None
    time: Optional[float] = None


class ControlResponse(BaseModel):
    ok: bool = True
    recording: str
    action: Optional[str] = None
    index: int = 0
    time_s: float = 0.0
