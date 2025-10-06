# Run:
#   pip install fastapi "uvicorn[standard]" pydantic
#   uvicorn main:app --reload --port 8000
from __future__ import annotations
import os, json
from typing import Literal, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "FalaRobo API")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
Position = Literal["Forwards", "Backwards", "Unknown", "Invalid"]

class PressureIn(BaseModel):
    bar: float = Field(..., ge=0, le=10)

class StartIn(BaseModel):
    running: bool

class SensorsMsg(BaseModel):
    type: Literal["SENSORS"] = "SENSORS"
    small: dict        # { "1S1": bool, "1S2": bool }
    big: dict          # { "2S1": bool, "2S2": bool }
    positions: dict    # { "small": Position, "big": Position }

class SensorsHTTPIn(BaseModel):
    """Same shape as WS SENSORS but for HTTP testing/dev."""
    small: dict        # { "1S1": bool, "1S2": bool }
    big: dict          # { "2S1": bool, "2S2": bool }
    positions: dict    # { "small": Position, "big": Position }

# -----------------------------------------------------------------------------
# In-memory state
# -----------------------------------------------------------------------------
STATE: dict = {
    "running": True,
    "bar": 6.0,
    "sensors": {"1S1": False, "1S2": False, "2S1": False, "2S2": False},
    "positions": {"small": "Unknown", "big": "Unknown"},   # authoritative (last stored)
    "validation": {
        "small": "ok",  # ok | mismatch | invalid_sensors
        "big": "ok",
        "errors": []
    },
}

active_ws: set[WebSocket] = set()

# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def _norm_position(p: Optional[str]) -> Position:
    p = (p or "Unknown").capitalize()
    if p not in ("Forwards", "Backwards", "Unknown", "Invalid"):
        return "Unknown"
    return p  # type: ignore[return-value]

def _validate_axis(s1: bool, s2: bool, provided: Position, axis_name: str) -> tuple[str, list[str]]:
    """
    Returns (status, errors):
      - status: "ok" | "mismatch" | "invalid_sensors"
    Rules:
      * conflict (both true) -> invalid_sensors
      * both false -> sensors say Unknown; mismatch if provided != Unknown
      * one true  -> expected from sensor; mismatch if provided != expected
    """
    errors: list[str] = []
    if s1 and s2:
        errors.append(f"{axis_name}: sensors conflict (both true)")
        return "invalid_sensors", errors

    # deduce expectation from sensors
    if not s1 and not s2:
        expected: Position = "Unknown"
    else:
        expected = "Backwards" if s1 else "Forwards"

    if provided != expected:
        errors.append(f"{axis_name}: provided={provided} but sensors expect {expected}")
        return "mismatch", errors

    return "ok", errors

def _recompute_validation():
    s = STATE["sensors"]
    p = STATE["positions"]
    small_status, small_errs = _validate_axis(bool(s["1S1"]), bool(s["1S2"]), _norm_position(p.get("small")), "small")
    big_status,   big_errs   = _validate_axis(bool(s["2S1"]), bool(s["2S2"]), _norm_position(p.get("big")),   "big")
    STATE["positions"]["small"] = _norm_position(p.get("small"))
    STATE["positions"]["big"]   = _norm_position(p.get("big"))
    STATE["validation"] = {
        "small": small_status,
        "big": big_status,
        "errors": [*small_errs, *big_errs]
    }

def _snapshot_payload() -> dict:
    # always compute validation fresh before sending
    _recompute_validation()
    return {
        "running": STATE["running"],
        "bar": STATE["bar"],
        "positions": dict(STATE["positions"]),
        "sensors": dict(STATE["sensors"]),
        "validation": {
            "small": STATE["validation"]["small"],
            "big": STATE["validation"]["big"],
            "errors": list(STATE["validation"]["errors"]),
        },
    }

async def _ws_broadcast(payload: dict):
    data = json.dumps(payload)
    dead = []
    for ws in list(active_ws):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_ws.discard(ws)

async def _push_state():
    await _ws_broadcast({"type": "STATE", "payload": _snapshot_payload()})

# REST
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/state")
async def get_state(x_user: str | None = Header(default=None)):
    return _snapshot_payload()

@app.post("/control/pressure")
async def set_pressure(body: PressureIn, x_user: str | None = Header(default=None)):
    STATE["bar"] = max(0.0, min(10.0, float(body.bar)))
    await _push_state()
    return {"ok": True, "bar": STATE["bar"]}

@app.post("/control/start")
async def set_start(body: StartIn, x_user: str | None = Header(default=None)):
    STATE["running"] = bool(body.running)
    await _push_state()
    return {"ok": True, "running": STATE["running"]}

@app.get("/ai/snapshot")
async def ai_snapshot():
    snap = _snapshot_payload()

    # movimentos individuais
    movements = {
        "small": snap["positions"]["small"],
        "big": snap["positions"]["big"]
    }

    return {
        "bar": snap["bar"],
        "running": snap["running"],
        "positions": dict(snap["positions"]),
        "movements": movements,
        "1S1": snap["sensors"].get("1S1", False),
        "1S2": snap["sensors"].get("1S2", False),
        "2S1": snap["sensors"].get("2S1", False),
        "2S2": snap["sensors"].get("2S2", False)
    }


# HTTP endpoint to update sensors+positions
@app.post("/sensors")
async def post_sensors(body: SensorsHTTPIn):
    # sensors
    for k in ("1S1", "1S2"):
        if k in body.small:
            STATE["sensors"][k] = bool(body.small[k])
    for k in ("2S1", "2S2"):
        if k in body.big:
            STATE["sensors"][k] = bool(body.big[k])
    # positions (frontend-provided)
    STATE["positions"]["small"] = _norm_position(body.positions.get("small"))
    STATE["positions"]["big"]   = _norm_position(body.positions.get("big"))
    await _push_state()
    return {"ok": True}

# -----------------------------------------------------------------------------
# WebSocket
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, x_user: str | None = None):
    await ws.accept()
    active_ws.add(ws)
    try:
        # initial snapshot
        await ws.send_text(json.dumps({"type": "STATE", "payload": _snapshot_payload()}))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type": "ACK", "ok": False, "reason": "invalid_json"}))
                continue

            mtype = msg.get("type")
            if mtype == "PING":
                await ws.send_text(json.dumps({"type": "PONG", "ts": msg.get("ts")}))
                continue

            if mtype == "SENSORS":
                # parse/store sensors
                small = msg.get("small") or {}
                big   = msg.get("big") or {}
                for k in ("1S1", "1S2"):
                    if k in small:
                        STATE["sensors"][k] = bool(small[k])
                for k in ("2S1", "2S2"):
                    if k in big:
                        STATE["sensors"][k] = bool(big[k])

                # parse/store positions
                pos = msg.get("positions") or {}
                STATE["positions"]["small"] = _norm_position(pos.get("small"))
                STATE["positions"]["big"]   = _norm_position(pos.get("big"))

                # broadcast snapshot + ACK
                await _push_state()
                await ws.send_text(json.dumps({"type": "ACK", "ok": True}))
                continue

            # unknown
            await ws.send_text(json.dumps({"type": "ACK", "ok": False, "reason": "unknown_type"}))

    except WebSocketDisconnect:
        pass
    finally:
        active_ws.discard(ws)
