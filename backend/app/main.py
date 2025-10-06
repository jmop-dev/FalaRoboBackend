# main.py — FalaRobo Backend (sensores + posições + válvulas 12/14)
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
CommandDir = Literal["Forwards", "Backwards", "None", "Invalid"]

class PressureIn(BaseModel):
    bar: float = Field(..., ge=0, le=10)

class StartIn(BaseModel):
    running: bool

class SensorsMsg(BaseModel):
    type: Literal["SENSORS"] = "SENSORS"
    small: dict        # { "1S1": bool, "1S2": bool }
    big: dict          # { "2S1": bool, "2S2": bool }
    positions: dict    # { "small": Position, "big": Position }
    valves: dict | None = None  # optional: { "small": {"12":bool,"14":bool}, "big":{"12":bool,"14":bool} }

class SensorsHTTPIn(BaseModel):
    """HTTP helper with the same shape used in WS SENSORS."""
    small: dict
    big: dict
    positions: dict
    valves: dict | None = None

# -----------------------------------------------------------------------------
# In-memory state
# -----------------------------------------------------------------------------
STATE: dict = {
    "running": True,
    "bar": 6.0,
    "sensors": {"1S1": False, "1S2": False, "2S1": False, "2S2": False},
    "positions": {"small": "Unknown", "big": "Unknown"},   # provided by frontend; backend armazena
    "valves": {  # válvulas 12/14 por eixo
        "small": {"12": False, "14": False},
        "big":   {"12": False, "14": False},
    },
    "validation": {
        "small": "ok",  # ok | mismatch | invalid_sensors
        "big": "ok",
        "errors": []
    },
}

active_ws: set[WebSocket] = set()

# -----------------------------------------------------------------------------
# Helpers: normalização, validação e derivação
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
    Regras:
      * ambos sensores true -> invalid_sensors
      * ambos false -> expected = Unknown; mismatch se provided != Unknown
      * um true -> expected do sensor; mismatch se provided != expected
    """
    errors: list[str] = []
    if s1 and s2:
        errors.append(f"{axis_name}: sensors conflict (both true)")
        return "invalid_sensors", errors

    expected: Position
    if not s1 and not s2:
        expected = "Unknown"
    else:
        expected = "Backwards" if s1 else "Forwards"

    if provided != expected:
        errors.append(f"{axis_name}: provided={provided} but sensors expect {expected}")
        return "mismatch", errors

    return "ok", errors

def _command_from_valves(v12: bool, v14: bool) -> CommandDir:
    # 12 => Backwards, 14 => Forwards, ambos false => None, ambos true => Invalid
    if v12 and v14:
        return "Invalid"
    if v12:
        return "Backwards"
    if v14:
        return "Forwards"
    return "None"

def _append_valve_errors(axis_name: str, v12: bool, v14: bool, errors: list[str]):
    if v12 and v14:
        errors.append(f"{axis_name}: valves conflict (12 and 14 both ON)")

def _recompute_validation():
    s = STATE["sensors"]
    p = STATE["positions"]
    # normaliza posições recebidas
    STATE["positions"]["small"] = _norm_position(p.get("small"))
    STATE["positions"]["big"]   = _norm_position(p.get("big"))

    small_status, small_errs = _validate_axis(bool(s["1S1"]), bool(s["1S2"]), STATE["positions"]["small"], "small")
    big_status,   big_errs   = _validate_axis(bool(s["2S1"]), bool(s["2S2"]), STATE["positions"]["big"],   "big")

    # checa conflito de válvulas (12/14) e adiciona erro descritivo se houver
    v_small = STATE["valves"]["small"]; _append_valve_errors("small", bool(v_small["12"]), bool(v_small["14"]), small_errs)
    v_big   = STATE["valves"]["big"];   _append_valve_errors("big",   bool(v_big["12"]),   bool(v_big["14"]),   big_errs)

    STATE["validation"] = {
        "small": small_status,
        "big": big_status,
        "errors": [*small_errs, *big_errs]
    }

def _commanded_payload() -> dict:
    vs = STATE["valves"]["small"]
    vb = STATE["valves"]["big"]
    return {
        "small": _command_from_valves(bool(vs["12"]), bool(vs["14"])),  # "Backwards"|"Forwards"|"None"|"Invalid"
        "big":   _command_from_valves(bool(vb["12"]), bool(vb["14"]))
    }

def _snapshot_payload() -> dict:
    _recompute_validation()
    return {
        "running": STATE["running"],
        "bar": STATE["bar"],
        "positions": dict(STATE["positions"]),
        "sensors": dict(STATE["sensors"]),
        "valves": {
            "small": dict(STATE["valves"]["small"]),
            "big":   dict(STATE["valves"]["big"]),
        },
        "commanded": _commanded_payload(),  # direção pedida pelas válvulas 12/14
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

# -----------------------------------------------------------------------------
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

# HTTP helper para enviar sensores/posições/válvulas via Postman
@app.post("/sensors")
async def post_sensors(body: SensorsHTTPIn):
    # sensores
    for k in ("1S1", "1S2"):
        if k in body.small:
            STATE["sensors"][k] = bool(body.small[k])
    for k in ("2S1", "2S2"):
        if k in body.big:
            STATE["sensors"][k] = bool(body.big[k])
    # posições (front define, back armazena)
    STATE["positions"]["small"] = _norm_position(body.positions.get("small"))
    STATE["positions"]["big"]   = _norm_position(body.positions.get("big"))
    # válvulas (opcional)
    if body.valves:
        v = body.valves
        if "small" in v:
            STATE["valves"]["small"]["12"] = bool(v["small"].get("12", STATE["valves"]["small"]["12"]))
            STATE["valves"]["small"]["14"] = bool(v["small"].get("14", STATE["valves"]["small"]["14"]))
        if "big" in v:
            STATE["valves"]["big"]["12"] = bool(v["big"].get("12", STATE["valves"]["big"]["12"]))
            STATE["valves"]["big"]["14"] = bool(v["big"].get("14", STATE["valves"]["big"]["14"]))
    await _push_state()
    return {"ok": True}

# -----------------------------------------------------------------------------
# AI Snapshot — inclui movimentos (posições) e comandos (válvulas)
# -----------------------------------------------------------------------------
@app.get("/ai/snapshot")
async def ai_snapshot():
    snap = _snapshot_payload()
    movements = {
        "small": snap["positions"]["small"],  # estado atual
        "big":   snap["positions"]["big"]
    }
    commands = dict(snap["commanded"])        # direção pedida pelas válvulas 12/14
    return {
        "bar": snap["bar"],
        "running": snap["running"],
        "positions": dict(snap["positions"]),
        "movements": movements,
        "commands": commands,                  # << novo campo
        "valves": dict(snap["valves"]),        # útil p/ debug/IA
        "1S1": snap["sensors"].get("1S1", False),
        "1S2": snap["sensors"].get("1S2", False),
        "2S1": snap["sensors"].get("2S1", False),
        "2S2": snap["sensors"].get("2S2", False)
    }

# -----------------------------------------------------------------------------
# WebSocket (front envia sensores + posições + válvulas; backend armazena e valida)
# -----------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, x_user: str | None = None):
    await ws.accept()
    active_ws.add(ws)
    try:
        # snapshot inicial
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
                # sensores
                small = msg.get("small") or {}
                big   = msg.get("big") or {}
                for k in ("1S1", "1S2"):
                    if k in small:
                        STATE["sensors"][k] = bool(small[k])
                for k in ("2S1", "2S2"):
                    if k in big:
                        STATE["sensors"][k] = bool(big[k])

                # posições (frontend → backend)
                pos = msg.get("positions") or {}
                STATE["positions"]["small"] = _norm_position(pos.get("small"))
                STATE["positions"]["big"]   = _norm_position(pos.get("big"))

                # válvulas (opcional no WS)
                if "valves" in msg and isinstance(msg["valves"], dict):
                    v = msg["valves"]
                    if "small" in v:
                        STATE["valves"]["small"]["12"] = bool(v["small"].get("12", STATE["valves"]["small"]["12"]))
                        STATE["valves"]["small"]["14"] = bool(v["small"].get("14", STATE["valves"]["small"]["14"]))
                    if "big" in v:
                        STATE["valves"]["big"]["12"] = bool(v["big"].get("12", STATE["valves"]["big"]["12"]))
                        STATE["valves"]["big"]["14"] = bool(v["big"].get("14", STATE["valves"]["big"]["14"]))

                # broadcast + ACK
                await _push_state()
                await ws.send_text(json.dumps({"type": "ACK", "ok": True}))
                continue

            # unknown
            await ws.send_text(json.dumps({"type": "ACK", "ok": False, "reason": "unknown_type"}))

    except WebSocketDisconnect:
        pass
    finally:
        active_ws.discard(ws)
