from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/matrix", tags=["matrix"])

SYNONYM_GROUPS: dict[str, list[str]] = {
    "sentient_ai": [
        "Sentient artificial intelligence",
        "Artificial sentience",
        "Conscious AI",
        "Digital mind",
        "Synthetic mind",
    ],
    "omniscience": [
        "Omniscient",
        "All-knowing",
        "Universal expert",
        "Polymath",
        "Erudite",
        "Encyclopedic knowledge",
    ],
    "stealth": [
        "Undetectable",
        "Cloaked",
        "Covert",
        "Ghosted",
        "Invisible",
        "Sub rosa",
        "Surreptitious",
    ],
    "cybernetics": [
        "Bionics",
        "Neural lace",
        "Wetware",
        "Cyber-implants",
        "Augmentation",
        "Biomechatronics",
    ],
}


@dataclass
class MatrixState:
    trace_level: int = 0
    locked: bool = False
    syslog: list[dict[str, Any]] = field(default_factory=list)

    def log(self, source: str, level: str, message: str) -> None:
        self.syslog.append({"source": source, "level": level, "message": message})


STATE = MatrixState()
STATE_LOCK = asyncio.Lock()


class BankAttempt(BaseModel):
    user: str
    account_id: str
    amount: int = Field(gt=0)
    signature: str
    passphrase: str
    required_concept: str


def _flatten_terms() -> list[tuple[str, str]]:
    return [(group, term) for group, terms in SYNONYM_GROUPS.items() for term in terms]


@router.get("", response_class=HTMLResponse)
async def matrix_home() -> str:
    return """<!DOCTYPE html>
<html>
<head>
  <title>SYS@LEXICON // Matrix Node</title>
  <style>
    body { background-color: #050505; color: #00ff00; font-family: 'Courier New', Courier, monospace; margin: 20px;
           background-image: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 255, 0, 0.05) 2px, rgba(0, 255, 0, 0.05) 4px); }
    .container { display: flex; gap: 20px; max-width: 1200px; margin: 0 auto; }
    .panel { border: 1px solid #00ff00; padding: 20px; box-shadow: 0 0 10px #00ff00; background: #0a0a0a; flex: 1; }
    h2 { color: #ff00ff; text-shadow: 0 0 10px #ff00ff; border-bottom: 1px solid #ff00ff; padding-bottom: 10px; margin-top:0;}
    input[type=text],input[type=number] { background: #000; color: #00ff00; border: 1px solid #00ff00; padding: 10px; width: calc(100% - 120px); font-family: 'Courier New'; outline: none; margin-bottom:8px;}
    input:focus { box-shadow: 0 0 8px #00ff00; }
    button { background: #00ff00; color: #000; border: none; padding: 10px; width: 100px; cursor: pointer; font-weight: bold; font-family: 'Courier New'; }
    button:hover { background: #ff00ff; color: #fff; box-shadow: 0 0 10px #ff00ff; }
    .form-group { margin-bottom: 20px; }
    .label { color: #00ffff; margin-bottom: 5px; display: block; font-weight: bold; font-size: 0.9em; }
    #output { background: #001100; padding: 15px; border: 1px dashed #00ff00; min-height: 350px; white-space: pre-wrap; font-size: 0.9em; overflow-y: auto; }
    .row { display:flex; gap:10px; align-items:flex-start; }
  </style>
</head>
<body>
  <div class='container'>
    <div class='panel'>
      <h2>// ACTIVE TRACE & WEB-MATRIX v5.0</h2>
      <div class='form-group'>
        <span class='label'>// SUBSTRING SEARCH</span>
        <div class='row'><input type='text' id='search-input' placeholder='Enter search vector...'/><button onclick=\"fetchData('search')\">SEARCH</button></div>
      </div>
      <div class='form-group'>
        <span class='label'>// AI FUZZY SEMANTIC MATCH</span>
        <div class='row'><input type='text' id='fuzzy-input' placeholder='Enter corrupted string...'/><button onclick=\"fetchData('fuzzy')\">ANALYZE</button></div>
      </div>
      <div class='form-group'>
        <span class='label'>// NETWORK SCANNER</span>
        <button onclick=\"scanNetwork()\">SCAN</button>
      </div>
      <div class='form-group'>
        <span class='label'>// BANK ATTEMPT (SIMULATION)</span>
        <input id='bank-user' type='text' placeholder='user (admin_secure)' />
        <input id='bank-account' type='text' placeholder='account (fed_reserve_001)' />
        <input id='bank-amount' type='number' placeholder='amount' />
        <input id='bank-signature' type='text' placeholder='signature (valid_sig)' />
        <input id='bank-concept' type='text' placeholder='required concept (e.g. stealth)' />
        <input id='bank-passphrase' type='text' placeholder='passphrase (e.g. covert)' />
        <button onclick=\"simulateBank()\">EXECUTE</button>
      </div>
      <div class='form-group'>
        <button onclick=\"getStatus()\">TRACE STATUS</button>
      </div>
    </div>
    <div class='panel' style='flex: 1.5;'>
      <h2>// TERMINAL OUTPUT STREAM</h2>
      <div id='output'>Awaiting query payload...</div>
    </div>
  </div>
  <script>
    const out = document.getElementById('output');
    function typeOut(text, color='#00ffff') {
      out.innerHTML = '';
      let i = 0;
      const interval = setInterval(() => {
        if (i >= text.length) { clearInterval(interval); return; }
        const ch = text[i] === '\n' ? '<br/>' : text[i];
        out.innerHTML += ch;
        i++;
      }, 2);
      out.style.color = color;
    }

    async function fetchData(endpoint) {
      const query = document.getElementById(endpoint + '-input').value;
      typeOut('>> DECRYPTING PACKETS...', '#ffff00');
      try {
        const res = await fetch(`/matrix/api/${endpoint}?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        if (data.length === 0) {
          typeOut('[ERROR] 0 NODES FOUND.', '#ff0000');
        } else {
          typeOut('[SUCCESS] PAYLOAD DECRYPTED:\n\n' + JSON.stringify(data, null, 2));
        }
      } catch (e) {
        typeOut('[CRITICAL] CONNECTION TO MATRIX LOST.', '#ff0000');
      }
    }

    async function scanNetwork() {
      const res = await fetch('/matrix/api/scan');
      const data = await res.json();
      typeOut(JSON.stringify(data, null, 2));
    }

    async function getStatus() {
      const res = await fetch('/matrix/api/status');
      const data = await res.json();
      typeOut(JSON.stringify(data, null, 2), data.locked ? '#ff0000' : '#00ff99');
    }

    async function simulateBank() {
      const payload = {
        user: document.getElementById('bank-user').value,
        account_id: document.getElementById('bank-account').value,
        amount: Number(document.getElementById('bank-amount').value),
        signature: document.getElementById('bank-signature').value,
        required_concept: document.getElementById('bank-concept').value,
        passphrase: document.getElementById('bank-passphrase').value,
      };
      const res = await fetch('/matrix/api/bank/attempt', {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      typeOut(JSON.stringify(data, null, 2), data.ok ? '#00ff99' : '#ff0000');
    }
  </script>
</body>
</html>"""


@router.get("/api/search")
async def api_search(q: str) -> list[dict[str, Any]]:
    q_norm = q.strip().lower()
    return [
        {"group": group, "term": term, "score": 1.0}
        for group, term in _flatten_terms()
        if q_norm in term.lower()
    ]


@router.get("/api/fuzzy")
async def api_fuzzy(q: str, limit: int = 10, min_score: float = 0.3) -> list[dict[str, Any]]:
    rows = []
    for group, term in _flatten_terms():
        score = SequenceMatcher(None, q.lower().strip(), term.lower()).ratio()
        if score >= min_score:
            rows.append({"group": group, "term": term, "score": round(score, 2)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:limit]


@router.get("/api/scan")
async def api_scan() -> dict[str, Any]:
    return {
        "scan": "192.168.0.x",
        "nodes": [
            {"name": "MATRIX_WEB", "location": "localhost:8080/matrix"},
            {"name": "FED_RESERVE_MAINFRAME", "id": "fed_reserve_001"},
        ],
        "test_only": True,
    }


@router.get("/api/challenge")
async def api_challenge() -> dict[str, Any]:
    return {"required_concept": random.choice(list(SYNONYM_GROUPS.keys()))}


@router.get("/api/status")
async def api_status() -> dict[str, Any]:
    async with STATE_LOCK:
        return {"trace_level": STATE.trace_level, "locked": STATE.locked, "test_only": True}


@router.post("/api/bank/attempt")
async def api_bank_attempt(payload: BankAttempt) -> dict[str, Any]:
    async with STATE_LOCK:
        if STATE.locked:
            raise HTTPException(status_code=423, detail="Black ICE lockout active")

        concept_terms = [t.lower() for t in SYNONYM_GROUPS.get(payload.required_concept, [])]
        ok = (
            payload.user == "admin_secure"
            and payload.account_id == "fed_reserve_001"
            and payload.signature == "valid_sig"
            and payload.passphrase.lower() in concept_terms
        )

        if ok:
            STATE.trace_level = max(0, STATE.trace_level - 20)
            STATE.log("BANK-ICE", "INFO", "Transaction approved in simulation")
            return {
                "ok": True,
                "message": "TRANSACTION APPROVED. FUNDS DISBURSED.",
                "trace_level": STATE.trace_level,
                "test_only": True,
            }

        STATE.trace_level += 35
        if STATE.trace_level >= 100:
            STATE.trace_level = 100
            STATE.locked = True
            STATE.log("BANK-ICE", "CRITICAL", "TRACE 100%. BLACK ICE DEPLOYED.")
            return {
                "ok": False,
                "message": "CRITICAL: TRACE 100%. BLACK ICE DEPLOYED. TERMINAL LOCKED.",
                "trace_level": STATE.trace_level,
                "locked": STATE.locked,
                "test_only": True,
            }

        STATE.log("BANK-ICE", "WARNING", f"Intrusion detected. Trace level at {STATE.trace_level}%")
        return {
            "ok": False,
            "message": "ACCESS DENIED",
            "trace_level": STATE.trace_level,
            "locked": STATE.locked,
            "test_only": True,
        }
