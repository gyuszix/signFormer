"""
src/server.py

FastAPI WebSocket server for real-time ASL inference from browser.

Browser sends keypoint frames (126 floats) via WebSocket.
Server accumulates a sliding window, runs ONNX inference, streams predictions back.
Keypoints never leave the device as video — only 126 floats/frame sent.

Run:
    uvicorn src.server:app --host 0.0.0.0 --port 8000
    # then open http://localhost:8000
"""

import json
import numpy as np
from collections import deque, Counter
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.config import WINDOW_FRAMES, SMOOTH_WINDOW, MAX_SEQ_LEN
from src.augmentations import normalize_keypoints

app = FastAPI()


# ── Serve frontend ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Frontend not found at frontend/index.html</h1>", status_code=404)


# ── Model — loaded once at startup ─────────────────────────────────────────────

_sess       = None
_idx2word   = {}
_vocab_size = 0


_output_name = "logits"   # "logits" for CE models, "log_probs" for CTC


@app.on_event("startup")
def load_model():
    global _sess, _idx2word, _vocab_size, _output_name
    try:
        import onnxruntime as ort
        onnx_path = Path("models/sign_model.onnx")
        if not onnx_path.exists():
            print("⚠ models/sign_model.onnx not found — export first:")
            print("    python3 -m src.export --checkpoint <ckpt> --vocab 300")
            return
        _sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # Detect CE vs CTC from ONNX output name
        _output_name = _sess.get_outputs()[0].name   # "logits" or "log_probs"

        with open("data/processed/vocab.json") as f:
            word2idx = json.load(f)
        _idx2word   = {v: k for k, v in word2idx.items()}
        _vocab_size = len(_idx2word)
        print(f"Model loaded: {_vocab_size} classes, output='{_output_name}'")
    except Exception as e:
        print(f"Model load failed: {e}")


# ── Greedy CTC decode ──────────────────────────────────────────────────────────

def _greedy_decode(log_probs: np.ndarray, blank: int) -> list[int]:
    """Collapse repeats, strip blank tokens. Returns list of label indices."""
    best = np.argmax(log_probs, axis=-1)
    prev, decoded = -1, []
    for b in best:
        if b != prev:
            if b != blank:
                decoded.append(int(b))
            prev = b
    return decoded


# ── WebSocket inference endpoint ───────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if _sess is None:
        await ws.send_json({"error": "Model not loaded. See server logs."})
        await ws.close()
        return

    blank        = _vocab_size
    buffer       = deque(maxlen=MAX_SEQ_LEN)
    pred_history = deque(maxlen=SMOOTH_WINDOW)
    frame_count  = 0
    INFER_EVERY  = 6   # ~200ms at 30fps

    try:
        while True:
            data = await ws.receive_json()

            if data.get("type") != "frame":
                continue

            kpts = np.array(data["keypoints"], dtype=np.float32)  # (126,)
            buffer.append(kpts)
            frame_count += 1

            fill = len(buffer) / WINDOW_FRAMES

            # Not enough frames yet — send progress only
            if len(buffer) < WINDOW_FRAMES:
                await ws.send_json({"buffer_fill": round(fill, 2), "prediction": None})
                continue

            # Run inference every INFER_EVERY frames
            if frame_count % INFER_EVERY != 0:
                continue

            seq  = normalize_keypoints(np.stack(list(buffer), axis=0))  # (T, 126)
            inp  = seq[np.newaxis].astype(np.float32)                    # (1, T, 126)
            mask = np.zeros((1, seq.shape[0]), dtype=bool)               # no padding

            output = _sess.run([_output_name], {
                "keypoints":    inp,
                "padding_mask": mask,
            })[0]

            # CE: output is (1, C) — argmax directly
            # CTC: output is (T, 1, C+1) — greedy decode
            if _output_name == "logits":
                pred_idx = int(np.argmax(output[0]))
                pred_history.append(pred_idx)
            else:
                decoded = _greedy_decode(output[:, 0, :], blank)
                if not decoded:
                    continue
                pred_history.append(decoded[0])

            best_idx   = Counter(pred_history).most_common(1)[0][0]
            confidence = pred_history.count(best_idx) / len(pred_history)
            await ws.send_json({
                "prediction":  _idx2word.get(best_idx, "?"),
                "confidence":  round(float(confidence), 2),
                "buffer_fill": 1.0,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
