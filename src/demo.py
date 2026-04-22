"""
src/demo.py

Real-time ASL recognition demo using ONNX model + MediaPipe HandLandmarker.
Shows live webcam feed with hand skeleton overlay and predicted word.

Usage:
    python src/demo.py --model models/sign_model.onnx --vocab 2000
"""

import argparse
import json
import collections
import time
import numpy as np
from pathlib import Path

from src.config import KEYPOINT_DIM, WINDOW_FRAMES, SMOOTH_WINDOW
from src.decode import greedy_decode_sequence
from src.keypoints import get_hand_detector, extract_keypoints_from_frame
from src.augmentations import normalize_keypoints


def load_vocab(vocab_path: str = "data/processed/vocab.json") -> dict:
    """Returns {label_idx: sign_word} mapping."""
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(frame, result):
    """Draw hand skeleton overlays on the frame using HandLandmarker result."""
    import cv2

    hand_colors = [(121, 22, 76), (245, 117, 66)]  # Left=purple, Right=orange
    h, w = frame.shape[:2]

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        color      = hand_colors[0] if handedness == "Left" else hand_colors[1]

        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 1)
        for pt in pts:
            cv2.circle(frame, pt, 3, color, -1)

    return frame


def run_demo(onnx_path: str, vocab_size: int = 2000, conf_threshold: float = 0.1, camera_idx: int = 1):
    try:
        import cv2
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

    idx_to_word = load_vocab()
    sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    detector    = get_hand_detector()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Make sure a camera is connected.")

    frame_buffer   = collections.deque(maxlen=WINDOW_FRAMES)
    hand_flags     = collections.deque(maxlen=WINDOW_FRAMES)  # 1 if hands detected, else 0
    pred_history   = collections.deque(maxlen=SMOOTH_WINDOW)
    current_word   = "Waiting..."
    current_conf   = 0.0
    fps_tracker    = collections.deque(maxlen=30)
    last_inference = time.time()
    no_hand_frames = 0
    RESET_AFTER    = 20  # frames without hands before clearing display

    print("Demo running — press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        rgb             = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts, mp_result = extract_keypoints_from_frame(rgb, detector)
        hands_visible   = len(mp_result.hand_landmarks) > 0
        frame_buffer.append(kpts)
        hand_flags.append(1 if hands_visible else 0)

        if hands_visible:
            no_hand_frames = 0
        else:
            no_hand_frames += 1
            if no_hand_frames >= RESET_AFTER:
                pred_history.clear()
                current_word = "..."
                current_conf = 0.0

        # Only infer if at least 50% of the buffer frames had hands
        hand_ratio = sum(hand_flags) / len(hand_flags) if hand_flags else 0
        if len(frame_buffer) == WINDOW_FRAMES and hand_ratio >= 0.5 and (time.time() - last_inference) > 0.5:
            seq_arr    = normalize_keypoints(np.stack(list(frame_buffer), axis=0).astype(np.float32))
            seq        = seq_arr[np.newaxis]
            msk        = np.zeros((1, WINDOW_FRAMES), dtype=bool)
            output_name = sess.get_outputs()[0].name  # "logits" (CE) or "log_probs" (CTC)
            output      = sess.run([output_name], {"keypoints": seq, "padding_mask": msk})[0]
            if output_name == "logits":
                # CE model: output is (B, C) — threshold on softmax confidence
                probs    = np.exp(output[0]) / np.exp(output[0]).sum()
                conf     = float(probs.max())
                pred_idx = int(np.argmax(probs))
                current_conf = conf
                if conf >= conf_threshold:
                    pred_history.append(pred_idx)
            else:
                # CTC model: output is (T, B, C) — greedy decode
                log_probs = output[:, 0, :]
                decoded   = greedy_decode_sequence(log_probs, blank=vocab_size)
                if decoded:
                    pred_history.append(decoded[0])
            if pred_history:
                most_common  = collections.Counter(pred_history).most_common(1)[0][0]
                current_word = idx_to_word.get(most_common, "Unknown")
            last_inference = time.time()

        frame = draw_skeleton(frame, mp_result)

        fps_tracker.append(1.0 / max(time.perf_counter() - t_start, 1e-6))
        fps = np.mean(fps_tracker)

        h, w = frame.shape[:2]

        # ── Centered prediction label ────────────────────────────────────────
        word_text  = current_word.upper()
        word_scale = 3.0
        word_thick = 5
        font       = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(word_text, font, word_scale, word_thick)

        # Place word at 68% down the frame
        word_x = (w - tw) // 2
        word_y = int(h * 0.68) + th

        # Semi-transparent background behind word
        pad = 18
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (word_x - pad, word_y - th - pad),
                      (word_x + tw + pad, word_y + baseline + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, word_text,
                    (word_x, word_y), font,
                    word_scale, (255, 255, 255), word_thick, cv2.LINE_AA)

        # ── Confidence bar ───────────────────────────────────────────────────
        bar_w      = tw + pad * 2          # same width as word background
        bar_h      = 14
        bar_x      = word_x - pad
        bar_y      = word_y + baseline + pad + 8
        fill_w     = int(bar_w * min(current_conf, 1.0))

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        # Filled portion — green → yellow → red based on confidence
        if current_conf >= conf_threshold:
            bar_color = (50, 220, 50)
        else:
            bar_color = (50, 100, 220)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          bar_color, -1)

        pct_text = f"{current_conf * 100:.0f}%"
        (pw, _), _ = cv2.getTextSize(pct_text, font, 0.55, 1)
        cv2.putText(frame, pct_text,
                    (bar_x + bar_w + 8, bar_y + bar_h - 2), font,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # ── Corner debug info ────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 120, 30), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{WINDOW_FRAMES}",
                    (20, 30), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"min conf: {conf_threshold:.2f}",
                    (20, 55), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("ASL Real-Time Demo — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str,   default="models/sign_model.onnx")
    parser.add_argument("--vocab",     type=int,   default=2000)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--camera",    type=int,   default=1)
    args = parser.parse_args()
    run_demo(args.model, args.vocab, args.threshold, args.camera)
