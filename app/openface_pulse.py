# app/openface_pulse.py
import os, sys, time, csv, subprocess, tempfile, pathlib, uuid
from typing import Dict, Tuple
from dataclasses import dataclass
import cv2
from dotenv import load_dotenv
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]   # project root (MoodWatch)
load_dotenv(ROOT / ".env")                           # load .env from root

OPENFACE_BIN = os.getenv("OPENFACE_BIN")
OUT_DIR = pathlib.Path(os.getenv("OF_OUT_DIR", str(ROOT / "processed")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG = True  # flip to False when you’re done

if not OPENFACE_BIN:
    raise EnvironmentError("OPENFACE_BIN not set. Put the full path to FeatureExtraction in your .env.")

@dataclass
class CaptureSpec:
    fps: int
    size: Tuple[int, int]
    fourcc: str = "mp4v"

class OpenFacePulse:
    """
    Record frames into a temp video during a pulse, then run OpenFace and summarize.
    Usage:
        pulse = OpenFacePulse(CaptureSpec(...))
        pulse.start()
        pulse.write(frame)   # per frame in your ON window
        summary, csv_path = pulse.finish()
    """
    def __init__(self, spec: CaptureSpec, session_id: str = None):
        self.spec = spec
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._tmpdir = tempfile.TemporaryDirectory()
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self._video_path = pathlib.Path(self._tmpdir.name) / f"pulse_{ts}.mp4"
        self._writer = None
        self._frames = 0
        self._t_start = None
        self._t_end = None
    # ... (rest unchanged)

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*self.spec.fourcc)
        self._writer = cv2.VideoWriter(str(self._video_path), fourcc, self.spec.fps, self.spec.size)
        if not self._writer.isOpened():
            raise RuntimeError("VideoWriter failed to open (try fourcc='XVID' and .avi)")
        self._t_start = time.time()
        if DEBUG:
            print(f"[OF] start  session={self.session_id} video={self._video_path.name} "
                f"{self.spec.size[0]}x{self.spec.size[1]}@{self.spec.fps} fourcc={self.spec.fourcc}", flush=True)

    def write(self, frame_bgr):
        if self._writer is None:
            raise RuntimeError("Call start() before write()")
        self._writer.write(frame_bgr)
        self._frames += 1
        if DEBUG and self._frames % 30 == 0:
            print(f"[OF] wrote {self._frames} frames…", flush=True)

    def finish(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        self._t_end = time.time()

        pulse_out = OUT_DIR
        pulse_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            OPENFACE_BIN, "-f", str(self._video_path),
            "-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp",
            "-out_dir", str(pulse_out),
            "-no_vis"  # keep it headless
        ]
        if DEBUG:
            print("[OF] run   ", " ".join(cmd), flush=True)

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            if DEBUG:
                print(proc.stdout)
                print(proc.stderr)
            raise RuntimeError("OpenFace failed (see logs above)")

        csvs = sorted(pulse_out.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {pulse_out}")
        csv_path = csvs[0]
        if DEBUG:
            print(f"[OF] done   out_csv={csv_path.name}", flush=True)

        summary = self._summarize_csv(csv_path)
        label, score = self._classify_expression(summary)
        summary["expr"] = label
        summary["expr_score"] = score

        if DEBUG:
            print(
                f"[OF] summary frames={summary.get('frames')} dur_s={summary.get('dur_s'):.3f} "
                f"AU12={summary.get('AU12_r'):.3f} AU04={summary.get('AU04_r'):.3f} AU26={summary.get('AU26_r'):.3f}",
                flush=True
            )
        self._append_session_row(summary, csv_path)
        self._tmpdir.cleanup()
        return summary, csv_path


    def _summarize_csv(self, csv_path: pathlib.Path) -> Dict[str, float]:
        import csv, statistics as stats

        # AUs we’ll use for rules (OpenFace CSV usually has these columns)
        aus = [
            "AU01_r","AU02_r","AU04_r","AU06_r","AU07_r","AU09_r","AU10_r",
            "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_c"
        ]
        pose = ["pose_Rx","pose_Ry","pose_Rz"]

        series = {k: [] for k in aus + pose}
        rows = 0
        with open(csv_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows += 1
                for k in series:
                    v = row.get(k, "")
                    if v != "": series[k].append(float(v))

        if rows == 0:
            return {}

        out = {}
        for k, xs in series.items():
            out[k] = sum(xs)/len(xs) if xs else 0.0  # simple mean per pulse

        # Friendly proxies
        out["avg_smile"]   = out["AU12_r"]
        out["avg_furrow"]  = out["AU04_r"]
        out["avg_mouthop"] = out["AU26_r"]
        out["valence_proxy"] = out["AU12_r"] - out["AU04_r"]                    # ↑happy, ↓furrow
        out["arousal_proxy"] = out["AU25_r"] + out["AU26_r"] + out["AU45_c"]    # mouth/breath/blink

        out["frames"] = rows
        out["dur_s"]  = max(0.001, (self._t_end - self._t_start)) if self._t_start else 0.0
        return out
    
    def _classify_expression(self, s: Dict[str, float]) -> Tuple[str, float]:
        """
        Return (label, score 0..1). Rules are intentionally simple & adjustable.
        """
        # thresholds you can tune
        T = {
            "smile": 0.30, "duchenne": 0.15, "furrow": 0.25,
            "surprise_eyes": 0.30, "mouth_open": 0.25,
            "disgust": 0.25, "anger_tense": 0.25, "sad_mouth": 0.20
        }

        AU = lambda k: s.get(k, 0.0)
        candidates = []

        # Happy (Duchenne smile)
        score_happy = max(0.0, AU("AU12_r") - 0.5*AU("AU04_r"))
        if AU("AU12_r") > T["smile"] and AU("AU06_r") > T["duchenne"]:
            score_happy += 0.2
        candidates.append(("happy", min(1.0, score_happy)))

        # Sad (inner brow + down mouth, not much smile)
        score_sad = max(0.0, 0.5*(AU("AU01_r")+AU("AU04_r")) + AU("AU15_r") - 0.5*AU("AU12_r"))
        candidates.append(("sad", min(1.0, score_sad)))

        # Anger (brow lower + lid tighten, little smile)
        score_anger = max(0.0, AU("AU04_r") + 0.5*AU("AU07_r") - 0.5*AU("AU12_r"))
        if AU("AU23_r") > T["anger_tense"]:
            score_anger += 0.1
        candidates.append(("anger", min(1.0, score_anger)))

        # Surprise (raised brows + mouth open)
        score_surprise = max(0.0, 0.5*(AU("AU01_r")+AU("AU02_r")) + AU("AU26_r"))
        candidates.append(("surprise", min(1.0, score_surprise)))

        # Disgust (nose wrinkle + upper lip raise, low smile)
        score_disgust = max(0.0, AU("AU09_r")+AU("AU10_r") - 0.3*AU("AU12_r"))
        candidates.append(("disgust", min(1.0, score_disgust)))

        # Neutral: fallback when none are strong
        score_neutral = max(0.0, 1.0 - (AU("AU12_r")+AU("AU04_r")+AU("AU26_r")))
        candidates.append(("neutral", min(1.0, score_neutral)))

        label, score = max(candidates, key=lambda x: x[1])
        return label, float(score)


    def _append_session_row(self, summary: Dict[str, float], csv_path: pathlib.Path):
        session_csv = OUT_DIR / "session_summary.csv"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = ["ts","session_id","dur_s","frames",
                  "AU01_r","AU02_r","AU04_r","AU06_r","AU12_r","AU15_r","AU20_r","AU25_r","AU26_r","AU45_c",
                  "pose_Rx","pose_Ry","pose_Rz",
                  "avg_smile","avg_furrow","avg_mouthop","blink_presence_mean","expr","expr_score",
                  "src_csv"]
        first = not session_csv.exists()
        row = {
            "ts": ts,
            "session_id": self.session_id,
            "dur_s": round(summary.get("dur_s", 0.0), 3),
            "frames": int(summary.get("frames", 0)),
            "src_csv": str(csv_path)
        }
        for k in header:
            if k in summary:
                row[k] = round(summary[k], 6) if isinstance(summary[k], float) else summary[k]
        with open(session_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if first: w.writeheader()
            w.writerow(row)