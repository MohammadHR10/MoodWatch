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

    def write(self, frame_bgr):
        if self._writer is None:
            raise RuntimeError("Call start() before write()")
        self._writer.write(frame_bgr)
        self._frames += 1

    def finish(self) -> Tuple[Dict[str, float], pathlib.Path]:
        if self._writer:
            self._writer.release()
            self._writer = None
        self._t_end = time.time()

        pulse_out = OUT_DIR  # OpenFace will write CSVs here
        pulse_out.mkdir(parents=True, exist_ok=True)

        # Run OpenFace FeatureExtraction
        cmd = [
            OPENFACE_BIN,
            "-f", str(self._video_path),
            "-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp",
            "-out_dir", str(pulse_out)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"OpenFace failed:\n{proc.stdout}\n{proc.stderr}")

        # Get newest CSV produced
        csvs = sorted(pulse_out.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {pulse_out}")
        csv_path = csvs[0]

        summary = self._summarize_csv(csv_path)
        self._append_session_row(summary, csv_path)
        # Cleanup temp dir
        self._tmpdir.cleanup()
        return summary, csv_path

    def _summarize_csv(self, csv_path: pathlib.Path) -> Dict[str, float]:
        keep_means = {k:0.0 for k in [
            "AU01_r","AU02_r","AU04_r","AU06_r","AU12_r","AU15_r","AU20_r","AU25_r","AU26_r","AU45_c"
        ]}
        keep_pose  = {k:0.0 for k in ["pose_Rx","pose_Ry","pose_Rz"]}
        n = 0
        import math
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n += 1
                for k in keep_means:
                    v = row.get(k, "")
                    if v != "": keep_means[k] += float(v)
                for k in keep_pose:
                    v = row.get(k, "")
                    if v != "": keep_pose[k] += float(v)
        if n == 0: return {}
        for k in keep_means: keep_means[k] /= n
        for k in keep_pose:  keep_pose[k]  /= n

        out = {**keep_means, **keep_pose}
        # Friendly aliases
        out["avg_smile"]   = keep_means["AU12_r"]
        out["avg_furrow"]  = keep_means["AU04_r"]
        out["avg_mouthop"] = keep_means["AU26_r"]
        out["blink_presence_mean"] = keep_means["AU45_c"]  # 0..1 proportion across frames
        out["frames"] = n
        out["dur_s"]  = max(0.001, (self._t_end - self._t_start)) if self._t_start else 0.0
        return out

    def _append_session_row(self, summary: Dict[str, float], csv_path: pathlib.Path):
        session_csv = OUT_DIR / "session_summary.csv"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = ["ts","session_id","dur_s","frames",
                  "AU01_r","AU02_r","AU04_r","AU06_r","AU12_r","AU15_r","AU20_r","AU25_r","AU26_r","AU45_c",
                  "pose_Rx","pose_Ry","pose_Rz",
                  "avg_smile","avg_furrow","avg_mouthop","blink_presence_mean",
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
