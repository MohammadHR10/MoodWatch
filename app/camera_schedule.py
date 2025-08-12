import cv2, time
from .openface_pulse import OpenFacePulse, CaptureSpec

# ====== CONFIG ======
CAM_INDEX = 0
FRAME_SIZE = (640, 480)
REQUESTED_FPS = 15

FIRST_OFFSET = 2        # camera ON at T+2s
PULSE_DURATION = 6      # stay ON for 6s
RECURRING_GAP = 100     # then every 100s
REOPEN_COOLDOWN = 0.3   # tiny pause after closing
MIN_LEAD = 0.2          # require next start >= now + 0.2s
# ====================

def open_cam():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)  # macOS backend
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # fine if ignored
    if not cap.isOpened():
        print("[ERR] Could not open camera"); return None
    # warm-up a few frames so exposure/white balance stabilize
    for _ in range(8):
        cap.read(); time.sleep(0.02)
    return cap

def run_preview(duration_s: int) -> bool:
    """Show live video for duration_s AND record frames for OpenFace. Return False if user quits."""
    cap = open_cam()
    if cap is None:
        return True  # skip this pulse but keep scheduler running

    # --- NEW: set up pulse recorder ---
    spec = CaptureSpec(fps=REQUESTED_FPS, size=FRAME_SIZE, fourcc="mp4v")  # use XVID if mp4 fails
    pulse = OpenFacePulse(spec)
    pulse.start()

    t_end = time.monotonic() + duration_s
    summary = None
    try:
        while True:
            if time.monotonic() >= t_end: break
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue

            # record frame for OpenFace
            pulse.write(frame)

            # overlay countdown for the preview
            rem = max(0.0, t_end - time.monotonic())
            cv2.putText(frame, f"ON: {rem:4.1f}s left",
                        (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Webcam (Scheduled)", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):  # q or ESC aborts all
                return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # --- finalize pulse even if user aborted mid-loop ---
        try:
            summary, csv_path = pulse.finish()
            print("OpenFace summary:", summary)
        except Exception as e:
            print("[ERR] OpenFace analysis failed:", e)
        time.sleep(REOPEN_COOLDOWN)
    return True


def next_start_after(anchor, gap, now):
    """
    Given a repeating sequence starting at 'anchor' every 'gap' seconds,
    return the first start strictly in the future with a small lead.
    """
    if now < anchor:
        start = anchor
    else:
        n = int((now - anchor) // gap) + 1
        start = anchor + n * gap
    if start < now + MIN_LEAD:
        start += gap
    return start

def main():
    print("Schedule: T+2s for 6s, then every 100s for 6s. Ctrl+C or q/ESC to stop.")
    t0 = time.monotonic()

    # 1) First pulse at T+FIRST_OFFSET
    first_start = t0 + FIRST_OFFSET
    while True:
        now = time.monotonic()
        if now >= first_start: break
        time.sleep(min(0.1, first_start - now))
    print(f"[{time.strftime('%H:%M:%S')}] Camera ON (first) for {PULSE_DURATION}s")
    keep = run_preview(PULSE_DURATION)
    print(f"[{time.strftime('%H:%M:%S')}] Camera OFF")
    if not keep:
        print("[INFO] Stopped by user."); return

    # 2) Recurring pulses: every RECURRING_GAP seconds, each PULSE_DURATION
    anchor = first_start + RECURRING_GAP  # first recurring anchor after the first pulse
    try:
        while True:
            now = time.monotonic()
            abs_start = next_start_after(anchor, RECURRING_GAP, now)

            # wait until abs_start (light sleep, low CPU)
            while True:
                now = time.monotonic()
                abs_start = next_start_after(anchor, RECURRING_GAP, now)

                # wait until abs_start (with countdown display)
                while True:
                    now = time.monotonic()
                    delta = abs_start - now
                    if delta <= 0:
                        break
                    print(f"\rNext camera ON in {delta:5.1f} sec", end="", flush=True)
                    time.sleep(1)  # update every second

                print("\n" + f"[{time.strftime('%H:%M:%S')}] Camera ON for {PULSE_DURATION}s")
                keep = run_preview(PULSE_DURATION)
                print(f"[{time.strftime('%H:%M:%S')}] Camera OFF")
                if not keep:
                    print("[INFO] Stopped by user."); break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")

if __name__ == "__main__":
    main()
