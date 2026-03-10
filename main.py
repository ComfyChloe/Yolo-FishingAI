import cv2
import numpy as np
import time
import win32gui
import win32ui
import win32con
import win32api
import ctypes
import keyboard
import sys
import os
import json
import torch
from ultralytics import YOLO

def capture_window(hwnd):
    rect = win32gui.GetClientRect(hwnd)
    w, h = rect[2], rect[3]
    if w <= 0 or h <= 0:
        return None
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    # PrintWindow (PW_RENDERFULLCONTENT=2)
    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype="uint8")
    img.shape = (h, w, 4)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    if result != 1:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ======================
# Self-learning helpers
# ======================
DIFF_BUCKET_SIZE = 20       # px — group distances into buckets for lookup
SPEED_TIER_BOUNDS = [40, 100]  # px/s boundaries: slow < 40 < medium < 100 < fast
LEARNING_RATE = 0.02        # EMA step — conservative (~50+ sessions to converge)
LEARN_CONFIDENCE_THRESHOLD = 10  # samples before trusting learned value

def _default_learned_data():
    return {
        "version": 1,
        "total_sessions": 0,
        "tiers": {
            "slow":   {"sessions": 0, "hold_table": {}},
            "medium": {"sessions": 0, "hold_table": {}},
            "fast":   {"sessions": 0, "hold_table": {}},
        },
    }

def load_learned_data(path):
    if not os.path.exists(path):
        return _default_learned_data()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "tiers" not in data:
            return _default_learned_data()
        return data
    except Exception:
        return _default_learned_data()

def save_learned_data(data, path):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on Windows NTFS
    except Exception as e:
        print(f"[LEARN] Failed to save: {e}")

def get_speed_tier(avg_speed):
    if avg_speed < SPEED_TIER_BOUNDS[0]:
        return "slow"
    elif avg_speed < SPEED_TIER_BOUNDS[1]:
        return "medium"
    return "fast"

def get_diff_bucket(diff):
    return round(diff / DIFF_BUCKET_SIZE)

def get_learned_hold(diff, speed_tier, learned_data, formula_hold):
    """Blend learned hold time with formula-computed hold. Returns (hold, is_learned)."""
    bucket = str(get_diff_bucket(diff))
    table = learned_data["tiers"].get(speed_tier, {}).get("hold_table", {})
    entry = table.get(bucket)
    if entry is None:
        return formula_hold, False
    learned = entry["hold"]
    count = entry["count"]
    if count >= LEARN_CONFIDENCE_THRESHOLD:
        blended = learned * 0.7 + formula_hold * 0.3
    else:
        blended = learned * 0.3 + formula_hold * 0.7
    return max(MIN_HOLD, min(MAX_HOLD, blended)), True

def process_session_learning(session_frames, speed_tier, learned_data):
    """Update learned hold table from session frame data. Modifies learned_data in-place."""
    tier = learned_data["tiers"].setdefault(speed_tier, {"sessions": 0, "hold_table": {}})
    tier["sessions"] += 1
    learned_data["total_sessions"] += 1
    table = tier.setdefault("hold_table", {})
    for i in range(len(session_frames) - 1):
        diff_i, hold_i, _ = session_frames[i]
        diff_next, _, _ = session_frames[i + 1]
        bucket = str(get_diff_bucket(diff_i))
        entry = table.get(bucket, {"hold": hold_i, "count": 0})
        old_hold = entry["hold"]
        if abs(diff_next) < abs(diff_i):
            # Tracking improved — reinforce this hold time
            new_hold = old_hold + LEARNING_RATE * (hold_i - old_hold)
        else:
            # Tracking worsened — nudge away from this hold time
            new_hold = old_hold - LEARNING_RATE * (hold_i - old_hold)
        entry["hold"] = max(MIN_HOLD, min(MAX_HOLD, new_hold))
        entry["count"] = entry["count"] + 1
        table[bucket] = entry

# ======================
# Load configuration
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "recast_interval": 1.0,
    "force_cpu": False,
    "margin_x": 0.2,
    "margin_y": 0.1,
    "overlay": True,
    "hotkeys": {
        "toggle_input": "ctrl+x",
        "exit": "ctrl+e",
        "recapture_window": "ctrl+r",
    },
}


def load_config():
    """Load config.json and fall back to defaults if missing or invalid."""
    # Deep copy defaults so hotkeys can be merged safely.
    cfg = DEFAULT_CONFIG.copy()
    cfg["hotkeys"] = DEFAULT_CONFIG["hotkeys"].copy()

    if not os.path.exists(CONFIG_PATH):
        # Write the template on first run.
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
            print("config.json が存在しないため、新規作成しました。")
        except Exception as e:
            print(f"config.json の作成に失敗しました: {e}")
        return cfg

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if isinstance(user_cfg, dict):
            for k, v in user_cfg.items():
                if k == "hotkeys" and isinstance(v, dict):
                    cfg["hotkeys"].update(v)
                else:
                    cfg[k] = v
    except Exception as e:
        print(f"config.json の読み込みに失敗しました。デフォルト値を使用します: {e}")
    return cfg


config = load_config()

# ======================
# 1. Settings and customization
# ======================
WINDOW_NAME = "VRChat"
input_enabled = False

MARGIN_X = float(config.get("margin_x", DEFAULT_CONFIG["margin_x"]))
MARGIN_Y = float(config.get("margin_y", DEFAULT_CONFIG["margin_y"]))

OVERLAY_ENABLED = bool(config.get("overlay", DEFAULT_CONFIG["overlay"]))

# Control parameters
CYCLE_TIME = 0.12           # Click cycle interval
DETECTION_CYCLE = 0.005    # Detection wait interval
SMOOTH_FACTOR = 0.5        # Coordinate smoothing factor

# Hold timing
BASE_HOLD = 0.035
STEP_ADJUST = 0.06
MAX_HOLD = 0.08
MIN_HOLD = 0.02
UP_COUNTER_HOLD = 0.02
DOWN_COUNTER_HOLD = 0.08
JITTER_PIXELS = 4          # Small mouse jitter amplitude (px)

# Thresholds
BOOST_THRESHOLD = 80       # Boost distance threshold (px)
SPEED_THRESHOLD = 70       # Brake speed threshold
RECAST_INTERVAL = float(config.get("recast_interval", DEFAULT_CONFIG["recast_interval"]))  # Seconds before recast
LOST_TRACK_THRESHOLD = 1.5 # Grace period for keeping last coordinates

# Hotkey settings
hotkeys_cfg = config.get("hotkeys", {})
TOGGLE_INPUT_HOTKEY = hotkeys_cfg.get("toggle_input", DEFAULT_CONFIG["hotkeys"]["toggle_input"])
EXIT_HOTKEY = hotkeys_cfg.get("exit", DEFAULT_CONFIG["hotkeys"]["exit"])
RECAPTURE_WINDOW_HOTKEY = hotkeys_cfg.get("recapture_window", DEFAULT_CONFIG["hotkeys"]["recapture_window"])

# State tracking
last_detected_time = time.time()
last_fish_cy = last_bar_cy = prev_bar_cy = None
fish_lost_at = bar_lost_at = prev_diff = 0
prev_time = time.time()
locked_hwnd = None  # Lock to first captured VRChat window
jitter_direction = 1   # 1 = right first, flips to -1 = left each fish
jitter_done = False    # Reset when fish disappears, fires once on first detection
sessions_total = 0     # Total completed fishing sessions
was_fishing = False    # True while both bar+fish were tracked this session
prev_fish_cy = None    # Previous frame's fish Y — for speed calculation
session_frames = []    # Per-frame (diff, hold_time, fish_speed) for learning
current_speed_tier = "medium"  # Running speed tier classification
hold_source = "F"      # "L" = learned, "F" = formula (for overlay)

# Self-learning data
LEARN_FILE = os.path.join(BASE_DIR, "learned_params.json")
learned_data = load_learned_data(LEARN_FILE)
if learned_data["total_sessions"] > 0:
    tier_stats = [f"{tn}: {td['sessions']}s" for tn, td in learned_data["tiers"].items() if td.get("sessions", 0) > 0]
    print(f"[LEARN] Loaded {learned_data['total_sessions']} sessions — {', '.join(tier_stats) if tier_stats else 'no tier data'}")
else:
    print("[LEARN] No learned data yet — starting fresh with formula values")

# Device selection with force_cpu support
# Supported: RTX 20-series (Turing), 30-series (Ampere), 40-series (Ada Lovelace), 50-series (Blackwell)
GPU_FAMILIES = {
    7: "Turing (RTX 20-series)",
    8: "Ampere/Ada (RTX 30/40-series)",
    9: "Ada (RTX 40-series)",
    10: "Blackwell (RTX 50-series)",
}
force_cpu = bool(config.get("force_cpu", DEFAULT_CONFIG["force_cpu"]))
if force_cpu:
    device = torch.device("cpu")
    print("Device: cpu (force_cpu=true)")
elif not torch.cuda.is_available():
    device = torch.device("cpu")
    print("Device: cpu (CUDA not available — check PyTorch/driver install)")
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA built with: {torch.version.cuda or 'None (CPU-only build)'}")
    print()
    print("  To fix: install CUDA-enabled PyTorch for RTX 30/40/50 series:")
    print("    pip uninstall -y torch torchvision torchaudio")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
else:
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    arch = GPU_FAMILIES.get(props.major, f"Compute {props.major}.{props.minor}")
    print(f"Device: cuda — {props.name} ({props.total_memory // 1024**2} MB VRAM)")
    print(f"  Architecture  : {arch} (SM {props.major}.{props.minor})")
    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA version  : {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True  # optimise for fixed-size inputs

USE_HALF = (device.type == "cuda")
model = YOLO("best.pt")

# Warmup — lets Ultralytics initialise + fuse layers before the main loop
# half= is passed here so AutoBackend handles dtype conversion correctly
print("Warming up model...")
_dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(_dummy_img, conf=0.4, verbose=False, imgsz=640, half=USE_HALF, device=device)
del _dummy_img
print("Warmup done.")

def toggle_input():
    global input_enabled
    input_enabled = not input_enabled
    status = "ENABLED" if input_enabled else "DISABLED"
    print(f"--- INPUT {status} ---")

def recapture_window():
    global locked_hwnd, last_detected_time, last_fish_cy, last_bar_cy, prev_bar_cy
    global fish_lost_at, bar_lost_at, prev_diff, was_fishing, jitter_done
    global prev_fish_cy, session_frames
    locked_hwnd = None
    last_detected_time = time.time()
    last_fish_cy = last_bar_cy = prev_bar_cy = None
    fish_lost_at = bar_lost_at = prev_diff = 0
    jitter_done = False
    prev_fish_cy = None
    session_frames = []
    if was_fishing:
        was_fishing = False
    print(f"--- RECAPTURE: Scanning for focused {WINDOW_NAME} window ---")

keyboard.add_hotkey(TOGGLE_INPUT_HOTKEY, toggle_input)
keyboard.add_hotkey(RECAPTURE_WINDOW_HOTKEY, recapture_window)

# Increase Windows timer resolution to 1ms for accurate short sleeps
ctypes.windll.winmm.timeBeginPeriod(1)

print(f"--- AI Fishing Full System (ROI + Hold + Recast) 起動 ---")

# ======================
# 2. Main loop
# ======================
while True:
    if keyboard.is_pressed(EXIT_HOTKEY): break

    # Use locked window handle; only search for a new one if it's gone
    if locked_hwnd and win32gui.IsWindow(locked_hwnd) and win32gui.IsWindowVisible(locked_hwnd):
        hwnd = locked_hwnd
    else:
        # Prioritize the currently active/foreground window
        fg_hwnd = win32gui.GetForegroundWindow()
        fg_title = win32gui.GetWindowText(fg_hwnd)
        
        if WINDOW_NAME in fg_title and win32gui.IsWindowVisible(fg_hwnd):
            # Foreground window is VRChat — lock to it
            hwnd = fg_hwnd
        else:
            # Foreground isn't VRChat, search for any VRChat window
            hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        
        if not hwnd or not win32gui.IsWindowVisible(hwnd):
            time.sleep(1); continue
        locked_hwnd = hwnd
        print(f"Locked to window: {WINDOW_NAME} (hwnd={hwnd:#x})")

    full_frame = capture_window(hwnd)
    if full_frame is None: continue

    # Calculate the ROI bounds.
    h, w = full_frame.shape[:2]
    x_start, x_end = int(w * MARGIN_X), int(w * (1 - MARGIN_X))
    y_start, y_end = int(h * MARGIN_Y), int(h * (1 - MARGIN_Y))
    roi_frame = full_frame[y_start:y_end, x_start:x_end]

    # Run AI inference.
    results = model.predict(roi_frame, conf=0.3, verbose=False, imgsz=640, half=USE_HALF)
    
    current_fish_cy_raw = None
    all_bar_y_coords = []
    current_time = time.time()
    dt = current_time - prev_time
    did_click = False
    hold_time = 0.0
    debug_text = "SEARCHING..."
    color = (0, 0, 255)

    # Parse results and translate coordinates back to full-frame space.
    for result in results:
        for box in result.boxes:
            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = rx1 + x_start, ry1 + y_start, rx2 + x_start, ry2 + y_start
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if OVERLAY_ENABLED:
                cv2.rectangle(full_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                cv2.putText(
                    full_frame,
                    f"{name} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            if name == 'fish icon': current_fish_cy_raw = (y1 + y2) / 2
            elif name == 'white bar':
                all_bar_y_coords.extend([y1, y2])

    # Update and retain bar coordinates.
    if all_bar_y_coords:
        mid_y = (min(all_bar_y_coords) + max(all_bar_y_coords)) / 2
        bar_v = (mid_y - prev_bar_cy) / dt if prev_bar_cy and dt > 0 else 0
        last_bar_cy = (last_bar_cy * (1-SMOOTH_FACTOR)) + (mid_y * SMOOTH_FACTOR) if last_bar_cy else mid_y
        prev_bar_cy = mid_y
        bar_lost_at = 0
    else:
        if bar_lost_at == 0: bar_lost_at = current_time
        if current_time - bar_lost_at > LOST_TRACK_THRESHOLD: last_bar_cy = None

    # Track whether a fishing session is active.
    if last_bar_cy is not None and last_fish_cy is not None:
        was_fishing = True

    # Update and retain fish coordinates.
    if current_fish_cy_raw is not None:
        # Fire one micro jitter on first detection of each new fish
        if not jitter_done and input_enabled:
            cx, cy = w // 2, h // 2
            jitter_offset = JITTER_PIXELS * jitter_direction
            win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(cx + jitter_offset, cy))
            win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(cx, cy))
            jitter_done = True
            jitter_direction = -jitter_direction  # Flip for next fish
        # Track fish speed for learning (px/s between consecutive fish detections)
        fish_speed = 0.0
        if prev_fish_cy is not None and dt > 0:
            fish_speed = abs(current_fish_cy_raw - prev_fish_cy) / dt
        prev_fish_cy = current_fish_cy_raw
        last_fish_cy, last_detected_time, fish_lost_at = current_fish_cy_raw, current_time, 0
    else:
        prev_fish_cy = None  # Reset speed tracking when fish lost
        if fish_lost_at == 0: fish_lost_at = current_time
        if current_time - fish_lost_at > LOST_TRACK_THRESHOLD:
            last_fish_cy = None
            jitter_done = False  # Allow jitter again on next fish

    # Control logic.
    if last_bar_cy is not None and last_fish_cy is not None:
        diff = last_fish_cy - last_bar_cy
        fish_visible = current_fish_cy_raw is not None

        if fish_visible:
            # Fish is actively detected — velocity-based brake only
            if all_bar_y_coords and bar_v < -SPEED_THRESHOLD:
                status, hold_time, color = "!! UP-BRAKE !!", UP_COUNTER_HOLD, (255, 255, 255)
            elif all_bar_y_coords and bar_v > SPEED_THRESHOLD:
                status, hold_time, color = "!! DOWN-BRAKE !!", DOWN_COUNTER_HOLD, (255, 50, 50)
            else:
                is_boost = abs(diff) >= BOOST_THRESHOLD
                adj = STEP_ADJUST * 2.0 if is_boost else STEP_ADJUST
                if diff < 0:
                    formula_hold = min(BASE_HOLD + adj, MAX_HOLD)
                    hold_time, is_learned = get_learned_hold(diff, current_speed_tier, learned_data, formula_hold)
                    hold_source = "L" if is_learned else "F"
                    status, color = ("BOOST-UP" if is_boost else "ASCENDING"), (0, 255, 255)
                else:
                    formula_hold = max(BASE_HOLD - adj, MIN_HOLD)
                    hold_time, is_learned = get_learned_hold(diff, current_speed_tier, learned_data, formula_hold)
                    hold_source = "L" if is_learned else "F"
                    status, color = ("BOOST-DOWN" if is_boost else "DESCENDING"), (255, 200, 0)
            prev_diff = diff  # Only update when fish is real to avoid false brake on re-detection
            # Record frame for learning (only when fish is visible and we have speed data)
            if current_fish_cy_raw is not None:
                session_frames.append((diff, hold_time, fish_speed))
        else:
            # Fish is hidden (inside bar) — skip all velocity/brake logic to prevent wild swings.
            # Use a neutral hold based only on last known position sign.
            if diff < 0:
                status, hold_time, color = "HOLDING-UP", BASE_HOLD, (0, 180, 180)
            else:
                status, hold_time, color = "HOLDING-DOWN", BASE_HOLD, (180, 140, 0)
            # prev_diff intentionally NOT updated — avoids false brake trigger when fish reappears

        if not all_bar_y_coords: status += " (BAR LOST)"

        if input_enabled:
            lParam = win32api.MAKELONG(w // 2, h // 2)
            win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
            time.sleep(hold_time)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
            did_click = True
        else:
            time.sleep(hold_time)
        
        debug_text = f"{status} | HOLD: {hold_time:.3f} [{hold_source}]"
        if OVERLAY_ENABLED:
            cv2.line(full_frame, (15, int(last_bar_cy)), (15, int(last_fish_cy)), color, 2)

    # Recovery logic (recast).
    elif current_time - last_detected_time > RECAST_INTERVAL:
        if was_fishing:
            sessions_total += 1
            # --- Self-learning: update hold table from this session ---
            if len(session_frames) >= 5:
                speeds = [s for _, _, s in session_frames if s > 0]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                current_speed_tier = get_speed_tier(avg_speed)
                process_session_learning(session_frames, current_speed_tier, learned_data)
                save_learned_data(learned_data, LEARN_FILE)
                tier_d = learned_data["tiers"][current_speed_tier]
                print(f"[SESSION #{sessions_total}] Tier={current_speed_tier} (avg {avg_speed:.0f} px/s) | "
                      f"{len(session_frames)} frames | Tier sessions: {tier_d['sessions']}")
            else:
                print(f"[SESSION #{sessions_total}] Complete (insufficient frames to learn)")
            session_frames = []
            was_fishing = False
        if input_enabled:
            lParam = win32api.MAKELONG(w // 2, h // 2)
            win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
            time.sleep(0.15)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
            did_click = True
        last_detected_time = current_time
        debug_text = "RECASTING..."
        color = (0, 255, 0)

    # Overlay information.
    if OVERLAY_ENABLED:
        cv2.rectangle(full_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)  # ROI outline
        input_txt = f"[INPUT: {'ON' if input_enabled else 'OFF'} ({TOGGLE_INPUT_HOTKEY})]"
        cv2.putText(full_frame, input_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(full_frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Sessions counter + learning tier display
        cv2.putText(full_frame, f"Sessions: {sessions_total}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)
        tier_color = {"slow": (0, 200, 0), "medium": (0, 200, 255), "fast": (80, 80, 255)}.get(current_speed_tier, (200, 200, 200))
        cv2.putText(full_frame, f"Tier: {current_speed_tier}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tier_color, 2)
        tier_strs = [f"{tn[0].upper()}:{learned_data['tiers'].get(tn, {}).get('sessions', 0)}s"
                     for tn in ("slow", "medium", "fast")
                     if learned_data["tiers"].get(tn, {}).get("sessions", 0) > 0]
        if tier_strs:
            cv2.putText(full_frame, " ".join(tier_strs), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.imshow("AI Fishing (Full-Integrated)", cv2.resize(full_frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    prev_time = current_time
    time.sleep(max(0, CYCLE_TIME - hold_time) if did_click else DETECTION_CYCLE)

ctypes.windll.winmm.timeEndPeriod(1)
cv2.destroyAllWindows()