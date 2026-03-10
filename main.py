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

# Window capture helper (supports background capture)
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


def detect_progress_bar(frame, x_start, y_start, x_end, y_end):
    """Detect green progress bar fill in the left portion of the fishing ROI.
    Returns fill ratio 0.0-1.0 (bottom-up fill), or None if not found."""
    roi_width = x_end - x_start
    search_end = x_start + int(roi_width * 0.4)
    region = frame[y_start:y_end, x_start:search_end]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 80, 80), (95, 255, 255))
    col_sums = mask.sum(axis=0) // 255
    if col_sums.max() < 15:
        return None
    peak_col = int(col_sums.argmax())
    col_lo = max(0, peak_col - 3)
    col_hi = min(mask.shape[1], peak_col + 4)
    strip = mask[:, col_lo:col_hi]
    row_hits = strip.max(axis=1)
    green_rows = np.where(row_hits > 0)[0]
    if len(green_rows) < 5:
        return None
    green_height = green_rows[-1] - green_rows[0] + 1
    roi_height = y_end - y_start
    return min(1.0, max(0.0, green_height / roi_height))


# ======================
# Self-learning helpers
# ======================
DIFF_BUCKET_SIZE = 20         # px — group diffs into buckets for lookup
LEARN_BASE_RATE = 0.025       # EMA base step (scaled by quality score)
LEARN_COUNT_CAP = 300.0       # Float cap on bucket count (prevents entrenchment)
LEARN_COUNT_DECAY = 0.997     # Per-session multiplier on all counts (~50% after 231 sessions)

def _default_tier():
    return {"sessions": 0, "avg_quality": 0.0, "hold_table": {}}

def _default_learned_data():
    return {
        "version": 2,
        "total_sessions": 0,
        "tiers": {
            "slow":     _default_tier(),
            "medium":   _default_tier(),
            "fast":     _default_tier(),
            "veryfast": _default_tier(),
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
        if data.get("version") != 2:
            print("[LEARN] Schema version mismatch — resetting to v2 (old data discarded)")
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

# Rarity → speed tier mapping (replaces velocity-based classification)
RARITY_TO_TIER = {
    "trash":    "slow",
    "abundant": "slow",
    "common":   "medium",   # confirmed faster than trash/abundant
    "curious":  "medium",
    "elusive":  "medium",
    "fabled":   "fast",
    "relic":    "fast",
    "mythic":   "veryfast",
    "exotic":   "veryfast",
    "unknown":  "medium",   # fallback — logged to console for identification
}

RARITY_OVERLAY_COLORS = {  # BGR for OpenCV
    "trash":    (100, 100, 100),
    "abundant": (180, 180, 180),
    "common":   (50,  210, 80),
    "curious":  (210, 180, 0),
    "elusive":  (190, 80,  180),
    "fabled":   (0,   180, 220),
    "relic":    (50,  120, 240),
    "mythic":   (200, 50,  210),
    "exotic":   (140, 30,  255),
    "unknown":  (200, 200, 200),
}

def sample_fish_color(frame, x1, y1, x2, y2):
    """Sample the dominant non-background color of the fish bounding box.
    Returns median HSV array [H, S, V] or None if insufficient pixels.
    Center-crops to inner 70% to avoid edge bleed from the UI bar background."""
    h_sz, w_sz = y2 - y1, x2 - x1
    my, mx = int(h_sz * 0.15), int(w_sz * 0.15)
    patch = frame[max(0, y1 + my):max(0, y2 - my), max(0, x1 + mx):max(0, x2 - mx)]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # S>100: excludes bar BG and white-fish edge tints (S~70-99)
    # V>90:  excludes dark anti-alias fringe pixels (V=52 type false positives)
    # V<220: excludes blown-out white glare
    valid = (hsv[:, 1] > 100) & (hsv[:, 2] > 90) & (hsv[:, 2] < 220)
    if valid.sum() < 10:
        return None
    return np.median(hsv[valid], axis=0).astype(int)

def _hue_name(h, s, v):
    """Human-readable color description for debug logs."""
    if s < 50:
        return "white/gray" if v >= 110 else "black/dark"
    if h < 14 or h >= 165: return "red/orange-red"
    if h < 25:  return "orange"
    if h < 40:  return "yellow/gold"
    if h < 88:  return "green"
    if h < 132: return "blue"
    if h < 142: return "blue-purple"
    return "purple/violet"

def classify_fish_rarity(hsv_med):
    """Classify fish rarity from HSV median. Returns (rarity_str, is_known).
    OpenCV HSV: H 0-180, S/V 0-255."""
    h, s, v = int(hsv_med[0]), int(hsv_med[1]), int(hsv_med[2])
    if s < 50:
        return ("trash" if v < 110 else "abundant"), True
    # Confirmed mappings (live sampling + hex codes):
    #   relic   = orange-red     (H=11 confirmed)
    #   fabled  = gold/yellow    (H ~12-40)
    #   common  = bright green   (H ~40-88)
    #   curious = blue #201f42   (H=111 confirmed)
    #   exotic  = purple #af0fc2 (H=147 confirmed)
    # Unconfirmed: elusive, mythic — will print [RARITY] unknown to calibrate
    if h < 14 or h >= 165: return "relic",   True   # orange-red (confirmed at H=11)
    if 14 <= h < 40:       return "fabled",  True   # gold/yellow (confirmed)
    if 40 <= h < 88:       return "common",  True   # bright green (confirmed)
    if 88 <= h < 132:      return "curious", True   # blue (confirmed at H=111)
    if 132 <= h < 142:     return "mythic",  True   # blue-purple (placeholder)
    if 142 <= h < 165:     return "exotic",  True   # purple-violet (confirmed #af0fc2 ≈ H=147)
    return "unknown", False  # unrecognised

def get_diff_bucket(diff):
    return round(diff / DIFF_BUCKET_SIZE)

def get_learned_hold(diff, speed_tier, learned_data, formula_hold):
    """Blend learned hold time with formula hold using continuous alpha. Returns (hold, is_learned)."""
    bucket = str(get_diff_bucket(diff))
    table = learned_data["tiers"].get(speed_tier, {}).get("hold_table", {})
    entry = table.get(bucket)
    if entry is None:
        return formula_hold, False
    learned = entry["hold"]
    count = float(entry.get("count", 0))
    # Continuous alpha: reaches ~0.75 at count=100, ~0.90 at count=300
    alpha = min(0.92, count / (count + 33.0)) if count > 0 else 0.0
    blended = alpha * learned + (1.0 - alpha) * formula_hold
    return max(MIN_HOLD, min(MAX_HOLD, blended)), True

def process_session_learning(session_frames, quality, speed_tier, learned_data):
    """Update learned hold table from session frame data. Modifies learned_data in-place.
    quality: float 0.0–1.0 (progress bar fill ratio for this session)."""
    tier = learned_data["tiers"].setdefault(speed_tier, _default_tier())
    tier["sessions"] += 1
    # EMA of session quality (how well we tracked this tier over time)
    tier["avg_quality"] = tier.get("avg_quality", 0.0) * 0.9 + quality * 0.1
    learned_data["total_sessions"] += 1

    # Quality-gradient learning rate: 0.3x at quality=0, ramps to 2.0x at quality=1.0
    rate = LEARN_BASE_RATE * max(0.3, min(2.0, 0.3 + 1.7 * quality))

    table = tier.setdefault("hold_table", {})

    # --- Step 1: aggregate session frames by bucket ---
    bucket_data = {}  # bucket -> {hold_sum, speed_sum, count}
    for diff_i, hold_i, speed_i, _formula_i in session_frames:
        b = str(get_diff_bucket(diff_i))
        if b not in bucket_data:
            bucket_data[b] = {"hold_sum": 0.0, "speed_sum": 0.0, "n": 0}
        bucket_data[b]["hold_sum"] += hold_i
        bucket_data[b]["speed_sum"] += speed_i
        bucket_data[b]["n"] += 1

    # Compute tier-average fish speed for erratic-frame detection
    all_speeds = [s for _, _, s, _ in session_frames]
    tier_avg_speed = (sum(all_speeds) / len(all_speeds)) if all_speeds else 1.0

    # --- Step 2: apply count decay to ALL existing buckets ---
    for entry in table.values():
        entry["count"] = max(0.0, entry["count"] * LEARN_COUNT_DECAY)

    # --- Step 3: EMA update for visited buckets ---
    for b, bd in bucket_data.items():
        avg_hold = bd["hold_sum"] / bd["n"]
        avg_speed = bd["speed_sum"] / bd["n"]
        # Halve rate when fish was erratic in this bucket (unreliable signal)
        bucket_rate = rate * (0.5 if tier_avg_speed > 0 and avg_speed > 2.0 * tier_avg_speed else 1.0)
        entry = table.get(b, {"hold": avg_hold, "count": 0.0})
        old_hold = entry["hold"]
        entry["hold"] = max(MIN_HOLD, min(MAX_HOLD, old_hold + bucket_rate * (avg_hold - old_hold)))
        entry["count"] = min(LEARN_COUNT_CAP, entry["count"] + 1.0)
        table[b] = entry

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
    """Load config.json and use defaults if it is missing or invalid."""
    # Deep copy defaults so hotkeys can be merged safely.
    cfg = DEFAULT_CONFIG.copy()
    cfg["hotkeys"] = DEFAULT_CONFIG["hotkeys"].copy()

    if not os.path.exists(CONFIG_PATH):
        # Write the template file on first run.
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

# ROI boundary settings
MARGIN_X = float(config.get("margin_x", DEFAULT_CONFIG["margin_x"]))
MARGIN_Y = float(config.get("margin_y", DEFAULT_CONFIG["margin_y"]))

OVERLAY_ENABLED = bool(config.get("overlay", DEFAULT_CONFIG["overlay"]))

DEBUG_COLOR_LOG = True      # Print [COLOR] each frame a fish is detected (calibration aid)

# Control parameters
CYCLE_TIME = 0.055          # Click cycle interval (~18Hz control rate, was 0.12)
DETECTION_CYCLE = 0.005    # Detection wait interval when not clicking
SMOOTH_FACTOR = 0.5        # Coordinate smoothing factor

# Hold timing
BASE_HOLD = 0.035      # Neutral hold — equilibrium where bar neither rises nor falls
GRAVITY_BIAS = 0.005   # Extra hold to offset bar's natural gravity sag (bar sinks if hold < BASE+bias)
STEP_ADJUST = 0.06     # Kp: proportional gain
D_GAIN = 0.008         # Kd: derivative gain — damps overshoot via measured bar velocity
MAX_HOLD = 0.08
MIN_HOLD = 0.02
CENTER_ZONE_RATIO = 0.25  # Fraction of bar_half_height that counts as "well centered"
UP_COUNTER_HOLD = 0.02
DOWN_COUNTER_HOLD = 0.08
JITTER_PIXELS = 4          # Small mouse jitter amplitude (px)

# Thresholds
BOOST_THRESHOLD = 80       # Boost distance threshold (px)
SPEED_THRESHOLD = 70       # Brake speed threshold
CAUGHT_THRESHOLD = 0.3     # Progress bar fill ratio to classify as "caught"
FISH_CONF_MIN = 0.25       # Minimum confidence to accept a 'fish icon' detection
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
current_speed_tier = "medium"   # Running speed tier classification
current_fish_rarity = "unknown" # Last detected fish rarity (from color)
_rarity_vote_buffer = []        # Multi-frame voting buffer (up to 5 recent rarity samples)
hold_source = "F"               # "L" = learned, "F" = formula (for overlay)
last_known_progress = 0.0  # Progress bar fill ratio (0.0-1.0)
fish_caught = 0            # Total caught count
fish_escaped = 0           # Total escaped count
bar_half_height = None     # Half the white bar height (px), smoothed
bar_v = 0.0                # Bar vertical velocity (px/s), positive = moving down

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

def _roi_imgsz(roi_h, roi_w):
    """Return a multiple-of-32 imgsz matching the ROI's longest side, capped at 1280.
    Running at the ROI's native resolution avoids crushing small fish icons that
    would lose critical pixels if downscaled to a fixed 640px grid."""
    longest = max(roi_h, roi_w)
    size = min(1280, ((longest + 31) // 32) * 32)
    return max(size, 320)  # floor at 320 for degenerate windows

# Warmup — primes CUDA kernels at the size used in the main loop
print("Warming up model...")
_dummy_img = np.zeros((1280, 1280, 3), dtype=np.uint8)
model.predict(_dummy_img, conf=0.2, verbose=False, imgsz=1280, half=USE_HALF, device=device)
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
    global prev_fish_cy, session_frames, last_known_progress, bar_half_height
    global current_fish_rarity, current_speed_tier, _rarity_vote_buffer
    locked_hwnd = None
    last_detected_time = time.time()
    last_fish_cy = last_bar_cy = prev_bar_cy = None
    fish_lost_at = bar_lost_at = prev_diff = 0
    jitter_done = False
    prev_fish_cy = None
    session_frames = []
    last_known_progress = 0.0
    bar_half_height = None
    current_fish_rarity = "unknown"
    current_speed_tier = "medium"
    _rarity_vote_buffer = []
    if was_fishing:
        was_fishing = False
    print(f"--- RECAPTURE: Scanning for focused {WINDOW_NAME} window ---")

keyboard.add_hotkey(TOGGLE_INPUT_HOTKEY, toggle_input)
keyboard.add_hotkey(RECAPTURE_WINDOW_HOTKEY, recapture_window)

# Increase Windows timer resolution to 1ms for accurate short sleeps
ctypes.windll.winmm.timeBeginPeriod(1)

print(f"--- AI Fishing Full System (ROI + Hold + Recast) 起動 ---")

# 2. Main loop
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

    # Calculate ROI boundaries.
    h, w = full_frame.shape[:2]
    x_start, x_end = int(w * MARGIN_X), int(w * (1 - MARGIN_X))
    y_start, y_end = int(h * MARGIN_Y), int(h * (1 - MARGIN_Y))
    roi_frame = full_frame[y_start:y_end, x_start:x_end]
    _imgsz = _roi_imgsz(roi_frame.shape[0], roi_frame.shape[1])

    # Run inference at the ROI's native size (rounded to 32, max 1280) so small fish
    # icons aren't crushed. conf=0.2 passes weak candidates; FISH_CONF_MIN gates fish.
    results = model.predict(roi_frame, conf=0.2, verbose=False, imgsz=_imgsz, half=USE_HALF)
    
    current_fish_cy_raw = None
    all_bar_y_coords = []
    current_time = time.time()
    dt = current_time - prev_time
    did_click = False
    hold_time = 0.0
    debug_text = "SEARCHING..."
    color = (0, 0, 255)

    # Parse detections and map them back to full-frame coordinates.
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

            if name == 'fish icon':
                if conf < FISH_CONF_MIN:
                    continue  # skip very weak fish detections
                current_fish_cy_raw = (y1 + y2) / 2
                hsv_med = sample_fish_color(full_frame, x1, y1, x2, y2)
                if hsv_med is not None:
                    rarity, known = classify_fish_rarity(hsv_med)
                    _cname = _hue_name(int(hsv_med[0]), int(hsv_med[1]), int(hsv_med[2]))
                    if DEBUG_COLOR_LOG:
                        lock_tag = " [locked]" if current_fish_rarity != "unknown" else ""
                        print(f"[COLOR] {rarity} | {_cname} (H={hsv_med[0]} S={hsv_med[1]} V={hsv_med[2]}){lock_tag}")
                    # Multi-frame voting: lock rarity once 2 of last 3 samples agree
                    if current_fish_rarity == "unknown":
                        if known:
                            _rarity_vote_buffer.append(rarity)
                            if len(_rarity_vote_buffer) > 5:
                                _rarity_vote_buffer.pop(0)
                            recent = _rarity_vote_buffer[-3:]
                            if len(recent) >= 2 and recent.count(recent[-1]) >= 2:
                                current_fish_rarity = recent[-1]
                                current_speed_tier = RARITY_TO_TIER.get(current_fish_rarity, "medium")
                        else:
                            print(f"[RARITY] Unknown | {_cname} (H={hsv_med[0]} S={hsv_med[1]} V={hsv_med[2]}) — please report!")
                elif DEBUG_COLOR_LOG and current_fish_rarity == "unknown":
                    print("[COLOR] insufficient saturated pixels (fish may be white/abundant)")
            elif name == 'white bar':
                # Validate whiteness — reject green progress bar false positives
                cx_s, cy_s = (x1 + x2) // 2, (y1 + y2) // 2
                patch = full_frame[max(0, cy_s-2):cy_s+3, max(0, cx_s-2):cx_s+3]
                if patch.size > 0:
                    avg_bgr = patch.mean(axis=(0, 1))
                    if avg_bgr[0] >= 200 and avg_bgr[1] >= 200 and avg_bgr[2] >= 200:
                        all_bar_y_coords.extend([y1, y2])

    # Update and retain bar coordinates.
    if all_bar_y_coords:
        mid_y = (min(all_bar_y_coords) + max(all_bar_y_coords)) / 2
        raw_half = (max(all_bar_y_coords) - min(all_bar_y_coords)) / 2
        if raw_half > 5:
            bar_half_height = raw_half if bar_half_height is None else bar_half_height * 0.8 + raw_half * 0.2
        bar_v = (mid_y - prev_bar_cy) / dt if prev_bar_cy and dt > 0 else 0
        last_bar_cy = (last_bar_cy * (1-SMOOTH_FACTOR)) + (mid_y * SMOOTH_FACTOR) if last_bar_cy else mid_y
        prev_bar_cy = mid_y
        bar_lost_at = 0
    else:
        if bar_lost_at == 0: bar_lost_at = current_time
        if current_time - bar_lost_at > LOST_TRACK_THRESHOLD: last_bar_cy = None

    # Track that a fishing session is active
    if last_bar_cy is not None and last_fish_cy is not None:
        was_fishing = True
        # Update progress bar tracking
        progress = detect_progress_bar(full_frame, x_start, y_start, x_end, y_end)
        if progress is not None:
            last_known_progress = progress

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
        formula_hold = BASE_HOLD  # default; overwritten by PD branch below

        if fish_visible:
            # Fish is actively detected — velocity-based brake only
            if all_bar_y_coords and bar_v < -SPEED_THRESHOLD:
                status, hold_time, color = "!! UP-BRAKE !!", UP_COUNTER_HOLD, (255, 255, 255)
            elif all_bar_y_coords and bar_v > SPEED_THRESHOLD:
                status, hold_time, color = "!! DOWN-BRAKE !!", DOWN_COUNTER_HOLD, (255, 50, 50)
            else:
                bh = bar_half_height if bar_half_height and bar_half_height > 0 else 50
                # Use actual current diff — higher update rate means we sample fast enough
                # to react to real position rather than needing feed-forward prediction.
                nd = max(-2.0, min(2.0, diff / bh))
                is_boost = abs(nd) > 1.0
                is_centered = abs(nd) < CENTER_ZONE_RATIO
                # D-term: measured bar velocity damps overshoot.
                # bar_v > 0 = falling → add hold; bar_v < 0 = rising → ease off.
                bv_norm = max(-1.0, min(1.0, bar_v / 200.0))
                # PD + gravity bias
                formula_hold = max(MIN_HOLD, min(MAX_HOLD,
                    (BASE_HOLD + GRAVITY_BIAS) - STEP_ADJUST * nd + D_GAIN * bv_norm))
                hold_time, is_learned = get_learned_hold(diff, current_speed_tier, learned_data, formula_hold)
                hold_source = "L" if is_learned else "F"
                if is_centered:
                    status, color = "CENTERED", (0, 255, 100)
                elif nd < 0:
                    status, color = ("BOOST-UP" if is_boost else "ASCENDING"), (0, 255, 255)
                else:
                    status, color = ("BOOST-DOWN" if is_boost else "DESCENDING"), (255, 200, 0)
            prev_diff = diff  # Only update when fish is real to avoid false brake on re-detection
            # Record frame for learning (only when fish is visible and we have speed data)
            if current_fish_cy_raw is not None:
                session_frames.append((diff, hold_time, fish_speed, formula_hold))
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
            caught = last_known_progress >= CAUGHT_THRESHOLD
            if caught:
                fish_caught += 1
            else:
                fish_escaped += 1
            outcome = "CATCH" if caught else "MISS"
            # --- Self-learning: update hold table from this session ---
            if len(session_frames) >= 5:
                process_session_learning(session_frames, last_known_progress, current_speed_tier, learned_data)
                save_learned_data(learned_data, LEARN_FILE)
                tier_d = learned_data["tiers"].get(current_speed_tier, {})
                avg_q = tier_d.get("avg_quality", 0.0)
                tier_sessions = tier_d.get("sessions", 1)
                print(f"[{outcome}] Session #{sessions_total} | Progress: {last_known_progress:.0%} | "
                      f"Rarity={current_fish_rarity} ({current_speed_tier}) | "
                      f"{len(session_frames)} frames | Tier avg quality: {avg_q:.0%} over {tier_sessions}s")
            else:
                print(f"[{outcome}] Session #{sessions_total} | Progress: {last_known_progress:.0%} | "
                      f"Rarity={current_fish_rarity} (insufficient frames, not learned)")
            session_frames = []
            was_fishing = False
            last_known_progress = 0.0
            current_fish_rarity = "unknown"
            _rarity_vote_buffer[:] = []
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
        cv2.putText(full_frame, f"Sessions: {sessions_total}  C:{fish_caught} E:{fish_escaped}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)
        rarity_color = RARITY_OVERLAY_COLORS.get(current_fish_rarity, (200, 200, 200))
        cv2.putText(full_frame, f"{current_fish_rarity} ({current_speed_tier})", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rarity_color, 2)
        tier_strs = [f"{tn[:3].upper()}:{learned_data['tiers'].get(tn, {}).get('sessions', 0)}s"
                     for tn in ("slow", "medium", "fast", "veryfast")
                     if learned_data["tiers"].get(tn, {}).get("sessions", 0) > 0]
        if tier_strs:
            cv2.putText(full_frame, " ".join(tier_strs), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        # Progress bar mini-visualization
        bar_x, bar_y_ov, bar_w_ov, bar_h_ov = w - 30, 30, 16, 100
        cv2.rectangle(full_frame, (bar_x, bar_y_ov), (bar_x + bar_w_ov, bar_y_ov + bar_h_ov), (80, 80, 80), -1)
        fill_h = int(bar_h_ov * last_known_progress)
        if fill_h > 0:
            fill_color = (0, 220, 0) if last_known_progress >= CAUGHT_THRESHOLD else (0, 180, 220)
            cv2.rectangle(full_frame, (bar_x, bar_y_ov + bar_h_ov - fill_h), (bar_x + bar_w_ov, bar_y_ov + bar_h_ov), fill_color, -1)
        cv2.rectangle(full_frame, (bar_x, bar_y_ov), (bar_x + bar_w_ov, bar_y_ov + bar_h_ov), (200, 200, 200), 1)
        cv2.putText(full_frame, f"{last_known_progress:.0%}", (bar_x - 10, bar_y_ov + bar_h_ov + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.imshow("AI Fishing (Full-Integrated)", cv2.resize(full_frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    prev_time = current_time
    time.sleep(max(0, CYCLE_TIME - hold_time) if did_click else DETECTION_CYCLE)

ctypes.windll.winmm.timeEndPeriod(1)
cv2.destroyAllWindows()