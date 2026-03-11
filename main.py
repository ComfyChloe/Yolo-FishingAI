import cv2
import numpy as np
import time
import math
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

# ── Log file (wiped on every startup) ──────────────────────────────────────────
class _Tee:
    """Mirrors all print() output to both stdout and a log file."""
    def __init__(self, path):
        self._log = open(path, 'w', encoding='utf-8', buffering=1)
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)
    def flush(self):
        self._stdout.flush()
        self._log.flush()
    def close(self):
        self._log.close()

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fishing_log.txt')
sys.stdout = _Tee(_LOG_PATH)

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



# ======================
# Self-learning helpers
# ======================
DIFF_BUCKET_SIZE = 20         # px — group diffs into buckets for lookup
LEARN_BASE_RATE = 0.025       # EMA base step (scaled by quality score)
LEARN_COUNT_CAP = 300.0       # Float cap on bucket count (prevents entrenchment)
LEARN_COUNT_DECAY = 0.997     # Per-session multiplier on all counts (~50% after 231 sessions)
LEARN_SCHEMA_VERSION = 3      # v3: offset-based learning for MPC
HUE_BOUNDARY_MARGIN = 3       # Treat near-boundary hues as unknown to avoid false locks

def _default_tier():
    return {"sessions": 0, "avg_quality": 0.0, "hold_table": {}}

def _default_learned_data():
    return {
        "version": LEARN_SCHEMA_VERSION,
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
        if data.get("version") != LEARN_SCHEMA_VERSION:
            print(f"[LEARN] Schema version mismatch (got v{data.get('version')}, need v{LEARN_SCHEMA_VERSION}) — resetting (old data discarded)")
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
    "relic":    "slow",
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
    usable = (hsv[:, 2] > 45) & (hsv[:, 2] < 235)
    if usable.sum() < 12:
        return None
    low_sat = usable & (hsv[:, 1] < 60)
    # Gray fish can have colored edge bleed; if the box is mostly low saturation,
    # trust the grayscale core instead of a smaller colored fringe.
    if low_sat.sum() >= max(12, int(usable.sum() * 0.55)):
        return np.median(hsv[low_sat], axis=0).astype(int)
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
    if h < 14 or h >= 165: return "brown/dark-red" if v < 170 else "red/orange-red"
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
    # Avoid unstable decisions near major class boundaries where anti-aliasing/noise flips rarity.
    # Do not guard around 14 (relic/fabled) — both sides confirmed by live data.
    # Do not guard around 142 because purple/violet (H>=142) is a validated exotic band.
    boundaries = (88, 132, 165)
    for b in boundaries:
        if abs(h - b) <= HUE_BOUNDARY_MARGIN:
            return "unknown", False
    # Confirmed mappings (live sampling + hex codes):
    #   unknown = true red  H=0-10, S>=140 (fast/aggressive, unidentified rarity)
    #   relic   = orange-brown H=11-13, S~180 (slow, confirmed)
    #   fabled  = gold/yellow  H ~14-40 (confirmed)
    #   common  = bright green H ~40-88 (confirmed)
    #   curious = blue         H=111 confirmed
    #   exotic  = purple       H=147 confirmed
    # Unconfirmed: elusive, mythic — will print [RARITY] unknown to calibrate
    if h < 11 and s >= 140:    return "unknown", False  # bright red — unidentified fast fish
    if h < 14 or h >= 165:     return "relic",   True   # orange-brown (confirmed H=11-13)
    if 14 <= h < 40:           return "fabled",  True   # gold/yellow (confirmed)
    if 40 <= h < 88:       return "common",  True   # bright green (confirmed)
    if 88 <= h < 132:      return "curious", True   # blue (confirmed at H=111)
    if 132 <= h < 142:     return "mythic",  True   # blue-purple (placeholder)
    if 142 <= h < 165:     return "exotic",  True   # purple-violet (confirmed, includes H=142 edge)
    return "unknown", False  # unrecognised

def get_diff_bucket(diff):
    return round(diff / DIFF_BUCKET_SIZE)

def get_learned_hold(diff, speed_tier, learned_data, mpc_hold):
    """Apply learned offset correction to MPC hold. Returns (hold, is_learned).
    v3 schema: hold_table stores {"offset": float, "count": float} per bucket."""
    bucket = str(get_diff_bucket(diff))
    table = learned_data["tiers"].get(speed_tier, {}).get("hold_table", {})
    entry = table.get(bucket)
    if entry is None:
        return mpc_hold, False
    learned_offset = entry.get("offset", 0.0)
    count = float(entry.get("count", 0))
    # Continuous alpha: reaches ~0.75 at count=100, ~0.90 at count=300
    alpha = min(0.92, count / (count + 33.0)) if count > 0 else 0.0
    corrected = mpc_hold + alpha * learned_offset
    return max(MIN_HOLD, min(MAX_HOLD, corrected)), True

def process_session_learning(session_frames, speed_tier, learned_data):
    """Update learned offset table from session frame data. Modifies learned_data in-place.
    v3: learns residual offsets (hold_used - mpc_hold) instead of absolute hold times."""
    tier = learned_data["tiers"].setdefault(speed_tier, _default_tier())
    tier["sessions"] += 1
    learned_data["total_sessions"] += 1
    rate = LEARN_BASE_RATE
    table = tier.setdefault("hold_table", {})
    # --- Step 1: aggregate session frames by bucket (residual = hold_used - mpc_hold) ---
    bucket_data = {}  # bucket -> {residual_sum, speed_sum, n}
    for diff_i, hold_i, speed_i, mpc_i in session_frames:
        b = str(get_diff_bucket(diff_i))
        if b not in bucket_data:
            bucket_data[b] = {"residual_sum": 0.0, "speed_sum": 0.0, "n": 0}
        bucket_data[b]["residual_sum"] += (hold_i - mpc_i)
        bucket_data[b]["speed_sum"] += speed_i
        bucket_data[b]["n"] += 1
    # Tier-average fish speed for erratic-frame detection
    all_speeds = [s for _, _, s, _ in session_frames]
    tier_avg_speed = (sum(all_speeds) / len(all_speeds)) if all_speeds else 1.0
    # --- Step 2: apply count decay to ALL existing buckets ---
    for entry in table.values():
        entry["count"] = max(0.0, entry["count"] * LEARN_COUNT_DECAY)
    # --- Step 3: EMA update for visited buckets (offset-based) ---
    for b, bd in bucket_data.items():
        avg_residual = bd["residual_sum"] / bd["n"]
        avg_speed = bd["speed_sum"] / bd["n"]
        bucket_rate = rate * (0.5 if tier_avg_speed > 0 and avg_speed > 2.0 * tier_avg_speed else 1.0)
        entry = table.get(b, {"offset": 0.0, "count": 0.0})
        old_offset = entry.get("offset", 0.0)
        entry["offset"] = old_offset + bucket_rate * (avg_residual - old_offset)
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
CYCLE_TIME = 0.055          # Click cycle interval (~18Hz control rate)
DETECTION_CYCLE = 0.005    # Detection wait interval when not clicking
SMOOTH_FACTOR = 0.5        # Coordinate smoothing factor

# Hold timing — MPC computes hold dynamically; these are safety clamps & fallback
MAX_HOLD = 0.08
MIN_HOLD = 0.02
UP_COUNTER_HOLD = 0.02    # Emergency brake: bar rising too fast
DOWN_COUNTER_HOLD = 0.08  # Emergency brake: bar falling too fast
JITTER_PIXELS = 4          # Small mouse jitter amplitude (px)

# PD fallback constants — used before MPC calibration converges
PD_BASE_HOLD = 0.035       # Neutral hold for PD fallback
PD_GRAVITY_BIAS = 0.005
PD_STEP_ADJUST = 0.06      # Kp
PD_D_GAIN = 0.008          # Kd

# MPC constants — game physics: gravity=1.25, playerSpeed=3.75, ratio=3.0
PHYSICS_SPEED_GRAVITY_RATIO = 3.0   # playerSpeed / gravity (from UdonSharp)
MPC_MIN_CALIBRATION = 20            # Min samples before MPC replaces PD fallback
MPC_FISH_PREDICT = 0.28             # Shorter prediction horizon reduces overshoot on target flips
MPC_CALIB_EMA = 0.05                # EMA rate for gravity estimation
MPC_CALIB_DENOM_MIN = 0.005         # Min denominator to accept calibration sample
MPC_CALIB_OUTLIER = 5.0             # Reject sample if > N× current estimate
APPROACH_DAMP_SPEED = 40.0          # px/s threshold for damping when already converging
APPROACH_DAMP_ZONE = 0.90           # Apply damping when |diff| < zone*bar_half_height
APPROACH_EQUIL_BLEND = 0.65         # Blend toward equilibrium hold while converging
HOLD_SLEW_MAX = 0.018               # Max hold-time change per cycle (seconds)
FAST_PREDICT_SCALE = 0.70           # Reduce prediction on fast/veryfast fish
FAST_DAMP_SPEED_SCALE = 0.70        # Start damping earlier on fast/veryfast fish
FAST_EQUIL_BLEND = 0.78             # Stronger equilibrium blend for jumpy fish
FAST_SLEW_MAX = 0.014               # Tighter slew limit for fast/veryfast fish


# Fight phase timing
GRACE_DURATION = 1.0       # 0-1s: zero escape penalty (game fact)
RAMP_END = 5.0             # 1-5s: penalty ramps linearly
CATCH_ZONE_FRACTION = 0.35 # Conservative overlap estimate (game: 0.4-0.65 of bar)

# Edge/bounce handling
EDGE_MARGIN_RATIO = 0.10   # % of bar range considered "near wall"

# Thresholds
SPEED_THRESHOLD = 120      # Emergency brake speed threshold (px/s) — raised for MPC

FISH_CONF_MIN = 0.25       # Minimum confidence to accept a 'fish icon' detection
RECAST_INTERVAL = float(config.get("recast_interval", DEFAULT_CONFIG["recast_interval"]))  # Seconds before recast
LOST_TRACK_THRESHOLD = 1.5 # Grace period for keeping last coordinates

def compute_mpc_hold(bar_pos, bar_vel, fish_pos, cycle_time, pg, ps):
    """Compute optimal hold time to move bar to fish position in one cycle.
    All values in pixel coordinates (y increases downward).
    Game physics in pixel-space:
      Gravity pulls bar DOWN (+y): accel = +pg
      Clicking pushes bar UP (-y): accel = -ps
    One-cycle prediction:
      p1 = p0 + v0*T + 0.5*pg*T^2 - ps*Th*(T - 0.5*Th)
    bar_pos/fish_pos in pixels, bar_vel in px/s, pg/ps in px/s^2 (positive)."""
    T = cycle_time
    # Position the bar would reach with zero click (gravity only, bar falls down)
    p_freefall = bar_pos + bar_vel * T + 0.5 * pg * T * T
    # How much upward displacement we need from clicking (negative = need to go up)
    need = fish_pos - p_freefall  # negative = fish is above freefall → need click
    # Full equation: p1 = p_freefall - ps*Th*(T - 0.5*Th) = fish_pos
    # → ps*Th*(T - 0.5*Th) = -need = p_freefall - fish_pos
    # → 0.5*ps*Th^2 - ps*T*Th + (p_freefall - fish_pos) = 0
    c = p_freefall - fish_pos  # positive = freefall overshoots downward, need more click
    # Quadratic: 0.5*ps*Th^2 - ps*T*Th + c = 0
    disc = ps * ps * T * T - 2.0 * ps * c
    if disc < 0:
        # Can't reach target in one cycle — clamp to boundary
        return MAX_HOLD if c > 0 else MIN_HOLD
    sqrt_disc = math.sqrt(disc)
    th1 = T - sqrt_disc / ps
    th2 = T + sqrt_disc / ps
    # Pick the solution in [0, T]; prefer th1 (shorter click = less aggressive)
    if 0.0 <= th1 <= T:
        th = th1
    elif 0.0 <= th2 <= T:
        th = th2
    else:
        th = max(0.0, min(T, th1))
    return max(MIN_HOLD, min(MAX_HOLD, th))

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
hold_source = "F"               # "L" = learned, "F" = formula/MPC (for overlay)

bar_half_height = None     # Half the white bar height (px), smoothed
# MPC calibration state
est_pixel_gravity = None   # Estimated gravity in px/s^2 (auto-calibrated)
est_pixel_player_speed = None  # = 3.0 * est_pixel_gravity
calibration_count = 0      # Number of accepted calibration samples
prev_hold_time = 0.0       # Hold time used in previous cycle (for calibration)
prev_bar_v = 0.0           # Bar velocity from previous cycle
last_applied_hold = PD_BASE_HOLD  # Previous commanded hold (for slew limiting)
fight_start_time = 0.0     # Timestamp when current fight began
# Edge tracking
session_bar_min = None     # Min bar_cy observed this fight (proxy for top wall)
session_bar_max = None     # Max bar_cy observed this fight (proxy for bottom wall)
bounce_skip = False        # Skip calibration sample after detected bounce
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
    global prev_fish_cy, session_frames, bar_half_height
    global current_fish_rarity, current_speed_tier, _rarity_vote_buffer
    global est_pixel_gravity, est_pixel_player_speed, calibration_count
    global prev_hold_time, prev_bar_v, last_applied_hold, fight_start_time
    global session_bar_min, session_bar_max, bounce_skip
    locked_hwnd = None
    last_detected_time = time.time()
    last_fish_cy = last_bar_cy = prev_bar_cy = None
    fish_lost_at = bar_lost_at = prev_diff = 0
    jitter_done = False
    prev_fish_cy = None
    session_frames = []
    bar_half_height = None
    current_fish_rarity = "unknown"
    current_speed_tier = "medium"
    _rarity_vote_buffer = []
    est_pixel_gravity = None
    est_pixel_player_speed = None
    calibration_count = 0
    prev_hold_time = 0.0
    prev_bar_v = 0.0
    last_applied_hold = PD_BASE_HOLD
    fight_start_time = 0.0
    session_bar_min = None
    session_bar_max = None
    bounce_skip = False
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
                    if DEBUG_COLOR_LOG and current_fish_rarity == "unknown":
                        print(f"[COLOR] {rarity} | {_cname} (H={hsv_med[0]} S={hsv_med[1]} V={hsv_med[2]})")
                    # Multi-frame voting: lock rarity once 2 of last 3 samples agree
                    if current_fish_rarity == "unknown":
                        if known:
                            _rarity_vote_buffer.append(rarity)
                            if len(_rarity_vote_buffer) > 7:
                                _rarity_vote_buffer.pop(0)
                            recent = _rarity_vote_buffer[-5:]
                            counts = {}
                            for r in recent:
                                counts[r] = counts.get(r, 0) + 1
                            sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                            top_label, top_count = sorted_counts[0]
                            second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
                            # Stronger lock: 4-of-5 majority with margin over runner-up.
                            if len(recent) >= 5 and top_count >= 4 and (top_count - second_count) >= 2:
                                current_fish_rarity = top_label
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
        if not was_fishing:
            fight_start_time = current_time  # Mark start of new fight
            session_bar_min = last_bar_cy
            session_bar_max = last_bar_cy
        was_fishing = True
        # Update edge tracking for bounce detection
        if session_bar_min is None or last_bar_cy < session_bar_min:
            session_bar_min = last_bar_cy
        if session_bar_max is None or last_bar_cy > session_bar_max:
            session_bar_max = last_bar_cy


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
        fish_vel_signed = 0.0  # Signed velocity for MPC fish prediction
        if prev_fish_cy is not None and dt > 0:
            fish_speed = abs(current_fish_cy_raw - prev_fish_cy) / dt
            fish_vel_signed = (current_fish_cy_raw - prev_fish_cy) / dt
        prev_fish_cy = current_fish_cy_raw
        last_fish_cy, last_detected_time, fish_lost_at = current_fish_cy_raw, current_time, 0
    else:
        prev_fish_cy = None  # Reset speed tracking when fish lost
        if fish_lost_at == 0: fish_lost_at = current_time
        if current_time - fish_lost_at > LOST_TRACK_THRESHOLD:
            last_fish_cy = None
            jitter_done = False  # Allow jitter again on next fish

    # --- Physics auto-calibration (Phase 2) ---
    # Estimate pixel-space gravity from observed bar acceleration.
    # In pixel coords (y↓): delta_v = g_px * (T - 3*Th)  [since s_px = 3*g_px]
    # Solving: g_px = delta_v / (T - 3*Th)
    if (all_bar_y_coords and was_fishing and prev_hold_time > 0
            and dt > 0 and not bounce_skip):
        delta_v = bar_v - prev_bar_v
        calib_denom = dt - PHYSICS_SPEED_GRAVITY_RATIO * prev_hold_time
        if abs(calib_denom) > MPC_CALIB_DENOM_MIN:
            sample = delta_v / calib_denom
            if sample > 0:  # gravity must be positive (downward in pixels)
                if est_pixel_gravity is None:
                    est_pixel_gravity = sample
                    est_pixel_player_speed = PHYSICS_SPEED_GRAVITY_RATIO * sample
                    calibration_count = 1
                elif abs(sample) < MPC_CALIB_OUTLIER * est_pixel_gravity:
                    est_pixel_gravity += MPC_CALIB_EMA * (sample - est_pixel_gravity)
                    est_pixel_player_speed = PHYSICS_SPEED_GRAVITY_RATIO * est_pixel_gravity
                    calibration_count += 1
    bounce_skip = False  # Reset after one skipped sample

    # --- Bounce detection (Phase 5) ---
    # Bar velocity sign flip with ~70% magnitude drop = wall bounce
    # Must check BEFORE updating prev_bar_v so we compare old vs new velocity
    if (all_bar_y_coords and prev_bar_v != 0 and bar_v != 0
            and (prev_bar_v > 0) != (bar_v > 0)
            and abs(bar_v) < 0.5 * abs(prev_bar_v)):
        bounce_skip = True  # Skip next calibration sample (bounce corrupts it)

    prev_bar_v = bar_v
    prev_hold_time = hold_time  # Will be used next cycle for calibration

    # Determine fight phase
    fight_elapsed = current_time - fight_start_time if was_fishing else 0.0
    if fight_elapsed < GRACE_DURATION:
        fight_phase = "GRACE"
    elif fight_elapsed < RAMP_END:
        fight_phase = "RAMP"
    else:
        fight_phase = "CRITICAL"

    # Use MPC if calibrated, else PD fallback
    mpc_ready = calibration_count >= MPC_MIN_CALIBRATION and est_pixel_gravity is not None

    # Control logic.
    if last_bar_cy is not None and last_fish_cy is not None:
        diff = last_fish_cy - last_bar_cy
        fish_visible = current_fish_cy_raw is not None
        bh = bar_half_height if bar_half_height and bar_half_height > 0 else 50
        # Equilibrium hold (duty cycle = 1/3 of cycle) — used for centered/grace
        equil_hold = max(MIN_HOLD, min(MAX_HOLD, CYCLE_TIME / PHYSICS_SPEED_GRAVITY_RATIO))
        mpc_hold = equil_hold  # default; overwritten by MPC/PD below
        hold_source = "F"  # default source label

        if fish_visible:
            # --- Emergency brakes (extreme velocity) ---
            if all_bar_y_coords and bar_v < -SPEED_THRESHOLD:
                status, hold_time, color = "!! UP-BRAKE !!", UP_COUNTER_HOLD, (255, 255, 255)
            elif all_bar_y_coords and bar_v > SPEED_THRESHOLD:
                status, hold_time, color = "!! DOWN-BRAKE !!", DOWN_COUNTER_HOLD, (255, 50, 50)
            else:
                # --- Edge override (Phase 5) ---
                near_edge_override = False
                if session_bar_min is not None and session_bar_max is not None:
                    bar_range = session_bar_max - session_bar_min
                    if bar_range > 20:  # Need meaningful range
                        edge_margin = bar_range * EDGE_MARGIN_RATIO
                        if last_bar_cy < session_bar_min + edge_margin and bar_v < -20:
                            # Near top wall, moving up — let gravity pull down
                            status, hold_time, color = "EDGE-TOP", MIN_HOLD, (200, 200, 255)
                            near_edge_override = True
                        elif last_bar_cy > session_bar_max - edge_margin and bar_v > 20:
                            # Near bottom wall, moving down — push up hard
                            status, hold_time, color = "EDGE-BOT", MAX_HOLD, (255, 200, 200)
                            near_edge_override = True

                if not near_edge_override:
                    nd = max(-2.0, min(2.0, diff / bh))
                    is_boost = abs(nd) > 1.35
                    # Catch zone check (generous overlap from game physics)
                    catch_zone_half = bh * CATCH_ZONE_FRACTION
                    in_catch_zone = abs(diff) < catch_zone_half

                    if mpc_ready:
                        # --- MPC controller (Phase 3) ---
                        # Predict fish position half-cycle ahead using measured velocity
                        predict_frac = MPC_FISH_PREDICT
                        if current_speed_tier in ("fast", "veryfast"):
                            predict_frac *= FAST_PREDICT_SCALE
                        if fight_phase == "CRITICAL":
                            predict_frac *= 0.5  # Conservative in critical phase
                        fish_target = last_fish_cy + fish_vel_signed * CYCLE_TIME * predict_frac
                        mpc_hold = compute_mpc_hold(
                            last_bar_cy, bar_v, fish_target,
                            CYCLE_TIME, est_pixel_gravity, est_pixel_player_speed)
                        hold_source = "M"  # MPC
                    else:
                        # --- PD fallback (pre-calibration) ---
                        bv_norm = max(-1.0, min(1.0, bar_v / 200.0))
                        mpc_hold = max(MIN_HOLD, min(MAX_HOLD,
                            (PD_BASE_HOLD + PD_GRAVITY_BIAS) - PD_STEP_ADJUST * nd + PD_D_GAIN * bv_norm))
                        hold_source = "P"  # PD fallback

                    # Grace phase: relax if already in catch zone
                    if fight_phase == "GRACE" and in_catch_zone:
                        hold_time = equil_hold
                        hold_source = "G"  # Grace
                    else:
                        hold_time, is_learned = get_learned_hold(
                            diff, current_speed_tier, learned_data, mpc_hold)
                        if is_learned:
                            hold_source = "L"  # Learned offset
                    # Anti-overshoot damping: if bar is already moving toward fish quickly,
                    # blend toward equilibrium rather than keep pushing hard.
                    damp_speed = APPROACH_DAMP_SPEED
                    damp_blend = APPROACH_EQUIL_BLEND
                    slew_limit = HOLD_SLEW_MAX
                    if current_speed_tier in ("fast", "veryfast"):
                        damp_speed *= FAST_DAMP_SPEED_SCALE
                        damp_blend = FAST_EQUIL_BLEND
                        slew_limit = FAST_SLEW_MAX
                    moving_toward_fish = (diff * bar_v) > 0
                    in_damp_zone = abs(diff) < (APPROACH_DAMP_ZONE * bh)
                    if moving_toward_fish and in_damp_zone and abs(bar_v) > damp_speed:
                        hold_time = (1.0 - damp_blend) * hold_time + damp_blend * equil_hold
                        hold_source = hold_source + "D"

                    # Slew limiter: prevent big hold jumps that cause bar slingshot.
                    hold_delta = hold_time - last_applied_hold
                    if abs(hold_delta) > slew_limit:
                        hold_time = last_applied_hold + (slew_limit if hold_delta > 0 else -slew_limit)
                        hold_source = hold_source + "S"

                    if in_catch_zone:
                        status, color = "CENTERED", (0, 255, 100)
                    elif nd < 0:
                        status, color = ("BOOST-UP" if is_boost else "ASCENDING"), (0, 255, 255)
                    else:
                        status, color = ("BOOST-DOWN" if is_boost else "DESCENDING"), (255, 200, 0)

            prev_diff = diff
            # Record frame for learning (residual = hold_used - mpc_hold)
            if current_fish_cy_raw is not None:
                session_frames.append((diff, hold_time, fish_speed, mpc_hold))
        else:
            # Fish is hidden (inside bar) — use equilibrium hold
            if diff < 0:
                status, hold_time, color = "HOLDING-UP", equil_hold, (0, 180, 180)
            else:
                status, hold_time, color = "HOLDING-DOWN", equil_hold, (180, 140, 0)

        if not all_bar_y_coords: status += " (BAR LOST)"
        # Append fight phase + elapsed time
        phase_tag = f" {fight_phase} {fight_elapsed:.1f}s" if was_fishing else ""

        if input_enabled:
            lParam = win32api.MAKELONG(w // 2, h // 2)
            win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
            time.sleep(hold_time)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
            did_click = True
        else:
            time.sleep(hold_time)
        last_applied_hold = hold_time
        
        calib_tag = f" g={est_pixel_gravity:.0f}" if est_pixel_gravity else " g=?"
        debug_text = f"{status} | HOLD: {hold_time:.3f} [{hold_source}]{calib_tag}{phase_tag}"
        if OVERLAY_ENABLED:
            cv2.line(full_frame, (15, int(last_bar_cy)), (15, int(last_fish_cy)), color, 2)

    # Recovery logic (recast).
    elif current_time - last_detected_time > RECAST_INTERVAL:
        if was_fishing:
            sessions_total += 1
            # --- Self-learning: update hold table from this session ---
            if len(session_frames) >= 5:
                process_session_learning(session_frames, current_speed_tier, learned_data)
                save_learned_data(learned_data, LEARN_FILE)
                tier_d = learned_data["tiers"].get(current_speed_tier, {})
                tier_sessions = tier_d.get("sessions", 1)
                print(f"[SESSION] #{sessions_total} | Rarity={current_fish_rarity} ({current_speed_tier}) | "
                      f"{len(session_frames)} frames | Tier sessions: {tier_sessions}")
            else:
                print(f"[SESSION] #{sessions_total} | Rarity={current_fish_rarity} (insufficient frames, not learned)")
            session_frames = []
            was_fishing = False
            current_fish_rarity = "unknown"
            _rarity_vote_buffer[:] = []
            session_bar_min = None
            session_bar_max = None
            fight_start_time = 0.0
            last_applied_hold = PD_BASE_HOLD
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
        rarity_color = RARITY_OVERLAY_COLORS.get(current_fish_rarity, (200, 200, 200))
        cv2.putText(full_frame, f"{current_fish_rarity} ({current_speed_tier})", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rarity_color, 2)
        tier_strs = [f"{tn[:3].upper()}:{learned_data['tiers'].get(tn, {}).get('sessions', 0)}s"
                     for tn in ("slow", "medium", "fast", "veryfast")
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