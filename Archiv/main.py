import json
import re
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List
from pathlib import Path
import os
import ctypes

import numpy as np
import pyautogui
import pytesseract
from mss import mss
from PIL import Image, ImageOps, ImageEnhance, ImageTk, ImageDraw, ImageFont

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import sys
from pathlib import Path

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base) / rel)
    return str(Path(__file__).parent / rel)

def auto_set_tesseract_path(cfg):
    # If user did not specify a custom path, use bundled tesseract
    if not cfg.tesseract_cmd.strip():
        cfg.tesseract_cmd = resource_path(r"tesseract\tesseract.exe")

CONFIG_DEFAULT = "ocr_symbol_gui_config.json"
APP_TITLE = "Symbol extractor ATLAS)"
# Windows taskbar identity (helps ensure the correct icon shows in the taskbar)
APP_USER_MODEL_ID = "ocr_symbol_bot.OCRSymbolBot"


# -----------------------------
# Tooltips (mouse-over help)
# -----------------------------
class ToolTip:
    """
    Simple tooltip for Tkinter widgets.
    Shows a small hover window near the cursor with help text.
    """
    def __init__(self, widget, text: str, delay_ms: int = 400, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength

        self._after_id = None
        self._tip = None

        self.widget.bind("<Enter>", self._on_enter, add=True)
        self.widget.bind("<Leave>", self._on_leave, add=True)
        self.widget.bind("<ButtonPress>", self._on_leave, add=True)

    def _on_enter(self, _evt=None):
        self._schedule()

    def _on_leave(self, _evt=None):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _unschedule(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        if not self.text:
            return

        try:
            x = self.widget.winfo_pointerx() + 12
            y = self.widget.winfo_pointery() + 14
        except Exception:
            x, y = 0, 0

        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_attributes("-topmost", True)

        frame = ttk.Frame(self._tip, padding=(8, 6))
        frame.pack(fill="both", expand=True)

        label = ttk.Label(frame, text=self.text, justify="left", wraplength=self.wraplength)
        label.pack(anchor="w")

        self._tip.wm_geometry(f"+{x}+{y}")

    def _hide(self):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
        self._tip = None


# -----------------------------
# Windows DPI / App identity
# -----------------------------
def enable_dpi_awareness():
    """
    Windows-specific: prevents coordinate mismatches between Tk (logical pixels)
    and MSS (physical pixels). Must be called before creating a Tk window.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI awareness
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # System DPI awareness fallback
    except Exception:
        pass


def set_windows_appusermodelid(appid: str):
    """
    Sets an explicit AppUserModelID on Windows so the taskbar groups the app correctly
    and uses the intended taskbar icon.
    """
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
    except Exception:
        pass


# -----------------------------
# App data paths
# -----------------------------
def _app_state_dir() -> Path:
    p = Path.home() / ".ocr_symbol_bot"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_config_path() -> Path:
    return _app_state_dir() / CONFIG_DEFAULT


def _default_screenshot_path() -> Path:
    return _app_state_dir() / "last_screenshot.png"


def _default_icon_png_path() -> Path:
    return _app_state_dir() / "app_icon.png"


def _default_icon_ico_path() -> Path:
    return _app_state_dir() / "app_icon.ico"


def _create_app_icon(png_path: Path, ico_path: Path) -> None:
    """
    Generates a clean app icon (magnifier + OCR) as PNG and multi-size ICO.
    """
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Background: two-tone diagonal + subtle accent
    bg1 = (24, 32, 52, 255)
    bg2 = (18, 86, 123, 255)
    accent = (0, 198, 255, 60)

    d.rectangle([0, 0, size, size], fill=bg1)
    d.polygon([(0, 0), (size, 0), (0, size)], fill=bg2)
    d.ellipse([size * 0.15, size * 0.10, size * 0.95, size * 0.90], fill=accent)

    # Rounded border
    border = (255, 255, 255, 40)
    margin = 10
    d.rounded_rectangle([margin, margin, size - margin, size - margin],
                        radius=34, outline=border, width=2)

    # Magnifier
    stroke = (240, 248, 255, 235)
    stroke2 = (0, 0, 0, 70)
    cx, cy = int(size * 0.43), int(size * 0.43)
    r = int(size * 0.20)
    w = 14

    # Shadow + ring
    d.ellipse([cx - r + 4, cy - r + 6, cx + r + 4, cy + r + 6], outline=stroke2, width=w)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=stroke, width=w)

    # Handle
    hx1, hy1 = int(cx + r * 0.62), int(cy + r * 0.62)
    hx2, hy2 = int(size * 0.78), int(size * 0.78)
    d.line([(hx1 + 4, hy1 + 6), (hx2 + 4, hy2 + 6)], fill=stroke2, width=w + 2)
    d.line([(hx1, hy1), (hx2, hy2)], fill=stroke, width=w)

    # Scan lines
    scan = (255, 255, 255, 70)
    d.line([(cx - r // 2, cy - r // 6), (cx + r // 2, cy - r // 6)], fill=scan, width=6)
    d.line([(cx - r // 2, cy + r // 6), (cx + r // 3, cy + r // 6)], fill=scan, width=6)

    # Text "OCR"
    try:
        font = ImageFont.truetype("segoeui.ttf", 44)
    except Exception:
        font = ImageFont.load_default()

    text = "OCR"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = int(size * 0.12)
    ty = int(size * 0.70)
    d.text((tx + 2, ty + 3), text, font=font, fill=(0, 0, 0, 90))
    d.text((tx, ty), text, font=font, fill=(255, 255, 255, 235))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path, format="PNG")

    img.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )


def ensure_app_icon() -> Tuple[Path, Path]:
    """
    Ensures icon files exist. Returns (ico_path, png_path).
    """
    ico_path = _default_icon_ico_path()
    png_path = _default_icon_png_path()

    if not ico_path.exists() or not png_path.exists():
        try:
            _create_app_icon(png_path=png_path, ico_path=ico_path)
        except Exception:
            pass

    return ico_path, png_path


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Config:
    monitor_index: int = 0
    roi: Roi = field(default_factory=lambda: Roi(0, 0, 300, 80))
    click_abs_x: int = 0
    click_abs_y: int = 0
    poll_seconds: float = 0.75
    type_interval: float = 0.03
    require_change: bool = True
    use_regex: bool = True
    regex_pattern: str = (
        r"\bSTOCK\s*[:\-]\s*([A-Z0-9][A-Z0-9\.\-]{0,9})"
        r"(?=\s*(?:DATE|TIME|STATUS)\b|$)"
    )
    psm: int = 6
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.- "
    tesseract_cmd: str = ""


# -----------------------------
# OCR preprocessing (no OpenCV)
# -----------------------------
def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    if total == 0:
        return 128

    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, var_max, threshold = 0.0, 0.0, 0.0, 128

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return int(threshold)


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    scale = 3
    w, h = pil_img.size
    pil_img = pil_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

    pil_img = ImageOps.grayscale(pil_img)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.5)

    gray = np.array(pil_img, dtype=np.uint8)
    thr = otsu_threshold(gray)
    bw = (gray > thr).astype(np.uint8) * 255
    return Image.fromarray(bw)


def normalize_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", "\n", t).strip()
    up = t.upper()

    up = up.replace("5TOCK", "STOCK")
    up = up.replace("ST0CK", "STOCK")
    up = up.replace("STOGK", "STOCK")

    up = re.sub(r"(?<=\w)(DATE|TIME|STATUS)\b", r" \1", up)
    return up


def extract_symbol(ocr_text: str, use_regex: bool, pattern: str) -> Optional[str]:
    t = normalize_text(ocr_text)

    if use_regex and pattern.strip():
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return None

        m = rx.search(t)
        if m:
            sym = m.group(1) if (m.lastindex and m.lastindex >= 1) else m.group(0)
            sym = re.sub(r"\s+", "", sym.strip().upper())
            return sym if sym else None
        return None

    tokens = re.findall(r"[A-Z0-9][A-Z0-9\.\-]{0,9}", t)
    if not tokens:
        return None
    tokens.sort(key=len, reverse=True)
    return tokens[0]


# -----------------------------
# Screen capture helpers
# -----------------------------
def list_monitors() -> List[dict]:
    with mss() as sct:
        return list(sct.monitors)


def grab_monitor_image_and_info(monitor_index: int) -> Tuple[Image.Image, dict]:
    with mss() as sct:
        mons = sct.monitors
        if monitor_index < 0 or monitor_index >= len(mons):
            raise ValueError(f"Invalid monitor index {monitor_index}. Available: 0..{len(mons)-1}")
        m = mons[monitor_index]
        shot = sct.grab(m)  # BGRA
        arr = np.array(shot, dtype=np.uint8)
        rgb = arr[:, :, :3][:, :, ::-1]  # BGR->RGB
        return Image.fromarray(rgb), m


def grab_roi_text(cfg: Config, mon_info: dict) -> Tuple[Optional[str], str, Image.Image]:
    if cfg.tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd.strip()

    with mss() as sct:
        x, y, w, h = cfg.roi.x, cfg.roi.y, cfg.roi.w, cfg.roi.h
        if w <= 0 or h <= 0:
            raise ValueError("ROI is invalid (width/height <= 0).")

        bbox = {
            "left": mon_info["left"] + x,
            "top": mon_info["top"] + y,
            "width": w,
            "height": h
        }
        shot = sct.grab(bbox)
        arr = np.array(shot, dtype=np.uint8)
        rgb = arr[:, :, :3][:, :, ::-1]
        pil = Image.fromarray(rgb)

    prep = preprocess_for_ocr(pil)
    tcfg = f"--psm {cfg.psm} -c tessedit_char_whitelist={cfg.whitelist}"
    raw = pytesseract.image_to_string(prep, config=tcfg)

    sym = extract_symbol(raw, cfg.use_regex, cfg.regex_pattern)
    return sym, raw, prep


def type_symbol_into_target(cfg: Config, symbol: str) -> None:
    pyautogui.click(cfg.click_abs_x, cfg.click_abs_y)
    time.sleep(0.05)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.02)
    pyautogui.typewrite(symbol, interval=cfg.type_interval)
    pyautogui.press("enter")


# -----------------------------
# GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self._apply_app_icon()

        self.title(APP_TITLE)
        self.geometry("1200x780")

        pyautogui.FAILSAFE = True

        self.cfg = Config()
        auto_set_tesseract_path(self.cfg)
        self.monitors = list_monitors()

        self.mon_info: Optional[dict] = None
        self.screen_img: Optional[Image.Image] = None
        self.screen_tk: Optional[ImageTk.PhotoImage] = None
        self.zoom: float = 1.0

        self.mode: str = "idle"
        self.roi_p1: Optional[Tuple[int, int]] = None
        self.roi_p2: Optional[Tuple[int, int]] = None
        self.point_xy: Optional[Tuple[int, int]] = None

        self.canvas_img_id = None
        self.rect_id = None
        self.cross_ids = []

        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._last_sent: Optional[str] = None

        self._build_ui()
        self._refresh_monitor_dropdown()

        self._load_startup_state()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_app_icon(self):
        """
        Sets window and taskbar icon.
        - Windows: iconbitmap(.ico) is the key for taskbar icon.
        - Fallback: iconphoto(.png).
        """
        ico_path, png_path = ensure_app_icon()

        try:
            if os.name == "nt" and ico_path.exists():
                self.iconbitmap(default=str(ico_path))
        except Exception:
            pass

        try:
            if png_path.exists():
                pil = Image.open(png_path).convert("RGBA")
                pil_small = pil.copy()
                pil_small.thumbnail((64, 64))
                tk_img = ImageTk.PhotoImage(pil_small)
                self.iconphoto(True, tk_img)
                self._tk_app_icon_ref = tk_img  # keep reference
        except Exception:
            pass

    # ---------- persistence ----------
    def _load_startup_state(self):
        cfg_path = _default_config_path()
        if cfg_path.exists():
            ok = self._load_config_from_path(cfg_path, silent=True)
            if ok:
                self.status_var.set(f"Config auto-loaded: {cfg_path}")
        else:
            self.status_var.set("Ready. Capture a screenshot.")

        shot_path = _default_screenshot_path()
        if shot_path.exists():
            self._load_screenshot_cache(shot_path)

        if 0 <= self.cfg.monitor_index < len(self.monitors):
            self.monitor_combo.current(self.cfg.monitor_index)
        else:
            self.cfg.monitor_index = 0
            self.monitor_combo.current(0)

        self._apply_cfg_to_ui()
        self._sync_visual_points_from_cfg()
        self.redraw_selections()

    def _apply_cfg_to_ui(self):
        self.roi_label.config(text=self._roi_text())
        self.click_label.config(text=f"{self.cfg.click_abs_x}, {self.cfg.click_abs_y}")

        self.poll_var.set(self.cfg.poll_seconds)
        self.type_var.set(self.cfg.type_interval)
        self.change_var.set(self.cfg.require_change)

        self.use_regex_var.set(self.cfg.use_regex)
        self.regex_var.set(self.cfg.regex_pattern)
        self.psm_combo.set(str(self.cfg.psm))
        self.whitelist_var.set(self.cfg.whitelist)
        self.tess_var.set(self.cfg.tesseract_cmd)

    def _save_default_config_silent(self):
        self.sync_cfg_from_ui(require_screenshot=False)
        try:
            self._save_config_to_path(_default_config_path())
        except Exception:
            pass

    def on_close(self):
        self._running = False
        self._save_default_config_silent()
        self.destroy()

    def _save_config_to_path(self, path: Path):
        data = asdict(self.cfg)
        data["roi"] = asdict(self.cfg.roi)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_config_from_path(self, path: Path, silent: bool = False) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            roi = raw.get("roi", {})
            self.cfg = Config(
                monitor_index=int(raw.get("monitor_index", 0)),
                roi=Roi(int(roi.get("x", 0)), int(roi.get("y", 0)),
                        int(roi.get("w", 300)), int(roi.get("h", 80))),
                click_abs_x=int(raw.get("click_abs_x", 0)),
                click_abs_y=int(raw.get("click_abs_y", 0)),
                poll_seconds=float(raw.get("poll_seconds", 0.75)),
                type_interval=float(raw.get("type_interval", 0.03)),
                require_change=bool(raw.get("require_change", True)),
                use_regex=bool(raw.get("use_regex", True)),
                regex_pattern=str(raw.get("regex_pattern", self.cfg.regex_pattern)),
                psm=int(raw.get("psm", 6)),
                whitelist=str(raw.get("whitelist", self.cfg.whitelist)),
                tesseract_cmd=str(raw.get("tesseract_cmd", "")),
            )
            return True
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Could not load config: {e}")
            return False

    def _load_screenshot_cache(self, path: Path):
        try:
            img = Image.open(path).convert("RGB")
            self.screen_img = img

            self.monitors = list_monitors()
            idx = self.cfg.monitor_index
            if idx < 0 or idx >= len(self.monitors):
                idx = 0
                self.cfg.monitor_index = 0

            self.mon_info = self.monitors[idx]
            self.zoom_var.set(1.0)
            self.zoom = 1.0

            self.clear_selections(keep_output=True)
            self.render_image()
            self.status_var.set(f"Screenshot cache loaded: {path}")
        except Exception:
            pass

    def _save_screenshot_cache(self):
        if self.screen_img is None:
            return
        path = _default_screenshot_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.screen_img.save(path, format="PNG")
        except Exception:
            pass

    def _sync_visual_points_from_cfg(self):
        r = self.cfg.roi
        if r.w > 0 and r.h > 0:
            self.roi_p1 = (r.x, r.y)
            self.roi_p2 = (r.x + r.w, r.y + r.h)
        else:
            self.roi_p1 = None
            self.roi_p2 = None

        if self.mon_info and self.cfg.click_abs_x and self.cfg.click_abs_y:
            ix = int(self.cfg.click_abs_x - self.mon_info["left"])
            iy = int(self.cfg.click_abs_y - self.mon_info["top"])
            self.point_xy = (ix, iy)
        else:
            self.point_xy = None

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        top = ttk.LabelFrame(root, text="Setup", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Monitor:").grid(row=0, column=0, sticky="w")
        self.monitor_combo = ttk.Combobox(top, state="readonly")
        self.monitor_combo.grid(row=0, column=1, sticky="we", padx=6)
        top.columnconfigure(1, weight=1)

        self.btn_capture = ttk.Button(top, text="Capture Screenshot", command=self.capture_screenshot)
        self.btn_capture.grid(row=0, column=2, padx=6)

        self.btn_roi = ttk.Button(top, text="Select ROI (2 clicks)", command=self.set_mode_roi)
        self.btn_roi.grid(row=0, column=3, padx=6)

        self.btn_point = ttk.Button(top, text="Select Input Point (1 click)", command=self.set_mode_point)
        self.btn_point.grid(row=0, column=4, padx=6)

        ttk.Label(top, text="Zoom:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = ttk.Scale(
            top, from_=0.25, to=4.0, orient="horizontal",
            variable=self.zoom_var, command=self.on_zoom_change
        )
        self.zoom_scale.grid(row=1, column=1, sticky="we", padx=6, pady=(8, 0))
        self.zoom_label = ttk.Label(top, text="100%")
        self.zoom_label.grid(row=1, column=2, sticky="w", pady=(8, 0))

        ttk.Label(top, text="ROI (x,y,w,h):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.roi_label = ttk.Label(top, text="0, 0, 300, 80")
        self.roi_label.grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(top, text="Click target (abs x,y):").grid(row=2, column=2, sticky="w", pady=(8, 0))
        self.click_label = ttk.Label(top, text="0, 0")
        self.click_label.grid(row=2, column=3, sticky="w", pady=(8, 0), columnspan=2)

        ocrf = ttk.LabelFrame(root, text="OCR & Extraction", padding=10)
        ocrf.pack(fill="x", pady=10)

        self.use_regex_var = tk.BooleanVar(value=True)
        self.chk_regex = ttk.Checkbutton(ocrf, text="Use regex", variable=self.use_regex_var)
        self.chk_regex.grid(row=0, column=0, sticky="w")

        ttk.Label(ocrf, text="Regex pattern:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.regex_var = tk.StringVar()
        self.regex_entry = ttk.Entry(ocrf, textvariable=self.regex_var)
        self.regex_entry.grid(row=1, column=1, columnspan=5, sticky="we", padx=6, pady=(6, 0))
        ocrf.columnconfigure(5, weight=1)

        ttk.Label(ocrf, text="PSM:").grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.psm_combo = ttk.Combobox(ocrf, state="readonly", values=[6, 7], width=5)
        self.psm_combo.grid(row=0, column=2, sticky="w", padx=6)

        ttk.Label(ocrf, text="Whitelist:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.whitelist_var = tk.StringVar()
        self.whitelist_entry = ttk.Entry(ocrf, textvariable=self.whitelist_var)
        self.whitelist_entry.grid(row=2, column=1, columnspan=5, sticky="we", padx=6, pady=(6, 0))

        ttk.Label(ocrf, text="tesseract.exe (optional):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.tess_var = tk.StringVar()
        self.tess_entry = ttk.Entry(ocrf, textvariable=self.tess_var)
        self.tess_entry.grid(row=3, column=1, columnspan=4, sticky="we", padx=6, pady=(6, 0))
        self.btn_browse_tess = ttk.Button(ocrf, text="Browse...", command=self.browse_tesseract)
        self.btn_browse_tess.grid(row=3, column=5, sticky="w", pady=(6, 0))

        runf = ttk.LabelFrame(root, text="Run", padding=10)
        runf.pack(fill="x")

        ttk.Label(runf, text="Polling interval (s):").grid(row=0, column=0, sticky="w")
        self.poll_var = tk.DoubleVar(value=0.75)
        self.entry_poll = ttk.Entry(runf, textvariable=self.poll_var, width=8)
        self.entry_poll.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(runf, text="Typing interval (s):").grid(row=0, column=2, sticky="w")
        self.type_var = tk.DoubleVar(value=0.03)
        self.entry_type = ttk.Entry(runf, textvariable=self.type_var, width=8)
        self.entry_type.grid(row=0, column=3, sticky="w", padx=6)

        self.change_var = tk.BooleanVar(value=True)
        self.chk_change = ttk.Checkbutton(runf, text="Only type on change", variable=self.change_var)
        self.chk_change.grid(row=0, column=4, sticky="w", padx=(12, 0))

        self.btn_test = ttk.Button(runf, text="Test once (OCR)", command=self.capture_once)
        self.btn_test.grid(row=1, column=0, pady=(10, 0))

        self.start_btn = ttk.Button(runf, text="Start", command=self.start)
        self.start_btn.grid(row=1, column=1, pady=(10, 0), padx=6, sticky="w")

        self.stop_btn = ttk.Button(runf, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=1, column=2, pady=(10, 0), padx=6, sticky="w")

        self.btn_load_cfg = ttk.Button(runf, text="Load config", command=self.load_config)
        self.btn_load_cfg.grid(row=1, column=3, pady=(10, 0), padx=6, sticky="w")

        self.btn_save_cfg = ttk.Button(runf, text="Save config", command=self.save_config)
        self.btn_save_cfg.grid(row=1, column=4, pady=(10, 0), padx=6, sticky="w")

        mid = ttk.PanedWindow(root, orient="horizontal")
        mid.pack(fill="both", expand=True, pady=10)

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=3)
        mid.add(right, weight=2)

        canvas_frame = ttk.LabelFrame(left, text="Screenshot (click to mark)", padding=6)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", highlightthickness=0)
        self.hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        out = ttk.LabelFrame(right, text="Output", padding=10)
        out.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready. Capture a screenshot.")
        ttk.Label(out, textvariable=self.status_var).pack(anchor="w")

        self.symbol_var = tk.StringVar(value="-")
        ttk.Label(out, text="Extracted symbol:", padding=(0, 10, 0, 0)).pack(anchor="w")
        ttk.Label(out, textvariable=self.symbol_var, font=("Segoe UI", 18, "bold")).pack(anchor="w")

        ttk.Label(out, text="Raw OCR text:", padding=(0, 10, 0, 0)).pack(anchor="w")
        self.ocr_text = ScrolledText(out, height=14)
        self.ocr_text.pack(fill="both", expand=True, pady=(6, 0))

        ttk.Label(out, text="ROI preview (binarized):", padding=(0, 10, 0, 0)).pack(anchor="w")
        self.preview_label = ttk.Label(out)
        self.preview_label.pack(anchor="w", pady=(6, 0))

        # ---- Tooltips ----
        ToolTip(self.monitor_combo, "Select which monitor to capture from.\n"
                                    "Monitor 0 is the virtual combined desktop (all monitors).")
        ToolTip(self.btn_capture, "Capture a full screenshot of the selected monitor.\n"
                                  "This screenshot is used for selecting ROI and the input point.")
        ToolTip(self.btn_roi, "ROI selection mode.\n"
                              "First click: top-left corner.\n"
                              "Second click: bottom-right corner.\n"
                              "The ROI defines the screen area that will be OCR'd.")
        ToolTip(self.btn_point, "Input point selection mode.\n"
                                "Click once on the screenshot where the bot should click\n"
                                "before typing the extracted symbol (e.g., a search field).")
        ToolTip(self.zoom_scale, "Zoom the screenshot view for easier ROI/point selection.\n"
                                 "Zoom affects only the view, not OCR coordinates.")
        ToolTip(self.chk_regex, "If enabled, extraction uses the regex pattern.\n"
                                "If disabled, the bot falls back to token extraction.")
        ToolTip(self.regex_entry, "Regex used to extract the symbol from OCR text.\n"
                                  "Capture groups are supported. Group 1 is used if present.")
        ToolTip(self.psm_combo, "Tesseract PSM (Page Segmentation Mode).\n"
                                "6: Assume a block of text.\n"
                                "7: Treat the image as a single text line.")
        ToolTip(self.whitelist_entry, "Whitelist of characters for Tesseract.\n"
                                      "Restricting characters often improves OCR accuracy.")
        ToolTip(self.tess_entry, "Optional: set a custom path to tesseract.exe.\n"
                                 "Leave empty if Tesseract is available via PATH.")
        ToolTip(self.btn_browse_tess, "Browse to select tesseract.exe on your system.")
        ToolTip(self.entry_poll, "Polling interval in seconds.\n"
                                 "Defines how often OCR is performed while running.")
        ToolTip(self.entry_type, "Typing interval per character in seconds.\n"
                                 "Increase this if the target app misses keystrokes.")
        ToolTip(self.chk_change, "If enabled, the bot types only when the extracted symbol changes.\n"
                                 "This prevents repeated sends of the same symbol.")
        ToolTip(self.btn_test, "Run OCR once on the current ROI and show results.\n"
                               "No typing is performed.")
        ToolTip(self.start_btn, "Start continuous OCR polling.\n"
                                "If a symbol is extracted, the bot clicks the input point and types it.")
        ToolTip(self.stop_btn, "Stop the running OCR loop.")
        ToolTip(self.btn_load_cfg, "Load configuration from a JSON file.\n"
                                   "Note: you may need to capture a new screenshot afterwards.")
        ToolTip(self.btn_save_cfg, "Save current configuration to a JSON file.")
        ToolTip(self.canvas, "Screenshot canvas.\n"
                             "Use it to select ROI and input point.\n"
                             "ROI mode: two clicks. Point mode: one click.")

    def _refresh_monitor_dropdown(self):
        vals = []
        for i, m in enumerate(self.monitors):
            if i == 0:
                vals.append("0: Virtual (All Monitors)")
            else:
                vals.append(f"{i}: left={m['left']} top={m['top']} {m['width']}x{m['height']}")
        self.monitor_combo["values"] = vals
        self.monitor_combo.current(self.cfg.monitor_index)
        self.monitor_combo.bind("<<ComboboxSelected>>", self._on_monitor_changed)

    def _on_monitor_changed(self, _evt=None):
        self.cfg.monitor_index = int(self.monitor_combo.current())
        self.status_var.set(f"Monitor set to {self.cfg.monitor_index}. Capture a screenshot.")
        self.clear_selections()

    def browse_tesseract(self):
        path = filedialog.askopenfilename(
            title="Select tesseract.exe",
            filetypes=[("tesseract.exe", "tesseract.exe"), ("All files", "*.*")]
        )
        if path:
            self.tess_var.set(path)

    def capture_screenshot(self):
        try:
            idx = int(self.monitor_combo.current())
            img, info = grab_monitor_image_and_info(idx)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.cfg.monitor_index = idx
        self.screen_img = img
        self.mon_info = info
        self.zoom_var.set(1.0)
        self.zoom = 1.0

        self.clear_selections(keep_output=True)
        self.render_image()

        self._save_screenshot_cache()
        self._sync_visual_points_from_cfg()
        self.redraw_selections()

        self.status_var.set("Screenshot loaded. Select ROI (2 clicks) or select input point (1 click).")

    def render_image(self):
        if self.screen_img is None:
            return

        z = float(self.zoom_var.get())
        self.zoom = max(0.05, min(10.0, z))
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")

        w, h = self.screen_img.size
        disp = self.screen_img.resize((int(w * self.zoom), int(h * self.zoom)), Image.Resampling.BILINEAR)
        self.screen_tk = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self.screen_tk)
        self.canvas.config(scrollregion=(0, 0, disp.size[0], disp.size[1]))

        self.rect_id = None
        self.cross_ids = []
        self.redraw_selections()

    def on_zoom_change(self, _val=None):
        if self.screen_img is None:
            return
        self.render_image()

    def clear_selections(self, keep_output: bool = False):
        self.mode = "idle"
        self.roi_p1 = None
        self.roi_p2 = None
        self.point_xy = None
        self._last_sent = None

        self.roi_label.config(text=self._roi_text())
        self.click_label.config(text=f"{self.cfg.click_abs_x}, {self.cfg.click_abs_y}")

        if not keep_output:
            self.symbol_var.set("-")
            self.ocr_text.delete("1.0", "end")
            self.preview_label.configure(image="")
            self.preview_label.image = None

    def set_mode_roi(self):
        if self.screen_img is None:
            messagebox.showwarning("Info", "Please capture a screenshot first.")
            return
        self.mode = "roi"
        self.roi_p1 = None
        self.roi_p2 = None
        self.status_var.set("ROI mode: 1st click = top-left, 2nd click = bottom-right (on the screenshot).")

    def set_mode_point(self):
        if self.screen_img is None:
            messagebox.showwarning("Info", "Please capture a screenshot first.")
            return
        self.mode = "point"
        self.status_var.set("Point mode: click once on the screenshot to set the input click target.")

    def canvas_to_image_coords(self, event) -> Optional[Tuple[int, int]]:
        if self.screen_img is None:
            return None
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix = int(cx / self.zoom)
        iy = int(cy / self.zoom)
        w, h = self.screen_img.size
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    def on_canvas_click(self, event):
        if self.screen_img is None or self.mon_info is None:
            return

        pos = self.canvas_to_image_coords(event)
        if pos is None:
            return
        ix, iy = pos

        if self.mode == "roi":
            if self.roi_p1 is None:
                self.roi_p1 = (ix, iy)
                self.roi_p2 = None
                self.status_var.set("ROI mode: 2nd click = bottom-right.")
                self.redraw_selections()
            else:
                self.roi_p2 = (ix, iy)
                self.apply_roi_from_points(self.roi_p1, self.roi_p2)
                self.mode = "idle"
                self.status_var.set("ROI set. You can now run 'Test once (OCR)' or 'Start'.")
                self.redraw_selections()

        elif self.mode == "point":
            self.point_xy = (ix, iy)
            abs_x = int(self.mon_info["left"] + ix)
            abs_y = int(self.mon_info["top"] + iy)
            self.cfg.click_abs_x = abs_x
            self.cfg.click_abs_y = abs_y
            self.click_label.config(text=f"{abs_x}, {abs_y}")
            self.mode = "idle"
            self.status_var.set("Input click target set.")
            self.redraw_selections()

    def on_canvas_motion(self, event):
        if self.mode != "roi" or self.screen_img is None or self.roi_p1 is None or self.roi_p2 is not None:
            return
        pos = self.canvas_to_image_coords(event)
        if pos is None:
            return
        self.redraw_selections(temp_p2=pos)

    def apply_roi_from_points(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        x1, y1 = p1
        x2, y2 = p2
        rx = int(min(x1, x2))
        ry = int(min(y1, y2))
        rw = int(abs(x2 - x1))
        rh = int(abs(y2 - y1))
        if rw < 2 or rh < 2:
            messagebox.showwarning("ROI", "ROI is too small. Please select again.")
            return
        self.cfg.roi = Roi(rx, ry, rw, rh)
        self.roi_label.config(text=self._roi_text())
        self._sync_visual_points_from_cfg()

    def _roi_text(self):
        r = self.cfg.roi
        return f"{r.x}, {r.y}, {r.w}, {r.h}"

    def redraw_selections(self, temp_p2: Optional[Tuple[int, int]] = None):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        for cid in self.cross_ids:
            self.canvas.delete(cid)
        self.cross_ids = []

        p1 = self.roi_p1
        p2 = self.roi_p2 or temp_p2
        if p1 is not None and p2 is not None:
            x1, y1 = p1
            x2, y2 = p2
            dx1, dy1 = x1 * self.zoom, y1 * self.zoom
            dx2, dy2 = x2 * self.zoom, y2 * self.zoom
            self.rect_id = self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="lime", width=2)

        if self.point_xy is not None:
            x, y = self.point_xy
            dx, dy = x * self.zoom, y * self.zoom
            self.cross_ids.append(self.canvas.create_line(dx - 10, dy, dx + 10, dy, fill="lime", width=2))
            self.cross_ids.append(self.canvas.create_line(dx, dy - 10, dx, dy + 10, fill="lime", width=2))

    def sync_cfg_from_ui(self, require_screenshot: bool = True) -> Optional[str]:
        try:
            self.cfg.monitor_index = int(self.monitor_combo.current())
            self.cfg.poll_seconds = float(self.poll_var.get())
            self.cfg.type_interval = float(self.type_var.get())
            self.cfg.require_change = bool(self.change_var.get())
            self.cfg.use_regex = bool(self.use_regex_var.get())
            self.cfg.regex_pattern = self.regex_var.get()
            self.cfg.psm = int(self.psm_combo.get())
            self.cfg.whitelist = self.whitelist_var.get()
            self.cfg.tesseract_cmd = self.tess_var.get().strip()
        except Exception as e:
            return f"Invalid input: {e}"

        if self.cfg.use_regex and self.cfg.regex_pattern.strip():
            try:
                re.compile(self.cfg.regex_pattern, re.IGNORECASE)
            except re.error as e:
                return f"Invalid regex: {e}"

        if self.cfg.poll_seconds <= 0:
            return "Polling interval must be > 0."
        if self.cfg.type_interval < 0:
            return "Typing interval must be >= 0."
        if require_screenshot and self.mon_info is None:
            return "Please capture a screenshot first."
        return None

    def capture_once(self):
        err = self.sync_cfg_from_ui(require_screenshot=True)
        if err:
            messagebox.showerror("Error", err)
            return

        try:
            sym, raw, prep = grab_roi_text(self.cfg, self.mon_info)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.update_output(sym, raw, prep)
        self.status_var.set("OCR test completed.")

    def update_output(self, sym: Optional[str], raw: str, prep: Image.Image):
        self.symbol_var.set(sym if sym else "-")
        self.ocr_text.delete("1.0", "end")
        self.ocr_text.insert("end", raw.strip() + "\n")

        prev = prep.copy()
        prev.thumbnail((420, 140))
        tkimg = ImageTk.PhotoImage(prev)
        self.preview_label.configure(image=tkimg)
        self.preview_label.image = tkimg

    def start(self):
        err = self.sync_cfg_from_ui(require_screenshot=True)
        if err:
            messagebox.showerror("Error", err)
            return

        if self.cfg.click_abs_x == 0 and self.cfg.click_abs_y == 0:
            if not messagebox.askyesno("Warning", "Click target is (0,0). Start anyway?"):
                return

        self._running = True
        self._last_sent = None
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running... (press Stop to end).")

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopped.")

    def _run_loop(self):
        while self._running:
            try:
                sym, raw, prep = grab_roi_text(self.cfg, self.mon_info)
                self.after(0, self.update_output, sym, raw, prep)

                if sym:
                    should_send = True
                    if self.cfg.require_change and self._last_sent == sym:
                        should_send = False

                    if should_send:
                        self.after(0, self.status_var.set, f"Typing symbol: {sym}")
                        type_symbol_into_target(self.cfg, sym)
                        self._last_sent = sym
                else:
                    self.after(0, self.status_var.set, "No symbol extracted.")

            except pyautogui.FailSafeException:
                self.after(0, self.status_var.set, "Aborted: PyAutoGUI FailSafe (mouse moved to top-left corner).")
                self.after(0, self.stop)
                return
            except Exception as e:
                self.after(0, self.status_var.set, f"Error: {e}")

            time.sleep(self.cfg.poll_seconds)

    def save_config(self):
        err = self.sync_cfg_from_ui(require_screenshot=False)
        if err:
            messagebox.showerror("Error", err)
            return

        path = filedialog.asksaveasfilename(
            title="Save config",
            defaultextension=".json",
            initialfile=CONFIG_DEFAULT,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self._save_config_to_path(Path(path))
            self.status_var.set(f"Config saved: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save config: {e}")

    def load_config(self):
        path = filedialog.askopenfilename(
            title="Load config",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        ok = self._load_config_from_path(Path(path), silent=False)
        if not ok:
            return

        self.monitors = list_monitors()
        if 0 <= self.cfg.monitor_index < len(self.monitors):
            self.mon_info = self.monitors[self.cfg.monitor_index]
            self.monitor_combo.current(self.cfg.monitor_index)
        else:
            self.mon_info = None
            self.cfg.monitor_index = 0
            self.monitor_combo.current(0)

        self._apply_cfg_to_ui()
        self._sync_visual_points_from_cfg()
        self.redraw_selections()
        self.status_var.set(f"Config loaded: {path}. (You may need to capture a new screenshot.)")


if __name__ == "__main__":
    enable_dpi_awareness()
    set_windows_appusermodelid(APP_USER_MODEL_ID)
    App().mainloop()
