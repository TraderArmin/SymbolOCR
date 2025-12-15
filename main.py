import json
import re
import threading
import time
import queue
import shutil
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List
from pathlib import Path
import os
import ctypes
import sys

import numpy as np
import pyautogui
import pytesseract
from mss import mss
from PIL import Image, ImageOps, ImageEnhance, ImageTk, ImageDraw, ImageFont

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base) / rel)
    return str(Path(__file__).parent / rel)


os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", resource_path("pw-browsers"))


def auto_set_tesseract_path(cfg):
    # If user did not specify a custom path, use bundled tesseract
    if not cfg.tesseract_cmd.strip():
        cfg.tesseract_cmd = resource_path(r"tesseract\tesseract.exe")


CONFIG_DEFAULT = "ocr_symbol_gui_config.json"
APP_TITLE = "Symbol Extractor (ATLAS) – OCR & Auto-Input"
APP_USER_MODEL_ID = "ocr_symbol_bot.OCRSymbolBot"


# -----------------------------
# Friendly UI mappings / help text
# -----------------------------
TARGET_MODE_DISPLAY_TO_VALUE = {
    "Desktop (mouse & keyboard) – PyAutoGUI": "pyautogui",
    "Web (ATLAS in browser) – Playwright": "playwright",
}
TARGET_MODE_VALUE_TO_DISPLAY = {v: k for k, v in TARGET_MODE_DISPLAY_TO_VALUE.items()}

PW_ENGINE_DISPLAY_TO_VALUE = {
    "Chromium engine (Playwright)": "chromium",
    "Firefox engine (Playwright)": "firefox",
    "WebKit engine (Safari-like, Playwright)": "webkit",
}
PW_ENGINE_VALUE_TO_DISPLAY = {v: k for k, v in PW_ENGINE_DISPLAY_TO_VALUE.items()}

PW_CHANNEL_DISPLAY_TO_VALUE = {
    "Auto (Playwright Chromium)": "",
    "Google Chrome (installed)": "chrome",
    "Microsoft Edge (installed)": "msedge",
    "Chromium (installed)": "chromium",
}
PW_CHANNEL_VALUE_TO_DISPLAY = {v: k for k, v in PW_CHANNEL_DISPLAY_TO_VALUE.items()}


def help_text_glossary() -> str:
    return (
        "Glossary (short & practical)\n\n"
        "• OCR:\n"
        "  Text recognition from a screenshot area (ROI).\n\n"
        "• ROI (Region of Interest):\n"
        "  The rectangular area of the screenshot used for OCR.\n\n"
        "• PyAutoGUI (Desktop mode):\n"
        "  Controls OS mouse/keyboard. Works with any app.\n"
        "  Downside: moves the real mouse pointer and depends on window focus.\n\n"
        "• Playwright (Web/Browser mode):\n"
        "  Controls the browser directly (without moving the OS mouse).\n"
        "  Upside: more stable for web inputs; login/session can be persisted.\n\n"
        "• Chromium / Firefox / WebKit:\n"
        "  Browser engines Playwright can drive.\n"
        "  Chromium is the base for Chrome/Edge; WebKit is Safari-like.\n\n"
        "• Channel (Chromium only):\n"
        "  Use an installed browser (Chrome/Edge) instead of Playwright’s bundled Chromium.\n\n"
        "• Input selector (CSS):\n"
        "  Optional CSS selector for the ATLAS symbol input field.\n"
        "  Leave empty = auto-detect (recommended).\n"
    )


def help_text_quick_start(mode: str) -> str:
    if mode == "playwright":
        return (
            "Quick Start (Web / Playwright)\n\n"
            "1) Capture screenshot (so you can define ROI)\n"
            "2) Select ROI (2 clicks)\n"
            "3) 'Open / Login Browser' → log in once\n"
            "4) Start\n\n"
            "Note: In Playwright mode you do NOT need a desktop click target."
        )
    return (
        "Quick Start (Desktop / PyAutoGUI)\n\n"
        "1) Capture screenshot\n"
        "2) Select ROI (2 clicks)\n"
        "3) Select input point (1 click) – that’s where it will click & type\n"
        "4) Start\n\n"
        "Tip: PyAutoGUI FailSafe: move mouse to top-left corner to abort."
    )


def info_text_playwright() -> str:
    return (
        "Playwright (browser automation)\n\n"
        "Playwright controls the browser directly. This is often more reliable than OS mouse clicks,\n"
        "because it targets the actual web element (the symbol input).\n\n"
        "Login/session is stored in:\n"
        "~/.ocr_symbol_bot/pw_profile/\n\n"
        "If ATLAS changes, you can provide a CSS selector — usually not needed."
    )


def info_text_chromium_channel() -> str:
    return (
        "Chromium vs. Channel\n\n"
        "• Browser engine = which engine Playwright drives (Chromium/Firefox/WebKit).\n"
        "• Channel (Chromium only) = use an installed browser instead of Playwright’s Chromium:\n"
        "  - 'Google Chrome (installed)' → channel 'chrome'\n"
        "  - 'Microsoft Edge (installed)' → channel 'msedge'\n"
    )


# -----------------------------
# Tooltips (mouse-over help)
# -----------------------------
class ToolTip:
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
        if self._tip is not None or not self.text:
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
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def set_windows_appusermodelid(appid: str):
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


def _playwright_profile_dir() -> Path:
    p = _app_state_dir() / "pw_profile"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _playwright_storage_state_path() -> Path:
    return _app_state_dir() / "atlas_storage_state.json"


def _create_app_icon(png_path: Path, ico_path: Path) -> None:
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bg1 = (24, 32, 52, 255)
    bg2 = (18, 86, 123, 255)
    accent = (0, 198, 255, 60)

    d.rectangle([0, 0, size, size], fill=bg1)
    d.polygon([(0, 0), (size, 0), (0, size)], fill=bg2)
    d.ellipse([size * 0.15, size * 0.10, size * 0.95, size * 0.90], fill=accent)

    border = (255, 255, 255, 40)
    margin = 10
    d.rounded_rectangle([margin, margin, size - margin, size - margin],
                        radius=34, outline=border, width=2)

    stroke = (240, 248, 255, 235)
    stroke2 = (0, 0, 0, 70)
    cx, cy = int(size * 0.43), int(size * 0.43)
    r = int(size * 0.20)
    w = 14

    d.ellipse([cx - r + 4, cy - r + 6, cx + r + 4, cy + r + 6], outline=stroke2, width=w)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=stroke, width=w)

    hx1, hy1 = int(cx + r * 0.62), int(cy + r * 0.62)
    hx2, hy2 = int(size * 0.78), int(size * 0.78)
    d.line([(hx1 + 4, hy1 + 6), (hx2 + 4, hy2 + 6)], fill=stroke2, width=w + 2)
    d.line([(hx1, hy1), (hx2, hy2)], fill=stroke, width=w)

    scan = (255, 255, 255, 70)
    d.line([(cx - r // 2, cy - r // 6), (cx + r // 2, cy - r // 6)], fill=scan, width=6)
    d.line([(cx - r // 2, cy + r // 6), (cx + r // 3, cy + r // 6)], fill=scan, width=6)

    try:
        font = ImageFont.truetype("segoeui.ttf", 44)
    except Exception:
        font = ImageFont.load_default()

    text = "OCR"
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

    # ---- Target / Playwright ----
    target_mode: str = "pyautogui"  # "pyautogui" | "playwright"
    atlas_url: str = "https://tradingterminal.com/atlas"
    pw_input_selector: str = ""  # optional override; if empty, auto-detect

    # Browser placement:
    browser_monitor_index: int = 0
    browser_x: int = 50
    browser_y: int = 50
    browser_w: int = 1400
    browser_h: int = 900

    # Browser selection:
    pw_browser: str = "chromium"   # chromium | firefox | webkit
    pw_channel: str = ""          # chromium only: "" | "chrome" | "msedge" | "chromium"


# -----------------------------
# Playwright controller (runs in its own thread)
# -----------------------------
class PlaywrightController:
    """
    Dedicated worker thread using playwright.sync_api.
    Uses a persistent profile directory so login/session is saved.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._cmd_q: queue.Queue = queue.Queue()
        self._running = False
        self._ready = threading.Event()
        self._last_cfg_sig: Optional[Tuple] = None

        # set in worker
        self._pw = None
        self._context = None
        self._page = None
        self._error: Optional[str] = None

    def is_running(self) -> bool:
        return bool(self._running and self._thread and self._thread.is_alive())

    def _cfg_sig(self, cfg: Config) -> Tuple:
        return (
            cfg.atlas_url,
            cfg.browser_monitor_index,
            cfg.browser_x, cfg.browser_y, cfg.browser_w, cfg.browser_h,
            cfg.pw_browser, cfg.pw_channel,
        )

    def ensure_started(self, cfg: Config, timeout_s: float = 20.0) -> Tuple[bool, str]:
        """
        Starts (or restarts) Playwright with current geometry if needed.
        """
        sig = self._cfg_sig(cfg)
        if self.is_running() and self._last_cfg_sig == sig:
            return True, "Playwright already running."

        # restart if running but geometry/url/browser changed
        if self.is_running():
            self.stop()

        self._ready.clear()
        self._error = None
        self._running = True
        self._last_cfg_sig = sig
        self._thread = threading.Thread(target=self._worker_main, args=(cfg,), daemon=True)
        self._thread.start()

        ok = self._ready.wait(timeout_s)
        if not ok:
            return False, "Playwright start timed out (is Playwright installed?)."
        if self._error:
            return False, self._error
        return True, "Playwright started."

    def stop(self):
        if not self.is_running():
            return
        try:
            self._cmd_q.put(("__stop__", None, None))
        except Exception:
            pass
        self._running = False
        try:
            if self._thread:
                self._thread.join(timeout=5.0)
        except Exception:
            pass

    def reset_login(self):
        """
        Clears the persistent profile directory (logout).
        """
        self.stop()
        prof = _playwright_profile_dir()
        try:
            shutil.rmtree(prof, ignore_errors=True)
        except Exception:
            pass
        prof.mkdir(parents=True, exist_ok=True)
        try:
            st = _playwright_storage_state_path()
            if st.exists():
                st.unlink(missing_ok=True)
        except Exception:
            pass

    def _call(self, name: str, payload: dict, timeout_s: float = 10.0):
        if not self.is_running():
            raise RuntimeError("Playwright not running.")
        resp_q: queue.Queue = queue.Queue(maxsize=1)
        self._cmd_q.put((name, payload, resp_q))
        try:
            ok, data = resp_q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(f"Playwright call timed out: {name}")
        if not ok:
            raise RuntimeError(data if isinstance(data, str) else str(data))
        return data

    def open_atlas(self, timeout_s: float = 30.0) -> Tuple[bool, str]:
        try:
            self._call("open_atlas", {}, timeout_s=timeout_s)
            return True, "ATLAS opened."
        except Exception as e:
            return False, str(e)

    def is_logged_in(self, cfg: Config, timeout_s: float = 5.0) -> Tuple[bool, str]:
        try:
            data = self._call("is_logged_in", {"selectors": self._candidate_selectors(cfg)}, timeout_s=timeout_s)
            return bool(data.get("logged_in")), "OK"
        except Exception as e:
            return False, str(e)

    def wait_for_login(self, cfg: Config, timeout_s: float = 180.0) -> Tuple[bool, str]:
        """
        Waits until a symbol input appears (heuristic). Also saves storage_state.
        """
        try:
            self._call("wait_for_login", {
                "selectors": self._candidate_selectors(cfg),
                "timeout_ms": int(timeout_s * 1000)
            }, timeout_s=max(5.0, timeout_s + 5.0))
            return True, "Login detected and saved."
        except Exception as e:
            return False, str(e)

    def send_symbol(self, cfg: Config, symbol: str, timeout_s: float = 10.0) -> Tuple[bool, str]:
        try:
            self._call("send_symbol", {
                "symbol": symbol,
                "selectors": self._candidate_selectors(cfg),
                "type_delay_ms": max(0, int(cfg.type_interval * 1000)),
            }, timeout_s=timeout_s)
            return True, "Symbol sent via Playwright."
        except Exception as e:
            return False, str(e)

    def _candidate_selectors(self, cfg: Config) -> List[str]:
        if cfg.pw_input_selector.strip():
            return [cfg.pw_input_selector.strip()]

        return [
            'input[aria-label="Symbol"]',
            'input[placeholder="Enter symbol"]',
            'input[placeholder*="symbol" i]',
            'input[aria-label*="symbol" i]',
            'input.obhm-input',
        ]

    @staticmethod
    def _compute_window_position(cfg: Config) -> Tuple[int, int]:
        """
        Returns absolute window X/Y.
        If browser_monitor_index > 0: base is that monitor's left/top.
        Else: use cfg.browser_x/y directly.
        """
        try:
            mons = list_monitors()
            idx = int(cfg.browser_monitor_index)
            if 0 < idx < len(mons):
                base = mons[idx]
                return int(base["left"] + cfg.browser_x), int(base["top"] + cfg.browser_y)
        except Exception:
            pass
        return int(cfg.browser_x), int(cfg.browser_y)

    def _worker_main(self, cfg: Config):
        try:
            from playwright.sync_api import sync_playwright  # noqa
        except Exception as e:
            self._error = (
                "Playwright import failed.\n\n"
                "Install:\n"
                "  pip install playwright\n"
                "  playwright install\n\n"
                f"Details: {e}"
            )
            self._ready.set()
            self._running = False
            return

        try:
            from playwright.sync_api import sync_playwright

            pos_x, pos_y = self._compute_window_position(cfg)

            args = [
                f"--window-size={int(cfg.browser_w)},{int(cfg.browser_h)}",
                f"--window-position={int(pos_x)},{int(pos_y)}",
            ]

            user_data_dir = str(_playwright_profile_dir())
            storage_state_path = str(_playwright_storage_state_path())

            pw_browser = (cfg.pw_browser or "chromium").strip().lower()
            pw_channel = (cfg.pw_channel or "").strip()

            with sync_playwright() as p:
                # Choose engine
                if pw_browser == "firefox":
                    launcher = p.firefox
                elif pw_browser == "webkit":
                    launcher = p.webkit
                else:
                    launcher = p.chromium
                    pw_browser = "chromium"  # normalize

                launch_kwargs = dict(
                    user_data_dir=user_data_dir,
                    headless=False,
                    args=args,
                    no_viewport=True,
                )

                # channel only for chromium
                if pw_browser == "chromium" and pw_channel:
                    launch_kwargs["channel"] = pw_channel

                context = launcher.launch_persistent_context(**launch_kwargs)

                page = context.pages[0] if context.pages else context.new_page()

                try:
                    page.set_default_timeout(15000)
                except Exception:
                    pass

                self._pw = p
                self._context = context
                self._page = page

                try:
                    page.goto(cfg.atlas_url, wait_until="domcontentloaded")
                except Exception:
                    pass

                self._ready.set()

                while True:
                    cmd, payload, resp_q = self._cmd_q.get()
                    if cmd == "__stop__":
                        break
                    try:
                        if cmd == "open_atlas":
                            page.goto(cfg.atlas_url, wait_until="domcontentloaded")
                            if resp_q:
                                resp_q.put((True, {}))

                        elif cmd == "is_logged_in":
                            selectors = payload.get("selectors", [])
                            logged_in = self._any_selector_visible(page, selectors, timeout_ms=1500)
                            if resp_q:
                                resp_q.put((True, {"logged_in": bool(logged_in)}))

                        elif cmd == "wait_for_login":
                            selectors = payload.get("selectors", [])
                            timeout_ms = int(payload.get("timeout_ms", 180000))
                            self._wait_any_selector(page, selectors, timeout_ms=timeout_ms)
                            try:
                                context.storage_state(path=storage_state_path)
                            except Exception:
                                pass
                            if resp_q:
                                resp_q.put((True, {}))

                        elif cmd == "send_symbol":
                            symbol = str(payload.get("symbol", "")).strip()
                            selectors = payload.get("selectors", [])
                            delay_ms = int(payload.get("type_delay_ms", 0))

                            loc = self._resolve_locator(page, selectors, timeout_ms=5000)
                            loc.click(timeout=5000)
                            try:
                                loc.press("Control+A")
                            except Exception:
                                pass
                            try:
                                loc.type(symbol, delay=delay_ms)
                            except Exception:
                                loc.fill(symbol)
                            try:
                                loc.press("Enter")
                            except Exception:
                                pass

                            if resp_q:
                                resp_q.put((True, {}))

                        else:
                            if resp_q:
                                resp_q.put((False, f"Unknown command: {cmd}"))
                    except Exception as e:
                        if resp_q:
                            resp_q.put((False, str(e)))

                try:
                    context.close()
                except Exception:
                    pass

        except Exception as e:
            self._error = str(e)
            self._ready.set()
        finally:
            self._running = False

    @staticmethod
    def _resolve_locator(page, selectors: List[str], timeout_ms: int = 5000):
        last_err = None
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="attached", timeout=timeout_ms)
                return loc
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Input field not found. Tried selectors: {selectors}. Last error: {last_err}")

    @staticmethod
    def _any_selector_visible(page, selectors: List[str], timeout_ms: int = 1500) -> bool:
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=timeout_ms)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _wait_any_selector(page, selectors: List[str], timeout_ms: int = 180000):
        last_err = None
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                loc.wait_for(state="visible", timeout=timeout_ms)
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Login/ready state not detected. Tried selectors: {selectors}. Last error: {last_err}")


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


def type_symbol_into_target_pyautogui(cfg: Config, symbol: str) -> None:
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
        self.geometry("1260x900")

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

        self.pw = PlaywrightController()

        self._build_ui()
        self._refresh_monitor_dropdowns()

        self._load_startup_state()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_app_icon(self):
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
                self._tk_app_icon_ref = tk_img
        except Exception:
            pass

    # ---------- Menu ----------
    def _build_menu(self):
        menubar = tk.Menu(self)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Quick Start (current mode)", command=self._show_quick_start)
        helpmenu.add_command(label="Glossary (Playwright, Chromium, ROI, ...)", command=self._show_glossary)
        helpmenu.add_separator()
        helpmenu.add_command(label="About", command=self._show_about)

        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

    def _show_glossary(self):
        messagebox.showinfo("Glossary", help_text_glossary())

    def _show_quick_start(self):
        mode = self.target_mode_var.get().strip().lower()
        # mode var is display text; convert to internal
        internal = TARGET_MODE_DISPLAY_TO_VALUE.get(mode, None)
        if internal is None:
            internal = self.cfg.target_mode
        messagebox.showinfo("Quick Start", help_text_quick_start(internal))

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Symbol Extractor (ATLAS)\n\n"
            "• Captures a screen region (ROI)\n"
            "• Runs OCR to extract a symbol\n"
            "• Types it into a target (Desktop via PyAutoGUI or Web via Playwright)\n"
        )

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

        # screenshot monitor
        if 0 <= self.cfg.monitor_index < len(self.monitors):
            self.monitor_combo.current(self.cfg.monitor_index)
        else:
            self.cfg.monitor_index = 0
            self.monitor_combo.current(0)

        # browser monitor
        if 0 <= self.cfg.browser_monitor_index < len(self.monitors):
            self.browser_monitor_combo.current(self.cfg.browser_monitor_index)
        else:
            self.cfg.browser_monitor_index = 0
            self.browser_monitor_combo.current(0)

        self._apply_cfg_to_ui()
        self._sync_visual_points_from_cfg()
        self.redraw_selections()
        self._update_target_ui_state()

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

        # Friendly labels in UI
        self.target_mode_var.set(TARGET_MODE_VALUE_TO_DISPLAY.get(self.cfg.target_mode, "Desktop (mouse & keyboard) – PyAutoGUI"))
        self.atlas_url_var.set(self.cfg.atlas_url)
        self.pw_selector_var.set(self.cfg.pw_input_selector)

        self.browser_monitor_var.set(self.cfg.browser_monitor_index)
        self.bx_var.set(self.cfg.browser_x)
        self.by_var.set(self.cfg.browser_y)
        self.bw_var.set(self.cfg.browser_w)
        self.bh_var.set(self.cfg.browser_h)

        self.pw_browser_var.set(PW_ENGINE_VALUE_TO_DISPLAY.get(self.cfg.pw_browser, "Chromium engine (Playwright)"))
        self.pw_channel_var.set(PW_CHANNEL_VALUE_TO_DISPLAY.get(self.cfg.pw_channel, "Auto (Playwright Chromium)"))

    def _save_default_config_silent(self):
        self.sync_cfg_from_ui(require_screenshot=False)
        try:
            self._save_config_to_path(_default_config_path())
        except Exception:
            pass

    def on_close(self):
        self._running = False
        try:
            self.pw.stop()
        except Exception:
            pass
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
            # Backward compatible defaults
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

                target_mode=str(raw.get("target_mode", "pyautogui")),
                atlas_url=str(raw.get("atlas_url", "https://tradingterminal.com/atlas")),
                pw_input_selector=str(raw.get("pw_input_selector", "")),

                browser_monitor_index=int(raw.get("browser_monitor_index", raw.get("monitor_index", 0))),
                browser_x=int(raw.get("browser_x", 50)),
                browser_y=int(raw.get("browser_y", 50)),
                browser_w=int(raw.get("browser_w", 1400)),
                browser_h=int(raw.get("browser_h", 900)),

                pw_browser=str(raw.get("pw_browser", "chromium")),
                pw_channel=str(raw.get("pw_channel", "")),
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
        self._build_menu()

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

        # ---- Target / Playwright ----
        targetf = ttk.LabelFrame(root, text="Target (Typing Backend)", padding=10)
        targetf.pack(fill="x", pady=(10, 0))

        ttk.Label(targetf, text="Mode:").grid(row=0, column=0, sticky="w")
        self.target_mode_var = tk.StringVar(value="Desktop (mouse & keyboard) – PyAutoGUI")
        self.target_mode_combo = ttk.Combobox(
            targetf, state="readonly",
            values=list(TARGET_MODE_DISPLAY_TO_VALUE.keys()),
            textvariable=self.target_mode_var, width=34
        )
        self.target_mode_combo.grid(row=0, column=1, sticky="w", padx=6)
        self.target_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_target_ui_state())

        self.btn_mode_help = ttk.Button(targetf, text="?", width=3, command=self._mode_help)
        self.btn_mode_help.grid(row=0, column=1, sticky="w", padx=(370, 0))

        ttk.Label(targetf, text="ATLAS URL:").grid(row=0, column=2, sticky="e")
        self.atlas_url_var = tk.StringVar()
        self.atlas_url_entry = ttk.Entry(targetf, textvariable=self.atlas_url_var)
        self.atlas_url_entry.grid(row=0, column=3, sticky="we", padx=6)
        targetf.columnconfigure(3, weight=1)

        ttk.Label(targetf, text="Input selector (optional):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.pw_selector_var = tk.StringVar()
        self.pw_selector_entry = ttk.Entry(targetf, textvariable=self.pw_selector_var)
        self.pw_selector_entry.grid(row=1, column=1, columnspan=3, sticky="we", padx=6, pady=(6, 0))

        self.btn_selector_help = ttk.Button(targetf, text="?", width=3, command=self._selector_help)
        self.btn_selector_help.grid(row=1, column=3, sticky="e", padx=(0, 6), pady=(6, 0))

        # Browser selection row
        ttk.Label(targetf, text="Browser engine:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.pw_browser_var = tk.StringVar(value="Chromium engine (Playwright)")
        self.pw_browser_combo = ttk.Combobox(
            targetf, state="readonly",
            values=list(PW_ENGINE_DISPLAY_TO_VALUE.keys()),
            textvariable=self.pw_browser_var, width=28
        )
        self.pw_browser_combo.grid(row=2, column=1, sticky="w", padx=(6, 2), pady=(6, 0))
        self.pw_browser_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_target_ui_state())

        ttk.Label(targetf, text="Channel:").grid(row=2, column=2, sticky="e", pady=(6, 0))
        self.pw_channel_var = tk.StringVar(value="Auto (Playwright Chromium)")
        self.pw_channel_combo = ttk.Combobox(
            targetf, state="readonly",
            values=list(PW_CHANNEL_DISPLAY_TO_VALUE.keys()),
            textvariable=self.pw_channel_var, width=26
        )
        self.pw_channel_combo.grid(row=2, column=3, sticky="w", padx=6, pady=(6, 0))

        self.btn_browser_help = ttk.Button(targetf, text="?", width=3, command=self._browser_help)
        self.btn_browser_help.grid(row=2, column=3, sticky="e", padx=(0, 6), pady=(6, 0))

        ttk.Label(targetf, text="Browser monitor:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.browser_monitor_var = tk.IntVar(value=0)
        self.browser_monitor_combo = ttk.Combobox(targetf, state="readonly", width=28)
        self.browser_monitor_combo.grid(row=3, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(targetf, text="Offset X/Y/W/H:").grid(row=3, column=2, sticky="e", pady=(6, 0))
        self.bx_var = tk.IntVar(value=50)
        self.by_var = tk.IntVar(value=50)
        self.bw_var = tk.IntVar(value=1400)
        self.bh_var = tk.IntVar(value=900)

        self.bx_entry = ttk.Entry(targetf, textvariable=self.bx_var, width=7)
        self.by_entry = ttk.Entry(targetf, textvariable=self.by_var, width=7)
        self.bw_entry = ttk.Entry(targetf, textvariable=self.bw_var, width=7)
        self.bh_entry = ttk.Entry(targetf, textvariable=self.bh_var, width=7)
        self.bx_entry.grid(row=3, column=3, sticky="w", padx=(6, 2), pady=(6, 0))
        self.by_entry.grid(row=3, column=3, sticky="w", padx=(70, 2), pady=(6, 0))
        self.bw_entry.grid(row=3, column=3, sticky="w", padx=(134, 2), pady=(6, 0))
        self.bh_entry.grid(row=3, column=3, sticky="w", padx=(198, 2), pady=(6, 0))

        self.btn_open_browser = ttk.Button(targetf, text="Open / Login Browser", command=self.open_login_browser)
        self.btn_open_browser.grid(row=4, column=0, padx=6, pady=(8, 0), sticky="w")

        self.btn_check_login = ttk.Button(targetf, text="Check Login", command=self.check_login_state)
        self.btn_check_login.grid(row=4, column=1, padx=6, pady=(8, 0), sticky="w")

        self.btn_reset_login = ttk.Button(targetf, text="Reset Login", command=self.reset_login_state)
        self.btn_reset_login.grid(row=4, column=2, padx=6, pady=(8, 0), sticky="w")

        # ---- OCR ----
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

        # ---- Run ----
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

        # ---- Mid pane ----
        mid = ttk.PanedWindow(root, orient="horizontal")
        mid.pack(fill="both", expand=True, pady=10)

        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=3)
        mid.add(right, weight=2)

        canvas_frame = ttk.LabelFrame(left, text="Screenshot (click to mark ROI / point)", padding=6)
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

        # ---- Tooltips (English) ----
        ToolTip(self.target_mode_combo,
                "Desktop/PyAutoGUI: clicks & types into any desktop app.\n"
                "Web/Playwright: types into ATLAS web without moving the OS mouse.\n"
                "Login is persisted in ~/.ocr_symbol_bot/pw_profile/")

        ToolTip(self.pw_browser_combo,
                "Choose the Playwright browser engine.\n"
                "Chromium is the base for Chrome/Edge; WebKit is Safari-like.")

        ToolTip(self.pw_channel_combo,
                "Chromium only: choose an installed browser (Chrome/Edge),\n"
                "or keep Auto to use Playwright Chromium.")

        ToolTip(self.browser_monitor_combo,
                "Monitor where the browser window should open.\n"
                "Position = monitor origin + Offset X/Y.")

        ToolTip(self.atlas_url_entry,
                "ATLAS start URL. Adjust if your workflow uses a different entry page.")

        ToolTip(self.pw_selector_entry,
                "Optional CSS selector for the symbol input field.\n"
                "Leave empty to auto-detect (recommended).")

        ToolTip(self.btn_open_browser,
                "Opens a Playwright-controlled browser.\n"
                "Log in once; the session will be saved.")

        ToolTip(self.btn_check_login,
                "Checks if the symbol input field is visible (heuristic = logged in).")

        ToolTip(self.btn_reset_login,
                "Clears the saved Playwright profile (forces fresh login).")

    # --- small help buttons ---
    def _mode_help(self):
        messagebox.showinfo("Mode help", info_text_playwright())

    def _browser_help(self):
        messagebox.showinfo("Browser help", info_text_chromium_channel())

    def _selector_help(self):
        messagebox.showinfo(
            "Input selector help",
            "This is optional.\n\n"
            "Leave it empty to let the app auto-detect the symbol input field.\n"
            "Only set it if auto-detection fails after ATLAS updates."
        )

    def _update_target_ui_state(self):
        display_mode = self.target_mode_var.get().strip()
        internal_mode = TARGET_MODE_DISPLAY_TO_VALUE.get(display_mode, "pyautogui")
        is_pw = (internal_mode == "playwright")

        # Point selection is relevant for pyautogui only
        st_point = "normal" if not is_pw else "disabled"
        self.btn_point.configure(state=st_point)

        # Playwright controls
        st_pw = "normal" if is_pw else "disabled"
        self.atlas_url_entry.configure(state=st_pw)
        self.pw_selector_entry.configure(state=st_pw)
        self.pw_browser_combo.configure(state=st_pw)

        # channel only meaningful for chromium
        if is_pw:
            engine_value = PW_ENGINE_DISPLAY_TO_VALUE.get(self.pw_browser_var.get().strip(), "chromium")
            if engine_value == "chromium":
                self.pw_channel_combo.configure(state="readonly")
            else:
                self.pw_channel_combo.configure(state="disabled")
                self.pw_channel_var.set("Auto (Playwright Chromium)")
        else:
            self.pw_channel_combo.configure(state="disabled")

        self.browser_monitor_combo.configure(state=st_pw)
        self.bx_entry.configure(state=st_pw)
        self.by_entry.configure(state=st_pw)
        self.bw_entry.configure(state=st_pw)
        self.bh_entry.configure(state=st_pw)
        self.btn_open_browser.configure(state=st_pw)
        self.btn_check_login.configure(state=st_pw)
        self.btn_reset_login.configure(state=st_pw)

    def _refresh_monitor_dropdowns(self):
        self.monitors = list_monitors()
        vals = []
        for i, m in enumerate(self.monitors):
            if i == 0:
                vals.append("0: Virtual (All Monitors)")
            else:
                vals.append(f"{i}: left={m['left']} top={m['top']} {m['width']}x{m['height']}")
        self.monitor_combo["values"] = vals
        self.browser_monitor_combo["values"] = vals

        self.monitor_combo.current(self.cfg.monitor_index)
        self.browser_monitor_combo.current(self.cfg.browser_monitor_index)

        self.monitor_combo.bind("<<ComboboxSelected>>", self._on_monitor_changed)
        self.browser_monitor_combo.bind("<<ComboboxSelected>>", self._on_browser_monitor_changed)

    def _on_monitor_changed(self, _evt=None):
        self.cfg.monitor_index = int(self.monitor_combo.current())
        self.status_var.set(f"Monitor set to {self.cfg.monitor_index}. Capture a screenshot.")
        self.clear_selections()

    def _on_browser_monitor_changed(self, _evt=None):
        self.cfg.browser_monitor_index = int(self.browser_monitor_combo.current())
        self.status_var.set(f"Browser monitor set to {self.cfg.browser_monitor_index}. (Re-open browser to apply.)")

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

        self.status_var.set("Screenshot loaded. Select ROI (2 clicks) or input point (1 click).")

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

        display_mode = self.target_mode_var.get().strip()
        internal_mode = TARGET_MODE_DISPLAY_TO_VALUE.get(display_mode, "pyautogui")
        if internal_mode == "playwright":
            messagebox.showinfo("Info", "In Playwright mode you do not need a desktop click target.")
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

            # Convert friendly display values back to internal values
            display_mode = self.target_mode_var.get().strip()
            self.cfg.target_mode = TARGET_MODE_DISPLAY_TO_VALUE.get(display_mode, "pyautogui")

            self.cfg.atlas_url = self.atlas_url_var.get().strip()
            self.cfg.pw_input_selector = self.pw_selector_var.get().strip()

            display_engine = self.pw_browser_var.get().strip()
            self.cfg.pw_browser = PW_ENGINE_DISPLAY_TO_VALUE.get(display_engine, "chromium")

            display_channel = self.pw_channel_var.get().strip()
            self.cfg.pw_channel = PW_CHANNEL_DISPLAY_TO_VALUE.get(display_channel, "")

            self.cfg.browser_monitor_index = int(self.browser_monitor_combo.current())
            self.cfg.browser_x = int(self.bx_var.get())
            self.cfg.browser_y = int(self.by_var.get())
            self.cfg.browser_w = int(self.bw_var.get())
            self.cfg.browser_h = int(self.bh_var.get())
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

        if self.cfg.target_mode not in ("pyautogui", "playwright"):
            return "Target mode must be Desktop (PyAutoGUI) or Web (Playwright)."

        if self.cfg.target_mode == "playwright":
            if not self.cfg.atlas_url:
                return "ATLAS URL is required for Playwright mode."
            if self.cfg.browser_w < 300 or self.cfg.browser_h < 300:
                return "Browser width/height too small."
            if self.cfg.pw_browser not in ("chromium", "firefox", "webkit"):
                return "Browser engine must be Chromium / Firefox / WebKit."
            if self.cfg.pw_browser != "chromium" and self.cfg.pw_channel.strip():
                # channel only valid for chromium
                self.cfg.pw_channel = ""

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

    # ---- Playwright actions from UI ----
    def open_login_browser(self):
        err = self.sync_cfg_from_ui(require_screenshot=False)
        if err:
            messagebox.showerror("Error", err)
            return
        if self.cfg.target_mode != "playwright":
            messagebox.showinfo("Info", "Switch mode to 'Web (ATLAS in browser) – Playwright' first.")
            return

        def work():
            ok, msg = self.pw.ensure_started(self.cfg)
            self.after(0, self.status_var.set, msg)
            if not ok:
                return
            ok2, msg2 = self.pw.open_atlas()
            self.after(0, self.status_var.set, msg2 if ok2 else f"Open failed: {msg2}")
            self.after(0, self.status_var.set, "Browser opened. Please log in. Waiting for login detection...")
            ok3, msg3 = self.pw.wait_for_login(self.cfg, timeout_s=180.0)
            self.after(0, self.status_var.set, msg3 if ok3 else f"Login not detected: {msg3}")

        threading.Thread(target=work, daemon=True).start()

    def check_login_state(self):
        err = self.sync_cfg_from_ui(require_screenshot=False)
        if err:
            messagebox.showerror("Error", err)
            return
        if self.cfg.target_mode != "playwright":
            return

        def work():
            ok, msg = self.pw.ensure_started(self.cfg)
            if not ok:
                self.after(0, self.status_var.set, msg)
                return
            li, li_msg = self.pw.is_logged_in(self.cfg)
            self.after(0, self.status_var.set, "Logged in." if li else f"Not logged in (yet): {li_msg}")

        threading.Thread(target=work, daemon=True).start()

    def reset_login_state(self):
        if not messagebox.askyesno("Reset Login", "This will clear the saved browser session. Continue?"):
            return
        try:
            self.pw.reset_login()
            self.status_var.set("Login reset. Use 'Open / Login Browser' to authenticate again.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---- Run loop ----
    def start(self):
        err = self.sync_cfg_from_ui(require_screenshot=True)
        if err:
            messagebox.showerror("Error", err)
            return

        if self.cfg.target_mode == "pyautogui":
            if self.cfg.click_abs_x == 0 and self.cfg.click_abs_y == 0:
                if not messagebox.askyesno("Warning", "Click target is (0,0). Start anyway?"):
                    return
        else:
            ok, msg = self.pw.ensure_started(self.cfg)
            if not ok:
                messagebox.showerror("Playwright", msg)
                return
            li, li_msg = self.pw.is_logged_in(self.cfg)
            if not li:
                if not messagebox.askyesno("Login required", "Not logged in yet. Open browser for login now?"):
                    return
                self.open_login_browser()
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

    def _send_symbol(self, sym: str):
        if self.cfg.target_mode == "pyautogui":
            type_symbol_into_target_pyautogui(self.cfg, sym)
            return True, "Typed via PyAutoGUI."
        else:
            ok, msg = self.pw.send_symbol(self.cfg, sym)
            return ok, msg

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
                        self.after(0, self.status_var.set, f"Sending symbol: {sym}")
                        ok, msg = self._send_symbol(sym)
                        if ok:
                            self._last_sent = sym
                            self.after(0, self.status_var.set, f"Sent: {sym} ({msg})")
                        else:
                            self.after(0, self.status_var.set, f"Send failed: {msg}")
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

        self._refresh_monitor_dropdowns()

        if 0 <= self.cfg.monitor_index < len(self.monitors):
            self.mon_info = self.monitors[self.cfg.monitor_index]
            self.monitor_combo.current(self.cfg.monitor_index)
        else:
            self.mon_info = None
            self.cfg.monitor_index = 0
            self.monitor_combo.current(0)

        if 0 <= self.cfg.browser_monitor_index < len(self.monitors):
            self.browser_monitor_combo.current(self.cfg.browser_monitor_index)
        else:
            self.cfg.browser_monitor_index = 0
            self.browser_monitor_combo.current(0)

        self._apply_cfg_to_ui()
        self._sync_visual_points_from_cfg()
        self.redraw_selections()
        self._update_target_ui_state()
        self.status_var.set(f"Config loaded: {path}. (You may need to capture a new screenshot.)")


if __name__ == "__main__":
    enable_dpi_awareness()
    set_windows_appusermodelid(APP_USER_MODEL_ID)
    App().mainloop()
