"""
Color Detection Project
=======================
An interactive tool that identifies colors in an image using OpenCV + NumPy.

Features
--------
1. Click a pixel      -> shows its color NAME, RGB, HEX and HSV values.
2. HSV trackbars      -> tune Hue / Saturation / Value ranges and see a live
                         mask of every pixel that falls inside the range.
3. Presets            -> one key press selects red, orange, yellow, green,
                         blue or purple.
4. Headless mode      -> run without a window and save a result image
                         (useful for screenshots and testing).

Usage
-----
    python color_detect.py --image images/pic1.jpg
    python color_detect.py --image images/pic3.jpg --preset blue --headless --output out.png

Keyboard / mouse controls (interactive mode)
--------------------------------------------
    Left click   identify the color under the cursor
    Right click  set the HSV trackbars around the clicked color
    1-6          presets: red, orange, yellow, green, blue, purple
    0 or r       reset trackbars to the full range
    s            save a screenshot to the screenshots/ folder
    q or ESC     quit
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
WINDOW_MAIN = "Color Detection"
WINDOW_CONTROLS = "HSV Controls"
MAX_DISPLAY_W, MAX_DISPLAY_H = 480, 640   # size of each panel on screen
BAR_HEIGHT = 64                           # height of the info bar on top

# OpenCV HSV ranges: H = 0-179, S = 0-255, V = 0-255.
# Each preset is (H_min, H_max, S_min, S_max, V_min, V_max).
# Red wraps around the hue circle, so H_min > H_max means "wrap around".
PRESETS = {
    "red":    (170, 10, 120, 255, 70, 255),
    "orange": (11, 22, 120, 255, 100, 255),
    "yellow": (23, 35, 100, 255, 100, 255),
    "green":  (36, 85, 70, 255, 50, 255),
    "blue":   (90, 130, 100, 255, 50, 255),
    "purple": (131, 160, 60, 255, 50, 255),
}
PRESET_KEYS = {ord(str(i + 1)): name for i, name in enumerate(PRESETS)}
FULL_RANGE = (0, 179, 0, 255, 0, 255)
TRACKBARS = [("H min", 179), ("H max", 179), ("S min", 255),
             ("S max", 255), ("V min", 255), ("V max", 255)]


# --------------------------------------------------------------------------
# Color-name lookup
# --------------------------------------------------------------------------
def load_color_table(csv_path):
    """Read colors.csv and return (names, rgb_array).

    Each CSV row looks like:  id, "Color Name", #hex, R, G, B
    """
    names, rgb = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            try:
                rgb.append([int(row[3]), int(row[4]), int(row[5])])
            except ValueError:
                continue                      # skip a header or bad row
            names.append(row[1])
    if not names:
        raise ValueError(f"No colors could be read from {csv_path}")
    return names, np.array(rgb, dtype=np.int32)


def get_color_name(r, g, b, names, rgb_table):
    """Return the name of the closest color in the table.

    'Closest' = smallest Euclidean distance in RGB space. NumPy does all
    865 comparisons in one vectorised step, so there is no slow Python loop.
    """
    diff = rgb_table - np.array([r, g, b], dtype=np.int32)
    distances = np.einsum("ij,ij->i", diff, diff)   # squared distance
    return names[int(np.argmin(distances))]


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def fit_to_screen(img, max_w=MAX_DISPLAY_W, max_h=MAX_DISPLAY_H):
    """Shrink a big photo so it fits on screen (never enlarges it)."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def pixel_info(img_bgr, x, y, names, rgb_table):
    """Collect every representation of the pixel at (x, y)."""
    b, g, r = (int(v) for v in img_bgr[y, x])
    h, s, v = (int(c) for c in cv2.cvtColor(
        np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0])
    return {
        "x": x, "y": y,
        "rgb": (r, g, b),
        "hsv": (h, s, v),
        "hex": f"#{r:02x}{g:02x}{b:02x}",
        "name": get_color_name(r, g, b, names, rgb_table),
    }


def build_mask(hsv_img, hsv_range):
    """Return a binary mask (255 = pixel inside the range).

    If H_min > H_max the hue range wraps around 0 (needed for red), so we
    combine two ranges: [H_min..179] and [0..H_max].
    """
    h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
    if h_min <= h_max:
        return cv2.inRange(hsv_img, (h_min, s_min, v_min), (h_max, s_max, v_max))
    low = cv2.inRange(hsv_img, (0, s_min, v_min), (h_max, s_max, v_max))
    high = cv2.inRange(hsv_img, (h_min, s_min, v_min), (179, s_max, v_max))
    return cv2.bitwise_or(low, high)


def range_around_pixel(h, s, v, h_tol=10, s_tol=70, v_tol=70):
    """Make an HSV range centred on one pixel (used for right-click)."""
    h_min, h_max = h - h_tol, h + h_tol
    # Wrap hue around the 0/179 boundary instead of clipping it.
    h_min, h_max = h_min % 180, h_max % 180
    return (h_min, h_max,
            max(s - s_tol, 0), min(s + s_tol, 255),
            max(v - v_tol, 0), min(v + v_tol, 255))


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def put_label(img, text, org, scale=0.5, color=(255, 255, 255)):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def compose_view(display_bgr, mask, hsv_range, info=None, label=None):
    """Build the picture shown on screen / saved as a screenshot.

    Layout:  [ info bar                         ]
             [ original | mask | masked result  ]
    """
    result = cv2.bitwise_and(display_bgr, display_bgr, mask=mask)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    original = display_bgr.copy()
    if info is not None:                      # mark the clicked pixel
        cv2.drawMarker(original, (info["x"], info["y"]), (255, 255, 255),
                       cv2.MARKER_CROSS, 22, 3)
        cv2.drawMarker(original, (info["x"], info["y"]), (0, 0, 0),
                       cv2.MARKER_CROSS, 22, 1)
    put_label(original, "Original", (8, 20))
    put_label(mask_bgr, "Mask", (8, 20))
    put_label(result, "Detected", (8, 20))

    panels = cv2.hconcat([original, mask_bgr, result])

    # ---- info bar -------------------------------------------------------
    bar = np.full((BAR_HEIGHT, panels.shape[1], 3), 40, dtype=np.uint8)
    coverage = 100.0 * np.count_nonzero(mask) / mask.size
    h0, h1, s0, s1, v0, v1 = hsv_range
    line1 = (f"{label + ' | ' if label else ''}H {h0}-{h1}  S {s0}-{s1}  "
             f"V {v0}-{v1}  |  matched pixels: {coverage:.1f}%")
    put_label(bar, line1, (10, 22), 0.5)

    if info is not None:
        r, g, b = info["rgb"]
        # colour swatch of the clicked pixel
        cv2.rectangle(bar, (10, 32), (50, 56), (b, g, r), -1)
        cv2.rectangle(bar, (10, 32), (50, 56), (255, 255, 255), 1)
        line2 = (f"{info['name']}  RGB({r},{g},{b})  {info['hex']}  "
                 f"HSV{info['hsv']}  at ({info['x']},{info['y']})")
        put_label(bar, line2, (62, 50), 0.5)
    else:
        put_label(bar, "Left-click the image to identify a color",
                  (10, 50), 0.5, (200, 200, 200))
    return cv2.vconcat([bar, panels])


# --------------------------------------------------------------------------
# Interactive app
# --------------------------------------------------------------------------
class ColorDetectorApp:
    def __init__(self, display_bgr, names, rgb_table, screenshot_dir):
        self.display = display_bgr
        self.hsv = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2HSV)
        self.names, self.rgb_table = names, rgb_table
        self.screenshot_dir = screenshot_dir
        self.info = None          # last clicked pixel
        self.label = "Custom"

    # -- trackbar helpers --------------------------------------------------
    def read_range(self):
        return tuple(cv2.getTrackbarPos(n, WINDOW_CONTROLS) for n, _ in TRACKBARS)

    def set_range(self, hsv_range, label="Custom"):
        for (name, _), value in zip(TRACKBARS, hsv_range):
            cv2.setTrackbarPos(name, WINDOW_CONTROLS, int(value))
        self.label = label

    # -- mouse -------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        # The image is shown as a 3-panel strip; only the first panel
        # (the original) is clickable. Subtract the info-bar height.
        y -= BAR_HEIGHT
        h, w = self.display.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.info = pixel_info(self.display, x, y, self.names, self.rgb_table)
            print(f"[click] {self.info['name']}  RGB={self.info['rgb']}  "
                  f"HEX={self.info['hex']}  HSV={self.info['hsv']}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.info = pixel_info(self.display, x, y, self.names, self.rgb_table)
            self.set_range(range_around_pixel(*self.info["hsv"]),
                           f"Around {self.info['name']}")

    # -- main loop ---------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_MAIN, self.on_mouse)
        for (name, maximum), start in zip(TRACKBARS, FULL_RANGE):
            cv2.createTrackbar(name, WINDOW_CONTROLS, start, maximum, lambda _: None)
        # The controls window needs some pixels to show the sliders.
        strip = np.full((1, 420, 3), 40, dtype=np.uint8)
        cv2.imshow(WINDOW_CONTROLS, strip)

        print("Controls:" + __doc__.split("(interactive mode)")[1])
        while True:
            hsv_range = self.read_range()
            mask = build_mask(self.hsv, hsv_range)
            view = compose_view(self.display, mask, hsv_range, self.info, self.label)
            cv2.imshow(WINDOW_MAIN, view)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in PRESET_KEYS:
                name = PRESET_KEYS[key]
                self.set_range(PRESETS[name], name.capitalize())
            elif key in (ord("0"), ord("r")):
                self.set_range(FULL_RANGE, "Full range")
            elif key == ord("s"):
                os.makedirs(self.screenshot_dir, exist_ok=True)
                path = os.path.join(self.screenshot_dir,
                                    time.strftime("screenshot_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(path, view)
                print(f"[saved] {path}")
            # Closing a window with the X button also exits cleanly.
            if cv2.getWindowProperty(WINDOW_MAIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# Headless mode (no window) - for screenshots, tests and servers
# --------------------------------------------------------------------------
def run_headless(display_bgr, names, rgb_table, args):
    hsv_img = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2HSV)
    info, label = None, "Custom"

    if args.point:
        x, y = args.point
        h, w = display_bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            sys.exit(f"--point must be inside the displayed image (0-{w-1}, 0-{h-1})")
        info = pixel_info(display_bgr, x, y, names, rgb_table)
        print(f"Pixel ({x},{y}): {info['name']}  RGB={info['rgb']}  "
              f"HEX={info['hex']}  HSV={info['hsv']}")

    if args.preset:
        hsv_range, label = PRESETS[args.preset], args.preset.capitalize()
    elif info is not None:
        hsv_range, label = range_around_pixel(*info["hsv"]), f"Around {info['name']}"
    else:
        hsv_range = FULL_RANGE

    mask = build_mask(hsv_img, hsv_range)
    print(f"Range {label}: {hsv_range}  ->  "
          f"{100.0 * np.count_nonzero(mask) / mask.size:.1f}% of pixels matched")
    view = compose_view(display_bgr, mask, hsv_range, info, label)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.output, view)
    print(f"Saved {args.output}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_point(text):
    try:
        x, y = (int(v) for v in text.split(","))
        return x, y
    except ValueError:
        raise argparse.ArgumentTypeError("use the form X,Y  (for example 120,200)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Identify colors in an image with click-to-name and HSV trackbars.")
    p.add_argument("--image", required=True, help="path to the input image")
    p.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "colors.csv"),
                   help="color-name table (default: colors.csv next to this script)")
    p.add_argument("--headless", action="store_true",
                   help="no window: process once and save --output")
    p.add_argument("--preset", choices=PRESETS, help="(headless) HSV preset to apply")
    p.add_argument("--point", type=parse_point,
                   help="(headless) identify the pixel at X,Y of the displayed image")
    p.add_argument("--output", default="screenshots/result.png",
                   help="(headless) output file (default: screenshots/result.png)")
    p.add_argument("--screenshot-dir", default="screenshots",
                   help="folder used by the 's' key (default: screenshots)")
    return p.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: could not read image '{args.image}'. Check the path.")
    if not os.path.isfile(args.csv):
        sys.exit(f"Error: color table '{args.csv}' not found.")

    names, rgb_table = load_color_table(args.csv)
    display = fit_to_screen(img)
    print(f"Loaded {args.image} ({img.shape[1]}x{img.shape[0]}), "
          f"showing at {display.shape[1]}x{display.shape[0]}; "
          f"{len(names)} named colors loaded.")

    if args.headless:
        run_headless(display, names, rgb_table, args)
    else:
        ColorDetectorApp(display, names, rgb_table, args.screenshot_dir).run()


if __name__ == "__main__":
    main()
