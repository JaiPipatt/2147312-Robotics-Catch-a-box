"""
boxbox_yolo.py — Pink box detection using YOLO + OpenCV color filtering
=======================================================================

Two detection modes (auto-selected):
  • YOLO mode   : uses a YOLOv8 model to locate boxes, then confirms
                  each detection contains pink pixels.
  • Color mode  : falls back to HSV pink-mask + contour when no model
                  is provided or ultralytics is not installed.

Orientation is always computed via cv2.minAreaRect on the pink contour.

Usage
-----
  # Run real-time with live HSV sliders (color mode):
      python boxbox_yolo.py

  # Run with a YOLO model:
      python boxbox_yolo.py --model best.pt

  # Collect labeled images for YOLO training:
      python boxbox_yolo.py --collect

Keys (during real-time):
  S — print current HSV values to copy into PINK_* constants
  Q — quit

Install ultralytics (once):
  .venv/Scripts/pip install ultralytics
"""

import argparse
import os
import socket
import threading
import time
import cv2
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────

CAM_INDEX    = 1          # USB camera index (use cv2.CAP_DSHOW on Windows)
VISION_PORT  = 2025       # TCP port the gripper script connects to
CONF_THRESH = 0.30       # YOLO confidence threshold

# Pink HSV range (OpenCV: H 0-179, S 0-255, V 0-255)
# These are defaults; the real-time slider window overrides them live.
PINK_H_LO, PINK_H_HI = 150, 173     # hue: magenta/pink
PINK_S_LO, PINK_S_HI = 34, 255      # saturation
PINK_V_LO, PINK_V_HI = 76, 255      # value / brightness

# Live HSV values — updated every frame from trackbars
_hsv_lo = np.array([PINK_H_LO, PINK_S_LO, PINK_V_LO])
_hsv_hi = np.array([PINK_H_HI, PINK_S_HI, PINK_V_HI])

# Minimum pink contour area (pixels²) to be considered a box
MIN_CONTOUR_AREA = 1500

# ROI exclusion (pixels from each edge, 0 = full frame)
roi_left   = 150
roi_right  = 170
roi_top    = 0
roi_bottom = 0

# Camera undistortion params (loaded from camera_params.npz if present)
_mtx  = None
_dist = None
_undistort_cache = {}
_center_mm_cache = {}

# Pixel-to-world homography (loaded from calibration.npz if present)
_H = None

# Latest detection result shared with the vision server thread
_latest_lock = threading.Lock()
_latest = {'valid': False, 'x_mm': 0.0, 'y_mm': 0.0, 'angle': 0.0}

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
SILENT_MODE = False


def _log(*args, force=False, **kwargs):
    """Print helper that can be silenced with --silent."""
    if force or not SILENT_MODE:
        print(*args, **kwargs)

# ── Vision server (responds to 'cap!' from the gripper script) ────────────────

def _vision_server(port=VISION_PORT):
    """
    Listens for TCP connections on `port`.
    Protocol: client sends b'cap!', server replies with b'x_mm,y_mm,angle_deg\n'
    or b'none\n' if no box is currently detected.
    Matches the existing cap! protocol used by main.py.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(5)
    _log(f"[Vision server] Listening on port {port}  (protocol: send 'cap!' -> receive 'x,y,angle')")

    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_client, args=(conn,), daemon=True).start()
        except Exception:
            break


def _handle_client(conn):
    try:
        data = conn.recv(64).decode('utf-8', errors='ignore').strip()
        if data == 'cap!':
            with _latest_lock:
                if _latest['valid']:
                    reply = f"{_latest['x_mm']:.2f},{_latest['y_mm']:.2f},{_latest['angle']:.2f}\n"
                else:
                    reply = "none\n"
            conn.sendall(reply.encode('utf-8'))
    except Exception:
        pass
    finally:
        conn.close()


def start_vision_server(port=VISION_PORT):
    t = threading.Thread(target=_vision_server, args=(port,), daemon=True)
    t.start()
    return t

# ── Camera params ─────────────────────────────────────────────────────────────

def load_camera_params(path="camera_params.npz"):
    global _mtx, _dist
    if not os.path.exists(path):
        _log(f"[INFO] {path} not found – running without undistortion.")
        return False
    cal   = np.load(path)
    _mtx  = cal["mtx"]
    _dist = cal["dist"]
    _undistort_cache.clear()
    _log(f"[INFO] Camera params loaded from '{path}'.")
    return True


def load_calibration(path="calibration.npz"):
    global _H
    if not os.path.exists(path):
        _log(f"[INFO] {path} not found – position will be in pixels only.")
        return False
    _H = np.load(path)["H"]
    _center_mm_cache.clear()
    _log(f"[INFO] Homography loaded from '{path}'.")
    return True


def px_to_mm(px, py):
    """Convert pixel coords to real-world mm using the homography matrix."""
    if _H is None:
        return None
    pt  = np.array([[[px, py]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def px_to_camera_centered_mm(px, py, img_w, img_h):
    """
    Convert pixel coords to camera-centered mm.
    Coordinate frame: origin at image center, +X up, +Y left.
    Therefore, moving down gives -X and moving left gives +Y.
    """
    point_mm = px_to_mm(px, py)
    if point_mm is None:
        return None

    key = (img_w, img_h)
    center_mm = _center_mm_cache.get(key)
    if center_mm is None:
        center_mm = px_to_mm(img_w * 0.5, img_h * 0.5)
        _center_mm_cache[key] = center_mm

    if center_mm is None:
        return None

    # Homography frame is +X right, +Y down. Rotate/sign-flip to requested frame.
    dx = point_mm[0] - center_mm[0]
    dy = point_mm[1] - center_mm[1]
    centered_x = -dy
    centered_y = -dx
    return centered_x, centered_y


def undistort(img):
    if _mtx is None:
        return img
    h, w = img.shape[:2]
    cache_key = (w, h)
    cached = _undistort_cache.get(cache_key)
    if cached is None:
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(_mtx, _dist, (w, h), 1, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(_mtx, _dist, None, new_mtx, (w, h), cv2.CV_16SC2)
        cached = (map1, map2, roi)
        _undistort_cache[cache_key] = cached
    map1, map2, roi = cached
    out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        out = cv2.resize(out[y:y+rh, x:x+rw], (w, h))
    return out

# ── YOLO loader (optional) ────────────────────────────────────────────────────

def try_load_yolo(model_path):
    """Return a YOLO model or None if ultralytics is not available."""
    if model_path is None or not os.path.exists(model_path):
        if model_path:
            _log(f"[WARNING] Model file not found: {model_path}")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        _log(f"[INFO] YOLO model loaded: {model_path}")
        return model
    except ImportError:
        _log("[WARNING] ultralytics not installed. Falling back to color mode.")
        _log("          Install with:  .venv/Scripts/pip install ultralytics")
        return None

# ── Pink color helpers ────────────────────────────────────────────────────────

def pink_mask(img):
    """Return binary mask of pink pixels using the current live HSV range."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _hsv_lo, _hsv_hi)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL, iterations=1)
    return mask


def pink_fraction(img, bbox=None):
    """Fraction of pink pixels in the image or in bbox (x1,y1,x2,y2)."""
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        img = img[y1:y2, x1:x2]
    if img.size == 0:
        return 0.0
    mask  = pink_mask(img)
    return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])


def contour_to_result(contour, offset_xy=(0, 0)):
    """
    Compute box center, angle, and dimensions from a contour.
    offset_xy is added back when the contour was found in an ROI sub-image.
    """
    rect  = cv2.minAreaRect(contour)   # ((cx,cy), (w,h), angle)
    box_pts = cv2.boxPoints(rect)
    box_pts = np.int32(box_pts)

    cx, cy  = rect[0]
    w, h    = rect[1]
    angle   = rect[2]  # OpenCV convention: -90..0 degrees

    # Normalise angle so the long axis is the reference
    if w < h:
        angle = angle + 90
    angle = -angle  # flip to match right-hand convention

    ox, oy  = offset_xy
    cx     += ox
    cy     += oy
    box_pts += np.array([ox, oy])

    return {
        'center':      (cx, cy),
        'orientation': angle,
        'box_pts':     box_pts,
    }


def roi_bounds(img_w, img_h):
    """Clamp ROI settings to valid image bounds."""
    x1 = max(0, min(roi_left, img_w))
    x2 = img_w - roi_right if roi_right > 0 else img_w
    y1 = max(0, min(roi_top, img_h))
    y2 = img_h - roi_bottom if roi_bottom > 0 else img_h
    x2 = max(x1, min(x2, img_w))
    y2 = max(y1, min(y2, img_h))
    return x1, x2, y1, y2


def box_fully_in_frame(box_pts, img_w, img_h, margin=10):
    """
    Return True only if all 4 corners of the rotated box are inside the frame
    with a safety margin to account for box thickness and rotation.
    
    Args:
        box_pts: 4 corner points of the rotated box
        img_w, img_h: Image width and height
        margin: Pixel buffer from frame edges (default 15px)
    """
    for x, y in box_pts:
        if x < margin or x >= img_w - margin or y < margin or y >= img_h - margin:
            return False
    return True

# ── Core detect function ──────────────────────────────────────────────────────

def detect(img, model=None):
    img   = undistort(img)
    img_h, img_w = img.shape[:2]

    x1, x2, y1, y2 = roi_bounds(img_w, img_h)
    roi = img[y1:y2, x1:x2]
    ox, oy = x1, y1

    # ── YOLO mode ──
    if model is not None:
        results = model(roi, verbose=False, conf=CONF_THRESH)
        best_box = None
        best_pink = -1

        for r in results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                frac = pink_fraction(roi, (bx1, by1, bx2, by2))
                if frac > best_pink:
                    best_pink = frac
                    best_box = (int(bx1), int(by1), int(bx2), int(by2))

        if best_box is not None and best_pink > 0.01:
            bx1, by1, bx2, by2 = best_box
            crop = roi[by1:by2, bx1:bx2]
            mask_crop = pink_mask(crop)
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > MIN_CONTOUR_AREA * 0.1:
                    res = contour_to_result(cnt, offset_xy=(ox + bx1, oy + by1))

                    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    full_mask[oy + by1:oy + by2, ox + bx1:ox + bx2] = mask_crop

                    cx, cy = res['center']

                    res.update({
                        'img': img.copy(),
                        'mask': full_mask,
                        'mode': 'yolo',
                        'yolo_box': (ox+bx1, oy+by1, ox+bx2, oy+by2),
                        'pos_mm': px_to_camera_centered_mm(cx, cy, img_w, img_h),
                        'partial': False
                    })
                    return res

    # ── Color mode ──
    mask = pink_mask(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None

    res = contour_to_result(cnt, offset_xy=(ox, oy))

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    cx, cy = res['center']

    res.update({
        'img': img.copy(),
        'mask': full_mask,
        'mode': 'color',
        'pos_mm': px_to_camera_centered_mm(cx, cy, img_w, img_h),
        'partial': False
    })

    return res

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_result(frame, result):
    cx, cy  = result['center']
    angle   = result['orientation']
    box_pts = result['box_pts']

    # Filled pink overlay on mask
    pink_overlay = np.zeros_like(frame)
    pink_overlay[result['mask'] > 0] = (180, 105, 255)
    frame[:] = cv2.addWeighted(frame, 1.0, pink_overlay, 0.35, 0)

    # Rotated bounding box — orange when partial, cyan when fully in frame
    box_color = (0, 200, 255) if not result.get('partial') else (0, 165, 255)
    cv2.polylines(frame, [box_pts], True, box_color, 2)

    # YOLO raw box (if present)
    if 'yolo_box' in result:
        x1, y1, x2, y2 = result['yolo_box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)

    # Center dot
    cv2.circle(frame, (int(cx), int(cy)), 7, (0, 255, 255), -1)

    # Orientation arrow
    length = 50
    rad = np.radians(-angle)
    ex  = int(cx + length * np.cos(rad))
    ey  = int(cy + length * np.sin(rad))
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), (0, 255, 0), 2, tipLength=0.3)

    # ROI border
    img_h, img_w = frame.shape[:2]
    rx1, rx2, ry1, ry2 = roi_bounds(img_w, img_h)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 180, 255), 1)

    # Text HUD
    pos_mm  = result.get('pos_mm')
    partial = result.get('partial', False)
    if partial:
        cv2.putText(frame, "PARTIAL - waiting for full box",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"Angle:  {angle:.1f} deg",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    elif pos_mm:
        cv2.putText(frame, f"Pos:    ({pos_mm[0]:.1f}, {pos_mm[1]:.1f}) mm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Pixel:  ({cx:.0f}, {cy:.0f})",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, f"Angle:  {angle:.1f} deg",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"Center: ({cx:.0f}, {cy:.0f}) px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle:  {angle:.1f} deg",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# ── Image collection helper (for YOLO training) ───────────────────────────────

def collect_images(cam_index=CAM_INDEX, save_dir="train_img"):
    """
    Press SPACE to save a frame for YOLO training data collection.
    Press 'Q' to quit.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {cam_index}")
        return

    count = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    _log(f"Collecting images into '{save_dir}'. SPACE=save, Q=quit. ({count} existing)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Saved: {count}  |  SPACE=capture  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            path = os.path.join(save_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            _log(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()

# ── Real-time detection loop (with live HSV sliders) ─────────────────────────

def run_realtime(cam_index=CAM_INDEX, model=None):
    global _hsv_lo, _hsv_hi

    WIN_DET  = "Pink Box Detection"
    WIN_MASK = "Pink Mask"

    cv2.namedWindow(WIN_DET)
    cv2.createTrackbar("H lo", WIN_DET, int(_hsv_lo[0]), 179, lambda _: None)
    cv2.createTrackbar("H hi", WIN_DET, int(_hsv_hi[0]), 179, lambda _: None)
    cv2.createTrackbar("S lo", WIN_DET, int(_hsv_lo[1]), 255, lambda _: None)
    cv2.createTrackbar("S hi", WIN_DET, int(_hsv_hi[1]), 255, lambda _: None)
    cv2.createTrackbar("V lo", WIN_DET, int(_hsv_lo[2]), 255, lambda _: None)
    cv2.createTrackbar("V hi", WIN_DET, int(_hsv_hi[2]), 255, lambda _: None)

    cv2.namedWindow(WIN_MASK)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {cam_index}.")
        return

    start_vision_server(VISION_PORT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_lo = cv2.getTrackbarPos("H lo", WIN_DET)
        h_hi = cv2.getTrackbarPos("H hi", WIN_DET)
        s_lo = cv2.getTrackbarPos("S lo", WIN_DET)
        s_hi = cv2.getTrackbarPos("S hi", WIN_DET)
        v_lo = cv2.getTrackbarPos("V lo", WIN_DET)
        v_hi = cv2.getTrackbarPos("V hi", WIN_DET)

        _hsv_lo = np.array([h_lo, s_lo, v_lo])
        _hsv_hi = np.array([h_hi, s_hi, v_hi])

        result = detect(frame, model=model)

        if result is not None:
            draw_result(frame, result)

            pos_mm = result.get('pos_mm')

            if pos_mm:
                with _latest_lock:
                    _latest['valid'] = True
                    _latest['x_mm'] = pos_mm[0]
                    _latest['y_mm'] = pos_mm[1]
                    _latest['angle'] = result['orientation']

            mask_bgr = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
            mask_bgr[result['mask'] > 0] = (180, 105, 255)
            cv2.imshow(WIN_MASK, mask_bgr)

        else:
            with _latest_lock:
                _latest['valid'] = False

            raw_mask = pink_mask(frame)
            mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
            mask_bgr[raw_mask > 0] = (180, 105, 255)
            cv2.imshow(WIN_MASK, mask_bgr)

        cv2.imshow(WIN_DET, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pink box detector (YOLO + OpenCV)")
    parser.add_argument("--model",   type=str, default=None,
                        help="Path to YOLOv8 .pt model file (optional)")
    parser.add_argument("--collect", action="store_true",
                        help="Run image collection mode for YOLO training")
    parser.add_argument("--cam",     type=int, default=CAM_INDEX,
                        help=f"Camera index (default {CAM_INDEX})")
    parser.add_argument("--silent",  action="store_true",
                        help="Suppress non-error console output")
    args = parser.parse_args()

    SILENT_MODE = args.silent

    load_camera_params("camera_params.npz")
    load_calibration("calibration.npz")

    if args.collect:
        collect_images(cam_index=args.cam)
    else:
        model = try_load_yolo(args.model)
        run_realtime(cam_index=args.cam, model=model)