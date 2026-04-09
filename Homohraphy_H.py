import os
import cv2
import numpy as np


# Camera and pattern settings
CAM_INDEX = 1
GRID_COLS = 15
GRID_ROWS = 15
DOT_SPACING_MM = 10.0

# Files
CAMERA_PARAMS_PATH = "camera_params.npz"
CALIBRATION_OUT_PATH = "calibration.npz"


_mtx = None
_dist = None
_undistort_cache = {}


def load_camera_params(path: str = CAMERA_PARAMS_PATH) -> bool:
	global _mtx, _dist
	if not os.path.exists(path):
		print(f"[INFO] {path} not found. Running without undistortion.")
		return False
	cal = np.load(path)
	_mtx = cal["mtx"]
	_dist = cal["dist"]
	_undistort_cache.clear()
	print(f"[INFO] Loaded camera params from '{path}'.")
	return True


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
		out = cv2.resize(out[y:y + rh, x:x + rw], (w, h))
	return out


def build_object_points() -> np.ndarray:
	# World points in mm on the conveyor plane from a symmetric circles grid.
	# Origin is the top-left grid center, +X to the right, +Y downward in the image.
	obj = np.zeros((GRID_ROWS * GRID_COLS, 2), dtype=np.float32)
	obj[:, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2) * DOT_SPACING_MM
	return obj


def save_homography(H: np.ndarray, image_points: np.ndarray, world_points: np.ndarray) -> None:
	np.savez(
		CALIBRATION_OUT_PATH,
		H=H.astype(np.float32),
		image_points=image_points.astype(np.float32),
		world_points_mm=world_points.astype(np.float32),
		grid_cols=np.int32(GRID_COLS),
		grid_rows=np.int32(GRID_ROWS),
		dot_spacing_mm=np.float32(DOT_SPACING_MM),
	)
	print(f"[INFO] Saved homography to '{CALIBRATION_OUT_PATH}' (key: H).")


def main():
	print("=== Camera-plane Homography Calibration ===")
	print(f"Grid: {GRID_COLS}x{GRID_ROWS}, spacing: {DOT_SPACING_MM} mm")
	print("Place the dot board on the conveyor plane at the same setup used in runtime.")
	print("Keys: S=solve/save, Q=quit")

	load_camera_params(CAMERA_PARAMS_PATH)
	world_pts = build_object_points()

	cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

	win = "Homography from Dot Grid"
	cv2.namedWindow(win)

	last_centers = None
	solved_H = None

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				print("[ERROR] Failed to read frame from camera.")
				break

			frame = undistort(frame)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			found, centers = cv2.findCirclesGrid(
				gray,
				(GRID_COLS, GRID_ROWS),
				flags=cv2.CALIB_CB_SYMMETRIC_GRID,
			)

			vis = frame.copy()
			if found:
				last_centers = centers
				cv2.drawChessboardCorners(vis, (GRID_COLS, GRID_ROWS), centers, found)
				cv2.putText(
					vis,
					"Grid DETECTED - press S to save H",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(0, 255, 0),
					2,
				)
			else:
				cv2.putText(
					vis,
					"Grid NOT detected",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(0, 0, 255),
					2,
				)

			if solved_H is not None:
				cv2.putText(
					vis,
					"Saved calibration.npz",
					(10, 60),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(0, 255, 255),
					2,
				)

			cv2.imshow(win, vis)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break

			if key == ord("s"):
				if last_centers is None:
					print("[WARN] Grid not detected yet. Cannot solve H.")
					continue

				img_pts = last_centers.reshape(-1, 2).astype(np.float32)
				H, mask = cv2.findHomography(img_pts, world_pts, method=0)
				if H is None:
					print("[ERROR] findHomography failed.")
					continue

				# Reprojection quality check in mm
				proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
				err = np.linalg.norm(proj - world_pts, axis=1)
				print("[INFO] Homography solved.")
				print(H)
				print(f"[INFO] Mean reprojection error: {err.mean():.3f} mm")
				print(f"[INFO] Max reprojection error:  {err.max():.3f} mm")

				save_homography(H, img_pts, world_pts)
				solved_H = H

	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
