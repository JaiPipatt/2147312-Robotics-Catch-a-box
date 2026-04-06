import cv2
import numpy as np

# --- Configuration ---
GRID_COLS = 15       # number of dots horizontally
GRID_ROWS = 15       # number of dots vertically
DOT_SPACING_MM = 10.0  # distance between dots in mm
MIN_CAPTURES = 10   # minimum images needed for calibration

# Prepare object points: (0,0,0), (10,0,0), (20,0,0), ...
objp = np.zeros((GRID_ROWS * GRID_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2) * DOT_SPACING_MM

objpoints = []  # 3D world points
imgpoints = []  # 2D image points

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera at index 1.")
    exit()

print(f"=== Camera Intrinsic Calibration ===")
print(f"Grid: {GRID_COLS}x{GRID_ROWS} dots, {DOT_SPACING_MM}mm spacing")
print(f"Hold the dot paper in front of the camera at different angles/distances.")
print(f"Press SPACE to capture when grid is detected (green dots shown).")
print(f"Press 'c' to compute after {MIN_CAPTURES}+ captures. Press 'q' to quit.\n")

img_size = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

    found, centers = cv2.findCirclesGrid(
        gray, (GRID_COLS, GRID_ROWS),
        flags=cv2.CALIB_CB_SYMMETRIC_GRID
    )

    if found:
        cv2.drawChessboardCorners(display, (GRID_COLS, GRID_ROWS), centers, found)
        status = "Grid DETECTED - press SPACE to capture"
        color = (0, 255, 0)
    else:
        status = "Grid not found"
        color = (0, 0, 255)

    cv2.putText(display, f"{status} | Captured: {len(objpoints)}/{MIN_CAPTURES}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Camera Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit without calibrating.")
        break
    elif key == ord(' '):
        if found:
            objpoints.append(objp)
            imgpoints.append(centers)
            img_size = gray.shape[::-1]
            print(f"  Captured {len(objpoints)}/{MIN_CAPTURES}")
        else:
            print("  Grid not detected in this frame, try again.")
    elif key == ord('c'):
        if len(objpoints) < MIN_CAPTURES:
            print(f"Need at least {MIN_CAPTURES} captures (have {len(objpoints)}).")
        else:
            print("\nComputing calibration...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None
            )

            # Compute reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
                total_error += err
            mean_error = total_error / len(objpoints)

            print(f"Mean reprojection error: {mean_error:.4f} px")
            print(f"\nCamera matrix:\n{mtx}")
            print(f"\nDistortion coefficients:\n{dist.ravel()}")

            np.savez("camera_params.npz", mtx=mtx, dist=dist)
            print("\nSaved to camera_params.npz")
            print("Use this to undistort frames before running detection.")
            break

cap.release()
cv2.destroyAllWindows()
