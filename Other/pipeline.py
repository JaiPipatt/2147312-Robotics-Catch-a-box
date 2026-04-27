import math
import socket
import time
import os
from enum import Enum
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rtde_receive import RTDEReceiveInterface

# ── Parameters ────────────────────────────────────────────────────────────────

blur_kernel       = (5, 5)
canny_low         = 10
canny_high        = 100

hough_threshold_1 = 25
hough_min_len_1   = 60
hough_gap_1       = 20

hough_threshold_2 = 20
hough_min_len_2   = 40
hough_gap_2       = 30

perp_tolerance    = 0.5
corner_proximity  = 20
border_margin     = 0.05

# ── Geometry helpers ──────────────────────────────────────────────────────────

def points_close(p1, p2, threshold=20):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1]) < threshold

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    return (x1 + t*(x2-x1), y1 + t*(y2-y1))

def _is_border_line(x1, y1, x2, y2, img_height, img_width):
    return (
        (y1 > img_height*(1-border_margin) and y2 > img_height*(1-border_margin)) or
        (y1 < img_height*border_margin      and y2 < img_height*border_margin)      or
        (x1 < img_width*border_margin       and x2 < img_width*border_margin)       or
        (x1 > img_width*(1-border_margin)   and x2 > img_width*(1-border_margin))
    )

# ── Detection pipeline ────────────────────────────────────────────────────────

def detect_edges(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    return cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

def detect_lines(edges):
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=hough_threshold_1,
        minLineLength=hough_min_len_1,
        maxLineGap=hough_gap_1
    )
    if lines is None or len(lines) == 0:
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=hough_threshold_2,
            minLineLength=hough_min_len_2,
            maxLineGap=hough_gap_2
        )
    return lines

def merge_parallel_connected_lines(line, lines, angle_tol=2, proximity=50):
    lx1, ly1, lx2, ly2 = line
    line_angle = np.arctan2(ly2-ly1, lx2-lx1)
    endpoints  = [(lx1, ly1), (lx2, ly2)]
    all_points = [(lx1, ly1), (lx2, ly2)]

    for candidate in lines:
        cx1, cy1, cx2, cy2 = candidate[0]

        if cx1 == lx1 and cy1 == ly1 and cx2 == lx2 and cy2 == ly2:
            continue

        cand_angle = np.arctan2(cy2-cy1, cx2-cx1)
        angle_diff = abs(np.degrees(cand_angle - line_angle)) % 180
        angle_diff = min(angle_diff, 180 - angle_diff)

        if angle_diff > angle_tol:
            continue

        for ep in endpoints:
            if points_close((cx1, cy1), ep, proximity) or points_close((cx2, cy2), ep, proximity):
                all_points.append((cx1, cy1))
                all_points.append((cx2, cy2))
                break

    if len(all_points) > 2:
        all_points = np.array(all_points)
        dx = np.cos(line_angle)
        dy = np.sin(line_angle)
        projections = [p[0]*dx + p[1]*dy for p in all_points]
        p_min = all_points[np.argmin(projections)]
        p_max = all_points[np.argmax(projections)]
        return (int(p_min[0]), int(p_min[1]), int(p_max[0]), int(p_max[1]))

    return line

def find_best_pair(lines, img_height, img_width):
    best_pair, best_score = None, -1

    for i, line_a in enumerate(lines):
        ax1, ay1, ax2, ay2 = line_a[0]
        if _is_border_line(ax1, ay1, ax2, ay2, img_height, img_width):
            continue
        len_a   = np.hypot(ax2-ax1, ay2-ay1)
        angle_a = np.arctan2(ay2-ay1, ax2-ax1)

        for line_b in lines[i+1:]:
            bx1, by1, bx2, by2 = line_b[0]
            if _is_border_line(bx1, by1, bx2, by2, img_height, img_width):
                continue
            len_b   = np.hypot(bx2-bx1, by2-by1)
            angle_b = np.arctan2(by2-by1, bx2-bx1)

            angle_diff = abs(np.degrees(angle_a - angle_b)) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)
            if abs(angle_diff - 90) > perp_tolerance:
                continue

            ep_pairs = [
                ((ax1, ay1), (bx1, by1)),
                ((ax1, ay1), (bx2, by2)),
                ((ax2, ay2), (bx1, by1)),
                ((ax2, ay2), (bx2, by2)),
            ]
            min_dist = min(np.hypot(p[0]-q[0], p[1]-q[1]) for p, q in ep_pairs)

            if min_dist > corner_proximity * 6:
                continue

            score = (len_a + len_b) / (1 + min_dist)
            if score > best_score:
                best_score = score
                best_pair  = (tuple(line_a[0]), tuple(line_b[0]))

    return best_pair

def detect(img_path_or_array, visualize=False, verbose=False):
    img = cv2.imread(img_path_or_array) if isinstance(img_path_or_array, str) else img_path_or_array
    if img is None:
        if verbose: print("ERROR: Could not load image!")
        return None

    img_height, img_width = img.shape[:2]
    edges = detect_edges(img)
    lines = detect_lines(edges)

    if lines is None or len(lines) == 0:
        if verbose: print("ERROR: No lines detected!")
        return None

    pair = find_best_pair(lines, img_height, img_width)
    if pair is None:
        if verbose: print("ERROR: No perpendicular corner pair found!")
        return None

    best_line, best_perp_line = pair
    best_line      = merge_parallel_connected_lines(best_line,      lines)
    best_perp_line = merge_parallel_connected_lines(best_perp_line, lines)

    x1,  y1,  x2,  y2  = best_line
    px1, py1, px2, py2 = best_perp_line
    corner = line_intersection(best_line, best_perp_line)

    if corner is None:
        return None

    corner_x, corner_y = corner
    edge1_mid_x, edge1_mid_y = (x1+x2)/2,   (y1+y2)/2
    edge2_mid_x, edge2_mid_y = (px1+px2)/2, (py1+py2)/2
    dist1 = np.hypot(edge1_mid_x-corner_x, edge1_mid_y-corner_y)
    dist2 = np.hypot(edge2_mid_x-corner_x, edge2_mid_y-corner_y)

    dir1_x = (edge1_mid_x-corner_x) / dist1 if dist1 > 0 else 0
    dir1_y = (edge1_mid_y-corner_y) / dist1 if dist1 > 0 else 0
    dir2_x = (edge2_mid_x-corner_x) / dist2 if dist2 > 0 else 0
    dir2_y = (edge2_mid_y-corner_y) / dist2 if dist2 > 0 else 0

    center_x = corner_x + dir1_x*dist1 + dir2_x*dist2
    center_y = corner_y + dir1_y*dist1 + dir2_y*dist2

    edge1_length = np.hypot(x2-x1,   y2-y1)
    edge2_length = np.hypot(px2-px1, py2-py1)
    edge_ratio   = max(edge1_length, edge2_length) / min(edge1_length, edge2_length)
    box_state    = 'stand' if edge_ratio > 1.35 else 'lie'
    angle_deg    = -np.degrees(np.arctan2(y2-y1, x2-x1)) if edge1_length < edge2_length else -np.degrees(np.arctan2(py2-py1, px2-px1))

    result = {
        'corner':       (corner_x, corner_y),
        'center':       (center_x, center_y),
        'orientation':  angle_deg,
        'edge1_length': edge1_length,
        'edge2_length': edge2_length,
        'best_line':    best_line,
        'perp_line':    best_perp_line,
        'img':          img,
        'all_lines':    lines,
        'box_state':    box_state,
    }

    if visualize:
        visualize_result(result)

    return result

def visualize_result(result):
    img      = result['img']
    img_copy = img.copy()
    x1,  y1,  x2,  y2  = result['best_line']
    px1, py1, px2, py2 = result['perp_line']
    corner_x, corner_y = result['corner']
    center_x, center_y = result['center']
    angle_deg          = result['orientation']

    np.random.seed(42)
    for line in result['all_lines']:
        lx1, ly1, lx2, ly2 = line[0]
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        cv2.line(img_copy, (lx1, ly1), (lx2, ly2), color, 1)

    cv2.line(img_copy,   (x1,  y1),  (x2,  y2),  (0, 255, 0),   3)
    cv2.line(img_copy,   (px1, py1), (px2, py2), (255, 0, 255),  3)
    cv2.circle(img_copy, (int(corner_x), int(corner_y)),          8, (255, 0,   0),   -1)
    cv2.circle(img_copy, (int((x1+x2)/2),   int((y1+y2)/2)),      5, (0,   0,   255), -1)
    cv2.circle(img_copy, (int((px1+px2)/2), int((py1+py2)/2)),    5, (0,   0,   255), -1)
    cv2.circle(img_copy, (int(center_x), int(center_y)),          6, (0,   255, 255), -1)

    edges = detect_edges(img)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis('on')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny edges")
    plt.axis('on')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Corner ({corner_x:.1f}, {corner_y:.1f})  |  Center ({center_x:.1f}, {center_y:.1f})  |  {angle_deg:.1f} deg")
    plt.axis('on')
    plt.tight_layout()
    plt.show()

# ── Robot Control Class ───────────────────────────────────────────────────────

class arm:
    def __init__(self, belt_speed=0.02):  # belt_speed in m/s
        self.belt_speed = belt_speed
        self.gripper_ip = "10.10.0.61"
        self.gripper_port = 63352
        self.robot_ip = "10.10.0.61"
        self.robot_port = 30003
        self.vision_ip = "10.10.0.14"
        self.vision_port = 2025
        self.start_pose = [116, -300, 200]  # mm
        self.start_rot = [0, -180, 0]  # degree
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.cam_read_pose = [116, -307, 319.5]  # mm
        self.cam_read_rot = [127, 127, 0]  # degree
        self.velocity = 0.01  # m/s
        self.acceleration = 0.05  # m/s^2
        self.conveyer_speed = 0.02  # m/s
        self.connect()

        # init gripper
        self.g.send(b'GET ACT\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        if '1' in g_recv :
            print ('Gripper Activated')
        print ('get ACT  == ' + g_recv)
        self.g.send(b'GET POS\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        if g_recv :
            self.g.send(b'SET ACT 1\n')
            g_recv = str(self.g.recv(255), 'UTF-8')
            print (g_recv)
            time.sleep(3)
            self.g.send(b'SET GTO 1\n')
            self.g.send(b'SET SPE 255\n')
            self.g.send(b'SET FOR 255\n')

    def go_to_start(self):
        self.move_abs(self.cam_read_pose[0], self.cam_read_pose[1], self.cam_read_pose[2], self.cam_read_rot[0], self.cam_read_rot[1], self.cam_read_rot[2])

    def connect(self)->None:
        self.g = self._wait_for_connection(self.gripper_ip, self.gripper_port)
        self.r = self._wait_for_connection(self.robot_ip, self.robot_port)
        
        # NOTE: Uncomment these two lines if you are connecting to your vision PC via Socket! 
        # If you are running the `detect` logic locally on the same PC, leave them commented.
        # self.v = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.v.connect((self.vision_ip, self.vision_port)) 

    def test_connection(self)->bool:
        return (
            hasattr(self, 'g') and self.g is not None and
            hasattr(self, 'r') and self.r is not None 
            # and hasattr(self, 'v') and self.v is not None
        )

    def disconnect(self):
        try:
            self.g.close()
        except Exception:
            pass
        try:
            self.r.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'v') and self.v is not None:
                self.v.close()
        except Exception:
            pass
        self.g = None
        self.r = None
        self.v = None

    def _wait_for_connection(self, ip: str, port: int, timeout: float = 10.0) -> socket.socket:
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            try:
                return socket.create_connection((ip, port), timeout=1.0)
            except OSError:
                if deadline and time.monotonic() >= deadline:
                    raise ConnectionError(f"Could not connect to {ip}:{port} within {timeout}s")

    def move_rel(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, v: float = None, a: float = None, wait: bool = True, timeout: float = 10.0, tol: float = 0.002) -> None:
        if v is None:
            v = self.velocity
        if a is None:
            a = self.acceleration

        dx, dy, dz = x / 1000.0, y / 1000.0, z / 1000.0
        drx, dry, drz = math.radians(rx), math.radians(ry), math.radians(rz)
        start_pose = self._read_actual_tcp_pose()
        target = None
        
        if start_pose is not None and len(start_pose) == 6:
            target = tuple(cur + delta for cur, delta in zip(start_pose, (dx, dy, dz, drx, dry, drz)))

        command = f"movel(pose_add(get_actual_tcp_pose(), p[{dx}, {dy}, {dz}, {drx}, {dry}, {drz}]), {v}, {a}, 2, 0)\n"
        self.r.send(command.encode("utf-8"))

        if not wait or target is None:
            return

        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            pose = self._read_actual_tcp_pose()
            if pose is not None:
                if all(abs(cur - goal) <= tol for cur, goal in zip(pose, target)):
                    return
            time.sleep(0.05)

    def move_abs(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, v: float = None, a: float = None, wait: bool = True, timeout: float = 10.0, tol: float = 0.002) -> None:
        if v is None:
            v = self.velocity
        if a is None:
            a = self.acceleration
        x_m, y_m, z_m = x / 1000.0, y / 1000.0, z / 1000.0
        rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
        
        command = f"movel(p[{x_m}, {y_m}, {z_m}, {rx_r}, {ry_r}, {rz_r}], {v}, {a})\n"
        self.r.send(command.encode("utf-8"))
        
        if not wait:
            return

        target = (x_m, y_m, z_m, rx_r, ry_r, rz_r)
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            pose = self._read_actual_tcp_pose()
            if pose is not None:
                if all(abs(cur - goal) <= tol for cur, goal in zip(pose, target)):
                    return
            time.sleep(0.05)

    def _read_actual_tcp_pose(self):
        try:
            pose = self.rtde_r.getActualTCPPose()
            if pose and len(pose) == 6:
                return tuple(pose)
            return None
        except Exception:
            return None
    
    def get_coordinates(self)->tuple[float, float]:
        # NOTE: If using the local detect() function, replace the socket logic here.
        # Assuming you are still using the socket approach based on your original get_coordinates:
        try:
            if hasattr(self, 'v') and self.v is not None:
                self.v.send(b'cap!')
                coor = self.v.recv(255)
                coor = coor.decode("utf-8").strip()
                parts = coor.split(',')
                if len(parts) < 2:
                    return None
                x = float(parts[0]) / 1000.0
                y = float(parts[1]) / 1000.0
                return x, y
            else:
                print("Vision socket not connected. Returning None.")
                return None
        except Exception:
            return None

    def gripper_open(self):
        self.g.send(b'SET POS 0\n')

    def gripper_close(self):
        self.g.send(b'SET POS 255\n')

    def gripped(self)->bool:
        self.g.send(b'GET POS\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        start_time = time.time()
        while True:
            time.sleep(0.5)
            self.g.send(b'GET POS\n')
            new_g_recv = str(self.g.recv(10), 'UTF-8')
            if new_g_recv != g_recv:
                return False
            g_recv = new_g_recv
            if time.time() - start_time > 2:
                return True

    def set_motion_params(self, speed: float = None, acceleration: float = None):
        if speed is not None:
            self.velocity = speed
        if acceleration is not None:
            self.acceleration = acceleration

    def hover_and_catch(self, init_x_m: float, init_y_m: float) -> bool:
        cam_x_mm = init_x_m * 1000.0
        cam_y_mm = init_y_m * 1000.0

        offset_x = 183.3  
        offset_y = 0.0    
        offset_z = -20.0  

        target_x_mm = cam_x_mm + offset_x
        target_y_mm = cam_y_mm + offset_y

        box_height_mm = 130.0
        safety_margin = 100.0

        z_hover_tcp = box_height_mm + safety_margin + offset_z + 50
        z_catch_tcp = box_height_mm - 15.0 + offset_z + 50

        print(f"Tracking... Hover Z={z_hover_tcp}mm, Catch Z={z_catch_tcp}mm")

        self.gripper_open()

        pose = self._read_actual_tcp_pose()
        cur_z_mm = pose[2] * 1000.0 if pose else 0.0

        dx = target_x_mm #tools reference not base frame
        dy = target_y_mm 
        dz = z_hover_tcp - cur_z_mm 

        print(f"Moving to Hover: dx: {dx}, dy: {dy}, dz: {dz}")
        self.move_rel(dx, dy, dz, 0, 0, 0, wait=True)

        pose = self._read_actual_tcp_pose()
        cur_z_mm = pose[2] * 1000.0 if pose else 0.0
        dz_down = z_catch_tcp - cur_z_mm

        print(f"Plunging down: dz: {dz_down}")
        self.move_rel(0, 0, dz_down, 0, 0, 0, wait=True)
        return True


# ── Main State Machine Pipeline ───────────────────────────────────────────────

class RobotState(Enum):
    START = 0
    STATE_1 = 1     # Camera Read Position & Detection
    STATE_2 = 2     # Hover on box position
    GRAB = 3        # Attempt Grab
    STATE_3 = 4     # Success / Done
    STATE_4 = 5     # Search / Recover (Left/Right)

def main_pipeline():
    my_arm = arm()
    
    if not my_arm.test_connection():
        print("Failed to connect to the robot/gripper. Exiting.")
        return

    current_state = RobotState.START
    box_coords = None

    try:
        while True:
            # ---------------------------------------------------------
            # START: Start Position
            # ---------------------------------------------------------
            if current_state == RobotState.START:
                print("[START] Opening gripper...")
                my_arm.gripper_open()
                
                print("[START] Moving to resting start position...")
                # Uses self.start_pose and self.start_rot
                my_arm.move_abs(
                    my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], 
                    my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2], 
                    wait=True
                )
                current_state = RobotState.STATE_1

            # ---------------------------------------------------------
            # STATE 1: Camera Read Position & Detection
            # ---------------------------------------------------------
            elif current_state == RobotState.STATE_1:
                print("[STATE 1] Moving to camera read position...")
                # Uses self.cam_read_pose and self.cam_read_rot
                my_arm.move_abs(
                    my_arm.cam_read_pose[0], my_arm.cam_read_pose[1], my_arm.cam_read_pose[2], 
                    my_arm.cam_read_rot[0], my_arm.cam_read_rot[1], my_arm.cam_read_rot[2], 
                    wait=True
                )
                
                print("[DETECTION] Looking for box...")
                coor = my_arm.get_coordinates()
                
                if coor is not None:
                    print(f"-> Success! Box found at {coor}. Moving to State 2.")
                    box_coords = coor
                    current_state = RobotState.STATE_2
                else:
                    time.sleep(0.1)

            # ---------------------------------------------------------
            # STATE 2: Hover on box position
            # ---------------------------------------------------------
            elif current_state == RobotState.STATE_2:
                print("[STATE 2] Hovering on box position...")
                box_x, box_y = box_coords
                
                my_arm.hover_and_catch(box_x, box_y)
                current_state = RobotState.GRAB

            # ---------------------------------------------------------
            # GRAB: Attempt to close gripper and evaluate
            # ---------------------------------------------------------
            elif current_state == RobotState.GRAB:
                print("[GRAB] Closing gripper...")
                my_arm.gripper_close()
                time.sleep(1.0) 
                
                if my_arm.gripped():
                    print("-> Grab Success! Moving to State 3.")
                    current_state = RobotState.STATE_3
                else:
                    print("-> Grab Fail! Moving to State 4.")
                    my_arm.gripper_open()
                    current_state = RobotState.STATE_4

            # ---------------------------------------------------------
            # STATE 3: DONE
            # ---------------------------------------------------------
            elif current_state == RobotState.STATE_3:
                print("[STATE 3] DONE!! Lifting box to starting height...")
                my_arm.move_rel(0, 0, 250, 0, 0, 0, wait=True)
                print("Demo complete.")
                break 

            # ---------------------------------------------------------
            # STATE 4: Recover / Search mode & Detection
            # ---------------------------------------------------------
            elif current_state == RobotState.STATE_4:
                print("[STATE 4] Sweeping conveyer to find box...")
                
                # You might need to adjust this sweeping coordinate later
                my_arm.move_abs(116, -500, 461, math.degrees(2.215), math.degrees(2.226), 0, wait=True)
                
                print("[DETECTION] Looking for box during recovery...")
                coor = my_arm.get_coordinates()
                
                if coor is not None:
                    print("-> Detect! Box found. Moving back to State 2.")
                    box_coords = coor
                    current_state = RobotState.STATE_2
                else:
                    print("-> Fail. Still looking...")
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping pipeline.")
    finally:
        my_arm.disconnect()
        print("Disconnected cleanly.")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main_pipeline()