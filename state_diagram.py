'''
State 0
- Move to initial position 116mm, -300mm, 200mm, 0rad, -3.143rad, 0rad
- Turn on the camera 
- Open gripper
State 1
- Move to the position to detect box 116mm, -293mm, 461mm, 2.215rad, 2.226rad, 0rad
- Wait until see full box
    - If center of box is detected
        -  Save the x, y, ceta, vx
        - Stop using the camera (still open the camera)
        - Go to state 2
    - If cant detect
        - Still in state 1
State 2
- Using function hover by
    - Within 1 sec (or can adjust to use less time as possible) -> the box going in -x direction of 2 cm then that is the position that the gripper will go hover above the box within 1 sec
    - If there is tilting -> tilt the gripper
    - Go to state 3
State 3
- Lower the gripper 
- Close the gripper
- If can successfully grab
    - Lift it to the same level as the starting point
    - success!
- If not successfully
    - Open the gripper
    - Go to state 1
'''
import math
import socket
import time
import re
from enum import Enum
from rtde_receive import RTDEReceiveInterface

# ── 1. State Machine Enum ─────────────────────────────────────────────────────
class RobotState(Enum):
    STATE_0 = 0     # Initial position, Open gripper
    STATE_1 = 1     # Detect position, Wait for FULL box, Save data
    STATE_2 = 2     # Hover with -X prediction (2cm in 1sec) & Tilt
    STATE_3 = 3     # Lower, Close, Check Grab -> Lift (Success) OR Open & Retry (Fail)

# ── 2. Robot Arm Class ────────────────────────────────────────────────────────
class arm:
    def __init__(self, belt_speed=0.02):  # belt_speed in m/s
        self.belt_speed = belt_speed
        
        # Default connection parameters
        self.gripper_ip = "10.10.0.8"
        self.gripper_port = 63352
        self.robot_ip = "10.10.0.8"
        self.robot_port = 30003
        self.vision_ip = "10.10.0.14"
        self.vision_port = 2025
        
        # State 0 - Initial Start Pose
        self.start_pose = [116, -300, 200]  # mm
        self.start_rot = [0, math.degrees(-3.143), 0]  # degree
        
        # State 1 - Camera Read Pose
        self.cam_read_pose = [116, -293, 461]  # mm
        self.cam_read_rot = [math.degrees(2.215), math.degrees(2.226), 0]  # degree
        
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.velocity = 1.0  # m/s
        self.acceleration = 1.0  # m/s^2
        self.connect()

        # Initialize gripper
        self.g.send(b'GET ACT\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        if '1' in g_recv:
            print('Gripper Activated')
        self.g.send(b'GET POS\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        if g_recv:
            self.g.send(b'SET ACT 1\n')
            time.sleep(3)
            self.g.send(b'SET GTO 1\n')
            self.g.send(b'SET SPE 255\n')
            self.g.send(b'SET FOR 255\n')

    def connect(self) -> None:
        self.g = socket.create_connection((self.gripper_ip, self.gripper_port), timeout=10.0)
        self.r = socket.create_connection((self.robot_ip, self.robot_port), timeout=10.0)
        # Connect to the Vision Server (boxbox_yolo.py)
        self.v = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.v.connect((self.vision_ip, self.vision_port)) 

    def test_connection(self) -> bool:
        return (hasattr(self, 'g') and self.g is not None and 
                hasattr(self, 'r') and self.r is not None and 
                hasattr(self, 'v') and self.v is not None)

    def disconnect(self):
        try: self.g.close()
        except Exception: pass
        try: self.r.close()
        except Exception: pass
        try: self.v.close()
        except Exception: pass

    def move_abs(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, v: float = None, a: float = None, wait: bool = True, timeout: float = 10.0, tol: float = 0.002) -> None:
        if v is None: v = self.velocity
        if a is None: a = self.acceleration
        x_m, y_m, z_m = x / 1000.0, y / 1000.0, z / 1000.0
        rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
        
        command = f"movel(p[{x_m}, {y_m}, {z_m}, {rx_r}, {ry_r}, {rz_r}], {v}, {a})\n"
        self.r.send(command.encode("utf-8"))
        
        if not wait: return

        target = (x_m, y_m, z_m, rx_r, ry_r, rz_r)
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            pose = self._read_actual_tcp_pose()
            if pose is not None:
                if all(abs(cur - goal) <= tol for cur, goal in zip(pose, target)): return
            time.sleep(0.05)

    def _read_actual_tcp_pose(self):
        try:
            pose = self.rtde_r.getActualTCPPose()
            if pose and len(pose) == 6: return tuple(pose)
            return None
        except Exception:
            return None

    def _get_position(self) -> int:
        self.g.send(b'GET POS\n')
        try:
            g_recv = self.g.recv(10).decode('utf-8').strip()
            match = re.search(r'\d+', g_recv)
            if match: return int(match.group())
            return -1
        except Exception:
            return -1

    def gripper_open(self, wait: bool = False, tol: int = 5):
        self.g.send(b'SET POS 0\n')
        while wait:
            pos = self._get_position()
            if pos != -1 and pos <= tol: return
            time.sleep(0.1)

    def gripper_close(self, wait: bool = True, tol: int = 5):
        self.g.send(b'SET POS 255\n')
        last_pos = -1
        stable_time = 0.0
        while wait:
            pos = self._get_position()
            if pos == -1: continue
            if pos >= (255 - tol): return # Closed on nothing
            if pos == last_pos:
                stable_time += 0.1
                if stable_time >= 1.0: return
            else:
                stable_time = 0.0
                last_pos = pos
            time.sleep(0.1)

    def gripped(self) -> bool:
        start_pos = self._get_position()
        if start_pos == -1: return False
        start_time = time.time()
        while time.time() - start_time < 1.0:
            time.sleep(0.5)
            current_pos = self._get_position()
            if current_pos != start_pos: return False
            if current_pos >= 180: return False # Fully closed without gripping anything
        return True

    def get_coordinates(self):
        """
        Sends 'cap!' to boxbox_yolo.py.
        If box is FULL, returns (x_mm, y_mm, ceta_deg).
        If box is missing or PARTIAL, returns None.
        """
        try:
            self.v.send(b'cap!')
            response = self.v.recv(255).decode("utf-8").strip()
            
            # YOLO script sends "none" if the box is cut off by the edge (partial=True)
            if response == 'none' or len(response) == 0:
                return None
                
            parts = response.split(',')
            if len(parts) >= 3:
                x = float(parts[0]) 
                y = float(parts[1]) 
                ceta = float(parts[2])
                return (x, y, ceta)
            else:
                return None
        except Exception:
            return None

    def hover_and_catch(self, x_mm: float, y_mm: float, ceta_deg: float, vision_delay: float) -> bool:
        # 1. Apply Camera-to-Robot Offset
        offset_x = 183.3  
        offset_y = 0.0    
        box_start_x = x_mm + offset_x
        box_start_y = y_mm + offset_y

        # 2. Set Heights
        box_height_mm = 130.0
        safety_margin = 100.0
        offset_z = -20.0 
        z_hover = box_height_mm + safety_margin + offset_z + 50 
        z_catch = box_height_mm - 15.0 + offset_z + 50

        # 3. Predict Movement: "going in -x direction of 2 cm within 1 sec"
        belt_vx_mm_s = -20.0 # -2 cm/s
        belt_vy_mm_s = 0.0
        
        # Adjust 'travel_time' to use less time if possible, matching your rule
        travel_time = 1.0 
        total_time = travel_time + vision_delay

        target_hover_x = box_start_x + (belt_vx_mm_s * total_time)
        target_hover_y = box_start_y + (belt_vy_mm_s * total_time)

        # 4. "If there is tilting -> tilt the gripper"
        tilt_rz_deg = self.cam_read_rot[2] + ceta_deg

        print(f"[STATE 2] Predicting hover point: Moving {-20.0 * total_time:.1f} mm in -X direction")
        
        # 5. Go hover above the box
        self.move_abs(target_hover_x, target_hover_y, z_hover, 
                      self.cam_read_rot[0], self.cam_read_rot[1], tilt_rz_deg, wait=True)
        
        # 6. Lower the gripper
        print("[STATE 3] Lowering the gripper...")
        old_v = self.velocity
        self.velocity = 0.15 # Move down quickly
        self.move_abs(target_hover_x, target_hover_y, z_catch, 
                      self.cam_read_rot[0], self.cam_read_rot[1], tilt_rz_deg, wait=True)
        self.velocity = old_v 
        return True


# ── 3. Main Pipeline ──────────────────────────────────────────────────────────
def main_pipeline():
    my_arm = arm()
    
    if not my_arm.test_connection():
        print("Failed to connect to the robot, gripper, or vision server. Exiting.")
        return

    current_state = RobotState.STATE_0
    box_data = None
    vision_delay = 0.0

    try:
        while True:
            # =========================================================
            # STATE 0: Initialization
            # =========================================================
            if current_state == RobotState.STATE_0:
                print("\n[STATE 0] Initialization")
                print("-> Turn on the camera (YOLO server running)")
                
                print("-> Open gripper")
                my_arm.gripper_open(wait=True)
                
                print("-> Move to initial position")
                my_arm.move_abs(
                    my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], 
                    my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2], 
                    wait=True
                )
                current_state = RobotState.STATE_1

            # =========================================================
            # STATE 1: Detect Box
            # =========================================================
            elif current_state == RobotState.STATE_1:
                print("\n[STATE 1] Move to the position to detect box")
                my_arm.move_abs(
                    my_arm.cam_read_pose[0], my_arm.cam_read_pose[1], my_arm.cam_read_pose[2], 
                    my_arm.cam_read_rot[0], my_arm.cam_read_rot[1], my_arm.cam_read_rot[2], 
                    wait=True
                )
                
                print("-> Wait until see FULL box...")
                t_start = time.time()
                
                # If YOLO sees a partial box, get_coordinates() returns None
                coor = my_arm.get_coordinates()
                
                if coor is not None:
                    # If center of box is detected (and is fully in frame)
                    vision_delay = time.time() - t_start
                    
                    x, y, ceta = coor
                    vx = -2.0 # cm/s
                    
                    # Save the x, y, ceta, vx
                    box_data = (x, y, ceta, vx)
                    print(f"-> Box detected! Saved X:{x:.1f}, Y:{y:.1f}, Ceta:{ceta:.1f}, Vx:{vx}")
                    
                    # Stop using the camera / Go to state 2
                    print("-> Stop using the camera. Go to state 2")
                    current_state = RobotState.STATE_2
                else:
                    # If cant detect -> Still in state 1
                    time.sleep(0.05)

            # =========================================================
            # STATE 2: Predictive Hover
            # =========================================================
            elif current_state == RobotState.STATE_2:
                print("\n[STATE 2] Using function hover by...")
                x, y, ceta, vx = box_data
                
                # Using hover function (predicts -X direction and tilts gripper)
                my_arm.hover_and_catch(x, y, ceta, vision_delay)
                
                # Go to state 3
                current_state = RobotState.STATE_3

            # =========================================================
            # STATE 3: Grab and Evaluate
            # =========================================================
            elif current_state == RobotState.STATE_3:
                print("\n[STATE 3] Attempting to grab...")
                
                # Note: "Lower the gripper" is handled inside hover_and_catch
                
                # Close the gripper
                print("-> Close the gripper")
                my_arm.gripper_close(wait=True)
                
                # If can successfully grab
                if my_arm.gripped():
                    print("-> Success! Box grabbed.")
                    
                    # Lift it to the same level as the starting point
                    print("-> Lift it to the same level as the starting point")
                    my_arm.move_abs(
                        my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], 
                        my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2], 
                        wait=True
                    )
                    print("!!! SUCCESS! Demo Complete !!!")
                    break # Task finished
                    
                # If not successfully
                else:
                    print("-> If not successfully: Open the gripper")
                    my_arm.gripper_open(wait=True)
                    
                    # Go to state 1
                    print("-> Go to state 1")
                    current_state = RobotState.STATE_1

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        my_arm.disconnect()

if __name__ == '__main__':
    main_pipeline()