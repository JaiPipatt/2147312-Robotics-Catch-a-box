import math
import socket
import struct
import time
from rtde_receive import RTDEReceiveInterface
import re
import numpy as np




class arm:
    def __init__(self, belt_speed=0.02):  # belt_speed in m/s
        self.belt_speed = belt_speed
        # Default connection parameters (from lab setup)
        self.gripper_ip = "10.10.0.14"
        self.gripper_port = 63352
        self.robot_ip = "10.10.0.14"
        self.robot_port = 30003
        self.vision_port = 2025  # local boxbox_yolo.py server port
        self.start_pose = [116, -300, 200]  # mm
        self.start_rot = [0, -180, 0]  # degree
        self.rtde_r = self.connect_rtde()
        self.cam_read_pose = [180, -292, 380]  # mm, position to read camera coordinates from the vision system
        self.cam_read_rot = [127, 127, 0]  # degree
        # Default movement parameters
        self.velocity = 5.0 # m/s
        self.acceleration = 5.0  # m/s^2

        # self.velocity = 0.5 # m/s
        # self.acceleration = 0.5  # m/s^2

        self.conveyer_speed = 0.02  # m/s
        # init conveyer to move at the same speed as the box

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

    def connect_rtde(self):
        try:
            rtde_r = RTDEReceiveInterface(self.robot_ip)
            
            # Optional: verify by calling something simple
            pose = rtde_r.getActualTCPPose()
            
            print("✅ RTDE Receive connected successfully")
            return rtde_r

        except Exception as e:
            print("❌ Failed to connect RTDE Receive")
            print(f"Error: {e}")
            return None

    def go_to_start(self):
        self.move_abs(self.cam_read_pose[0], self.cam_read_pose[1], self.cam_read_pose[2], self.cam_read_rot[0], self.cam_read_rot[1], self.cam_read_rot[2])

    def connect(self)->None:
        self.g = self._wait_for_connection(self.gripper_ip, self.gripper_port)
        self.r = self._wait_for_connection(self.robot_ip, self.robot_port)

    def test_connection(self)->bool:
        vision_reply = self._query_boxbox()
        return (
            hasattr(self, 'g') and self.g is not None and
            hasattr(self, 'r') and self.r is not None and
            vision_reply is not None
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
        self.g = None
        self.r = None

    def _wait_for_connection(self, ip: str, port: int, timeout: float = 10.0) -> socket.socket:
        """Block until a TCP connection succeeds or timeout elapses (no time.sleep used)."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            try:
                return socket.create_connection((ip, port), timeout=1.0)
            except OSError:
                if deadline and time.monotonic() >= deadline:
                    raise ConnectionError(f"Could not connect to {ip}:{port} within {timeout}s")

    def _query_boxbox(self, timeout_s: float = 1.0):
        """Request one coordinate sample from local boxbox_yolo.py vision server."""
        try:
            with socket.create_connection(("localhost", self.vision_port), timeout=timeout_s) as vision_sock:
                vision_sock.settimeout(timeout_s)
                vision_sock.sendall(b'cap!')
                return vision_sock.recv(255).decode("utf-8").strip()
        except Exception:
            return None

    def move_rel(self, x: float, y: float, z: float, rx: float, ry: float, rz: float,
             v: float = None, a: float = None, wait: bool = True,
             timeout: float = 10.0, tol: float = 0.002, tol_r: float = 0.01) -> bool:

        if v is None:
            v = self.velocity
        if a is None:
            a = self.acceleration

        dx, dy, dz = x / 1000.0, y / 1000.0, z / 1000.0
        drx, dry, drz = math.radians(rx), math.radians(ry), math.radians(rz)

        start_pose = self._read_actual_tcp_pose()
        target = None
        if start_pose is not None and len(start_pose) == 6:
            target = tuple(cur + delta for cur, delta in zip(
                start_pose, (dx, dy, dz, drx, dry, drz)
            ))

        command = f"movel(pose_add(get_actual_tcp_pose(), p[{dx}, {dy}, {dz}, {drx}, {dry}, {drz}]), {v}, {a}, 0, 0)\n"
        self.r.send(command.encode("utf-8"))

        if not wait or target is None:
            return True

        end_time = time.monotonic() + timeout
        last_pose = None

        while time.monotonic() < end_time:
            pose = self._read_actual_tcp_pose()
            if pose is not None:
                last_pose = pose

                # ✅ ONLY position check
                if all(abs(pose[i] - target[i]) <= tol for i in range(3)):
                    return True

            time.sleep(0.05)

        if last_pose is not None:
            pos_err = max(abs(last_pose[i] - target[i]) for i in range(3))
            print(f"[MOVE_REL] Wait timeout after {timeout:.2f}s (pos_err={pos_err:.4f})")
        else:
            print(f"[MOVE_REL] Wait timeout after {timeout:.2f}s (no RTDE pose read)")

        return False

    def move_abs(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, v: float = None, a: float = None, wait: bool = True, timeout: float = 10.0, tol: float = 0.002) -> bool:
        if v is None:
            v = self.velocity
        if a is None:
            a = self.acceleration
        # URScript expects meters and radians; inputs here are given in mm/deg from the rest of the script.
        x_m, y_m, z_m = x / 1000.0, y / 1000.0, z / 1000.0
        rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
        command = f"movel(p[{x_m}, {y_m}, {z_m}, {rx_r}, {ry_r}, {rz_r}], {v}, {a})\n"
        self.r.send(command.encode("utf-8"))
        if not wait:
            return True

        target = (x_m, y_m, z_m, rx_r, ry_r, rz_r)
        end_time = time.monotonic() + timeout
        last_pose = None
        while time.monotonic() < end_time:
            pose = self._read_actual_tcp_pose()
            # print(f"Current pose: {pose}, Target pose: {target}")
            if pose is not None:
                last_pose = pose
                if all(abs(cur - goal) <= tol for cur, goal in zip(pose, target)):
                    return True
            time.sleep(0.05)

        if last_pose is not None:
            max_err = max(abs(cur - goal) for cur, goal in zip(last_pose, target))
            print(f"[MOVE_ABS] Wait timeout after {timeout:.2f}s (max error {max_err:.4f})")
        else:
            print(f"[MOVE_ABS] Wait timeout after {timeout:.2f}s (no RTDE pose read)")
        return False

    def _read_actual_tcp_pose(self):
        """Best-effort read of the current TCP pose from the realtime stream (port 30003)."""
        try:
            pose = self.rtde_r.getActualTCPPose()  # returns [x, y, z, rx, ry, rz] in m/rad
            if pose and len(pose) == 6:
                return tuple(pose)
            return None
        except Exception:
            return None
    
    def _get_position(self) -> int:
        """
        Helper method to request, read, and parse the current gripper position.
        Returns the integer position, or -1 if the read fails.
        """
        self.g.send(b'GET POS\n')
        
        try:
            # Read and decode response
            g_recv = self.g.recv(10).decode('utf-8').strip()
            
            # Extract the first continuous block of digits
            match = re.search(r'\d+', g_recv)
            if match:
                return int(match.group())
            else:
                print(f"Invalid position response: {g_recv}")
                return -1
        except Exception as e:
            print(f"Error reading from gripper: {e}")
            return -1
    def gripper_open(self, wait: bool = False, tol: int = 5):
        self.g.send(b'SET POS 0\n')
        
        while wait:
            pos = self._get_position()
            
            if pos != -1:
                # print(f"Gripper position: {pos}")
                if pos <= tol:  # Target is 0
                    print("Gripper opened successfully.")
                    return
                    
            time.sleep(0.1)

    def gripper_close(self, wait: bool = True, tol: int = 5):
        self.g.send(b'SET POS 255\n')
        time.sleep(0.4)  # Short delay to allow command to take effect before checking position
        
        # # Track position stability locally to avoid blocking calls to self.gripped()
        # last_pos = -1
        # stable_time = 0.0

        # while wait:
        #     pos = self._get_position()
            
        #     if pos == -1:
        #         time.sleep(0.1)
        #         continue

        #     # print(f"Gripper position: {pos}")

        #     # Check 1: Did it reach the end without grabbing anything?
        #     if pos >= (255 - tol):
        #         print("Gripper closed but box not detected. Check alignment or gripper status.")
        #         return

        #     # Check 2: Has it stopped moving? (Meaning it grabbed the box)
        #     if pos == last_pos:
        #         stable_time += 0.1
        #         if stable_time >= 2.0:  # Position hasn't changed for 2 seconds
        #             print("Box gripped successfully.")
        #             return
        #     else:
        #         # Still moving, reset the stability timer
        #         stable_time = 0.0
        #         last_pos = pos

            # time.sleep(0.1)

    def box_gripped(self) -> bool:
        """
        Standalone method to verify if the gripper is currently holding an object.
        Returns True if the position remains unchanged for 2 seconds.
        """
        # start_pos = self._get_position()
        # if start_pos == -1:
            # return False
            
        start_time = time.time()

        # Check for 2 seconds
        while time.time() - start_time < 0.5:
            current_pos = self._get_position()
            print(f"Checking gripper position: {current_pos}")
            # If the position changes at all, it's not securely gripping
            # if current_pos != start_pos:
            #     return False
            if current_pos >= (180):  # If it's fully closed, also return False
                return False
        return True
    
    def gripped_box(self) -> bool:
        start_pos = self._get_position()
        if start_pos == -1:
            return False
            
        start_time = time.time()

        # if its not fully closed
        if start_pos < (220):
            return True
        return False
    
    def set_motion_params(self, speed: float = None, acceleration: float = None):
        if speed is not None:
            self.velocity = speed
        if acceleration is not None:
            self.acceleration = acceleration

    def get_coordinates(self, timeout_s: float = 0.2):
        """
        Sends 'cap!' to boxbox_yolo.py.
        If box is FULL, returns (x_mm, y_mm, ceta_deg).
        If box is missing or PARTIAL, returns None.
        """
        try:
            response = self._query_boxbox(timeout_s=timeout_s)
            
            # YOLO script sends "none" if the box is cut off by the edge (partial=True)
            if response is None or response == 'none' or len(response) == 0:
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

    import math

    def find_intercept_x_offset(self, x_rel, y_rel, z_rel, a_robot, v_robot, v_belt, t_delay):
        """
        Calculates the exact X-axis drift of the box during the vision delay and the robot's travel time.
        Returns the distance (offset) to add to the target X coordinate.
        
        Inputs:
        - x_rel, y_rel, z_rel: The relative distance the robot must travel to reach the box (in mm)
        - a_robot: Robot acceleration in mm/s^2
        - v_robot: Robot max velocity in mm/s
        - v_belt: Conveyor belt speed in mm/s
        - t_delay: Vision processing time in seconds
        """
        if v_belt <= 0:
            return 0.0

        # The initial distance based on the stale camera image
        current_travel_dist = math.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        # Accurate Kinematic Model: How long does it take the robot to move 'distance'?
        def time_to_travel(distance):
            if distance <= 0: return 0.0
            accel_dist = (v_robot**2) / a_robot
            if distance <= accel_dist:
                # Triangular profile (doesn't reach max speed)
                return 2.0 * math.sqrt(distance / a_robot)
            else:
                # Trapezoidal profile (hits max speed, cruises, decelerates)
                return (v_robot / a_robot) + (distance / v_robot)

        # Iterative Solver: 
        # Because the box moves while the robot moves, the target distance increases.
        # We iterate a few times to converge on the exact time and distance.
        t_travel = time_to_travel(current_travel_dist)
        
        for _ in range(3): # 5 iterations is more than enough for a slow conveyor to converge
            t_total = t_delay + t_travel
            
            # How far the box drifts in X during the total elapsed time
            drift_x = v_belt * t_total 
            
            # Recalculate the true distance the robot has to move, factoring in the drift
            current_travel_dist = math.sqrt((x_rel + drift_x)**2 + y_rel**2 + z_rel**2)
            
            # Recalculate travel time based on the new, slightly longer distance
            t_travel = time_to_travel(current_travel_dist)

        # Return the final converged distance the box moved
        # This is your offset.
        return v_belt * (t_delay + t_travel)

    # def hover(self, x_mm: float, y_mm: float, ceta_deg: float, vision_delay: float, max_wait_s: float = 3.0, belt_vx_mm_s: float = 0.0) -> bool:
    #     # 1. Apply Camera-to-Robot Offset
    #     offset_x = 85.0
    #     offset_y = 2.0    
    #     box_start_x = x_mm + offset_x
    #     box_start_y = y_mm + offset_y



    #     target_hover_x = box_start_x#+self.find_intercept_x_pos(box_start_x, box_start_y, 0, self.acceleration*1000, self.velocity*1000, belt_vx_mm_s, vision_delay)
    #     target_hover_y = box_start_y # + (belt_vy_mm_s * total_time)

    #     if ceta_deg < 0:
    #         tilt_rz_deg = -90 + np.abs(ceta_deg)
    #     else:
    #         tilt_rz_deg = 90 - np.abs(ceta_deg)
    #     print(f"Calculated tilt angle: {tilt_rz_deg:.1f}° based on ceta={ceta_deg:.1f}°")
    #     print(f" Pose from camera + offset: ({target_hover_x:.1f}, {target_hover_y:.1f}) {tilt_rz_deg:.1f}° to hover above the box start position")      
    #     current_pose = self._read_actual_tcp_pose()
    #     if current_pose is not None:
    #         print(f"current arm pose: {current_pose} will move to target hover pose: ({current_pose[0]+target_hover_x:.1f}, {current_pose[1]+target_hover_y:.1f}, {current_pose[2]:.1f}) with tilt {current_pose[5]+tilt_rz_deg:.1f}°")
    #     else:
    #         print("current arm pose unavailable, sending relative hover move anyway")

    #     # 5. Go hover above the box
    #     hover_start = time.monotonic()
    #     reached = self.move_rel(target_hover_x, target_hover_y, -215,
    #                             0, 0, tilt_rz_deg, wait=True, timeout=max_wait_s)
    #     hover_elapsed = time.monotonic() - hover_start
    #     if reached:
    #         print(f"[HOVER] Reached hover target in {hover_elapsed:.2f}s")
    #     else:
    #         print(f"[HOVER] Timed out after {hover_elapsed:.2f}s; continuing")
    
    #     return reached
    def hover(self, x_mm: float, y_mm: float, ceta_deg: float, vision_delay: float, max_wait_s: float = 3.0, belt_vx_mm_s: float = 20.0) -> bool:
        # 1. Apply Camera-to-Robot Offset
        offset_x = 60.0
        offset_y = 20.0    
        box_start_x = x_mm + offset_x
        box_start_y = y_mm + offset_y

        # Define the relative Z drop (you used -215 in your move_rel)
        relative_z = -215.0

        # Calculate the X offset caused by the moving belt
        x_intercept_offset = self.find_intercept_x_offset(
            x_rel=box_start_x, 
            y_rel=box_start_y, 
            z_rel=relative_z, 
            a_robot=self.acceleration * 1000, 
            v_robot=self.velocity * 1000, 
            v_belt=belt_vx_mm_s, 
            t_delay=vision_delay
        )

        target_hover_x = box_start_x - x_intercept_offset + 20
        target_hover_y = box_start_y 
        

        # if ceta_deg < 0:
        #     tilt_rz_deg = -90 + np.abs(ceta_deg)
        # else:
        #     tilt_rz_deg = 90 - np.abs(ceta_deg)
        if ceta_deg < 0:
            tilt_rz_deg = -90 + np.abs(ceta_deg)
        else:
            tilt_rz_deg = 90 - np.abs(ceta_deg)
        print(f"Calculated tilt angle: {tilt_rz_deg:.1f}° based on ceta={ceta_deg:.1f}°")
        print(f" Pose from camera + offset: ({target_hover_x:.1f}, {target_hover_y:.1f}) {tilt_rz_deg:.1f}° to hover above the box start position")      
        current_pose = self._read_actual_tcp_pose()
        if current_pose is not None:
            print(f"current arm pose: {current_pose} will move to target hover pose: ({current_pose[0]+target_hover_x:.1f}, {current_pose[1]+target_hover_y:.1f}, {current_pose[2]:.1f}) with tilt {current_pose[5]+tilt_rz_deg:.1f}°")
        else:
            print("current arm pose unavailable, sending relative hover move anyway")

        # 5. Go hover above the box
        hover_start = time.monotonic()
        reached = self.move_rel(target_hover_x, target_hover_y, -215,
                                0, 0, tilt_rz_deg, wait=True, timeout=max_wait_s)
        hover_elapsed = time.monotonic() - hover_start
        if reached:
            print(f"[HOVER] Reached hover target in {hover_elapsed:.2f}s")
        else:
            print(f"[HOVER] Timed out after {hover_elapsed:.2f}s; continuing")
    
        return reached
    
if __name__ == '__main__':
    my_arm = arm()
    my_arm.move_abs(my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2])
    my_arm.go_to_start()