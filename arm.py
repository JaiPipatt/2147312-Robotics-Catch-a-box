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
        self.velocity = 0.5  # m/s
        self.acceleration = 0.5  # m/s^2

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

    def _query_boxbox(self):
        """Request one coordinate sample from local boxbox_yolo.py vision server."""
        try:
            with socket.create_connection(("localhost", self.vision_port), timeout=1.0) as vision_sock:
                vision_sock.send(b'cap!')
                return vision_sock.recv(255).decode("utf-8").strip()
        except Exception:
            return None

    def move_rel(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, v: float = None, a: float = None, wait: bool = True, timeout: float = 10.0, tol: float = 0.002) -> None: # in mm and degree
        if v is None:
            v = self.velocity
        if a is None:
            a = self.acceleration

        # convert deltas from mm/deg to m/rad
        dx, dy, dz = x / 1000.0, y / 1000.0, z / 1000.0
        drx, dry, drz = math.radians(rx), math.radians(ry), math.radians(rz)

        # Capture starting pose to compute target if available
        start_pose = self._read_actual_tcp_pose()
        target = None
        if start_pose is not None and len(start_pose) == 6:
            target = tuple(cur + delta for cur, delta in zip(start_pose, (dx, dy, dz, drx, dry, drz)))

        command = f"movel(pose_add(get_actual_tcp_pose(), p[{dx}, {dy}, {dz}, {drx}, {dry}, {drz}]), {v}, {a}, 0, 0)\n"
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
        # URScript expects meters and radians; inputs here are given in mm/deg from the rest of the script.
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
            # print(f"Current pose: {pose}, Target pose: {target}")
            if pose is not None:
                if all(abs(cur - goal) <= tol for cur, goal in zip(pose, target)):
                    return
            time.sleep(0.05)

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
        
        # Track position stability locally to avoid blocking calls to self.gripped()
        last_pos = -1
        stable_time = 0.0

        while wait:
            pos = self._get_position()
            
            if pos == -1:
                time.sleep(0.1)
                continue

            # print(f"Gripper position: {pos}")

            # Check 1: Did it reach the end without grabbing anything?
            if pos >= (255 - tol):
                print("Gripper closed but box not detected. Check alignment or gripper status.")
                return

            # Check 2: Has it stopped moving? (Meaning it grabbed the box)
            if pos == last_pos:
                stable_time += 0.1
                if stable_time >= 2.0:  # Position hasn't changed for 2 seconds
                    print("Box gripped successfully.")
                    return
            else:
                # Still moving, reset the stability timer
                stable_time = 0.0
                last_pos = pos

            time.sleep(0.1)

    def gripped(self) -> bool:
        """
        Standalone method to verify if the gripper is currently holding an object.
        Returns True if the position remains unchanged for 2 seconds.
        """
        start_pos = self._get_position()
        if start_pos == -1:
            return False
            
        start_time = time.time()

        # Check for 2 seconds
        while time.time() - start_time < 0.5:
            current_pos = self._get_position()
            
            # If the position changes at all, it's not securely gripping
            if current_pos != start_pos:
                return False
            if current_pos >= (180):  # If it's fully closed without gripping, also return False
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

    def get_coordinates(self):
        """
        Sends 'cap!' to boxbox_yolo.py.
        If box is FULL, returns (x_mm, y_mm, ceta_deg).
        If box is missing or PARTIAL, returns None.
        """
        try:
            response = self._query_boxbox()
            
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

    def hover(self, x_mm: float, y_mm: float, ceta_deg: float, vision_delay: float) -> bool:
        # 1. Apply Camera-to-Robot Offset
        offset_x = 100.0
        offset_y = -15.0    
        box_start_x = x_mm + offset_x
        box_start_y = y_mm + offset_y

        # 2. Set Heights
        box_height_mm = 130.0
        safety_margin = 100.0
        offset_z = -20.0 
        z_hover = box_height_mm + safety_margin + offset_z + 50 
        # z_catch = box_height_mm - 15.0 + offset_z + 50

        # 3. Predict Movement: "going in -x direction of 2 cm within 1 sec"
        belt_vx_mm_s = 0.0
        # belt_vx_mm_s = -20.0 # -2 cm/s
        belt_vy_mm_s = 0.0
        
        # Adjust 'travel_time' to use less time if possible, matching your rule
        travel_time = 2.0 
        total_time = travel_time + vision_delay

        target_hover_x = box_start_x + (belt_vx_mm_s * total_time)
        target_hover_y = box_start_y + (belt_vy_mm_s * total_time)

       

        # 4. "If there is tilting -> tilt the gripper"\
        if ceta_deg < 0:
            tilt_rz_deg = -90 + np.abs(ceta_deg)
        else:
            tilt_rz_deg = 90 - np.abs(ceta_deg)
        print(f"Calculated tilt angle: {tilt_rz_deg:.1f}° based on ceta={ceta_deg:.1f}°")


        print(f"[STATE 1] Detected box at ({x_mm:.1f}, {y_mm:.1f}), ceta={ceta_deg:.1f}°")
        print(f"[STATE 2] Applying offset: moving to ({target_hover_x:.1f}, {target_hover_y:.1f}) {tilt_rz_deg:.1f}° to hover above the box start position")
        # wait for me to press space to continue
        curremt_pose = self._read_actual_tcp_pose()
        print(f"current arm pose: {curremt_pose} will move to target hover pose: ({curremt_pose[0]+target_hover_x:.1f}, {curremt_pose[1]+target_hover_y:.1f}, {curremt_pose[2]+z_hover:.1f}) with tilt {curremt_pose[5]+tilt_rz_deg:.1f}°")

        # 5. Go hover above the box
        self.move_rel(target_hover_x, target_hover_y, 0, 
                      0, 0, tilt_rz_deg, wait=True)
        
        # # 6. Lower the gripper
        # print("[STATE 3] Lowering the gripper...")
        # old_v = self.velocity
        # self.velocity = 0.15 # Move down quickly
        # self.move_abs(target_hover_x, target_hover_y, z_catch, 
        #               self.cam_read_rot[0], self.cam_read_rot[1], tilt_rz_deg, wait=True)
        # self.velocity = old_v 
        return True
    
def main():
    my_arm = arm()  # Example belt speed
    try:
        while True:
            coor = my_arm.get_coordinates()
            if coor is not None:
                # get x and y from coordinates
                x, y = coor
                # move robot to the box
                my_arm.move(x, y, 0, 0, 0, 0)
                my_arm.move(0, -0.09, 0, 0, 0, 0)  # move down to hover above the box
                # hover above the box with same speed and acceleration of the box + close the gripper at the same time (multithreading)
                while my_arm.gripped() is False:
                    my_arm.gripper_close() + my_arm.move(0, y+0.01, 0, 0, 0, 0)  # keep hovering with the same speed and acceleration of the box

                    time.sleep(0.5)
                
                # check if the box is gripped, if not, pass and wait for the next coordinates from the vision system
                # if gripped, move the robot to the drop-off location and open the gripper to release the box then exit the loo
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping loop.")
    finally:
        my_arm.disconnect()
    
if __name__ == '__main__':
    my_arm = arm()

    # # test movement to start position
    # print('moving to start position...')
    my_arm.move_abs(my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2])


    # # Move to safe start position
    # print("Moving to cam start position...")
    # my_arm.go_to_start()

    # success = my_arm.hover_and_catch(0,0)

    # #my_arm.gripper_open()
    # time.sleep(0.5)

    #         # Return to start
    # my_arm.go_to_start()

    # test grip
    my_arm.go_to_start()
    # print("Testing gripper open/close...")
    # print("Opening gripper...")
    # my_arm.gripper_open(wait=True)
    # time.sleep(2)
    # print("Closing gripper...")
    # my_arm.gripper_close(wait=True)
    # while not my_arm.gripped():
    #     my_arm.gripper_open(wait=True)
    #     time.sleep(0.5)
    #     my_arm.gripper_close(wait=True)
    #     time.sleep(0.5)
    # if my_arm.gripped():
    #     print("Gripper is holding an object.")
    #     my_arm.move_abs(my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2])
    # print(my_arm.gripped())
 
    # my_arm = None
    # try:
    #     print("Initializing robot arm...")
    #     my_arm = arm()  # Example belt speed
    #     print("Connection test passed:", my_arm.test_connection())
    #     print("Moving to start position...")
    #     my_arm.go_to_start()
    #     print("Starting main loop. Press Ctrl+C to exit.")

    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received, shutting down.")
    # finally:
    #     if my_arm:
    #         my_arm.disconnect()
