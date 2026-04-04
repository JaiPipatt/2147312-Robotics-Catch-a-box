import math
import socket
import struct
import time
from rtde_receive import RTDEReceiveInterface




class arm:
    def __init__(self, belt_speed=0.02):  # belt_speed in m/s
        self.belt_speed = belt_speed
        # Default connection parameters (from lab setup)
        self.gripper_ip = "10.10.0.61"
        self.gripper_port = 63352
        self.robot_ip = "10.10.0.61"
        self.robot_port = 30003
        self.vision_ip = "10.10.0.14"
        self.vision_port = 2025
        self.start_pose = [116, -300, 200]  # mm
        self.start_rot = [0, -180, 0]  # degree
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.cam_read_pose = [116, -307, 319.5]  # mm, position to read camera coordinates from the vision system
        self.cam_read_rot = [127, 127, 0]  # degree
        # Default movement parameters
        self.velocity = 0.01  # m/s
        self.acceleration = 0.05  # m/s^2

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


    def go_to_start(self):
        self.move_abs(self.cam_read_pose[0], self.cam_read_pose[1], self.cam_read_pose[2], self.cam_read_rot[0], self.cam_read_rot[1], self.cam_read_rot[2])

    def connect(self)->None:
        self.g = self._wait_for_connection(self.gripper_ip, self.gripper_port)
        self.r = self._wait_for_connection(self.robot_ip, self.robot_port)

        # self.v = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.v.connect((self.vision_ip, self.vision_port)) 
    def test_connection(self)->bool:
        return (
            hasattr(self, 'g') and self.g is not None and
            hasattr(self, 'r') and self.r is not None #and
            # hasattr(self, 'v') and self.v is not None
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
            self.v.close()
        except Exception:
            pass
        self.g = None
        self.r = None
        self.v = None

    def _wait_for_connection(self, ip: str, port: int, timeout: float = 10.0) -> socket.socket:
        """Block until a TCP connection succeeds or timeout elapses (no time.sleep used)."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            try:
                return socket.create_connection((ip, port), timeout=1.0)
            except OSError:
                if deadline and time.monotonic() >= deadline:
                    raise ConnectionError(f"Could not connect to {ip}:{port} within {timeout}s")
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
    
    def get_coordinates(self)->tuple[float, float]: # we care only about x and y
        try:
            self.v.send(b'cap!')
            coor = self.v.recv(255)
            coor = coor.decode("utf-8").strip()
            parts = coor.split(',')
            if len(parts) < 2:
                return None
            x = float(parts[0]) / 1000.0
            y = float(parts[1]) / 1000.0
            return x, y
        except Exception:
            return None
    def gripper_open(self):
        self.g.send(b'SET POS 0\n')
    def gripper_close(self):
        self.g.send(b'SET POS 255\n')
    def gripped(self)->bool:
        self.g.send(b'GET POS\n')
        g_recv = str(self.g.recv(10), 'UTF-8')
        # if the position stays the same for a while, we can assume that the box is gripped, otherwise, it is not gripped
        start_time = time.time()
        while True:
            time.sleep(0.5)
            self.g.send(b'GET POS\n')
            new_g_recv = str(self.g.recv(10), 'UTF-8')
            if new_g_recv != g_recv:
                return False
            g_recv = new_g_recv
            # if the position is the same for 2 seconds, we can assume that the box is gripped
            if time.time() - start_time > 2:
                return True
    def set_motion_params(self, speed: float = None, acceleration: float = None):
        if speed is not None:
            self.velocity = speed
        if acceleration is not None:
            self.acceleration = acceleration

    def hover_and_catch(self, init_x_m: float, init_y_m: float) -> bool:
    
        # --- 1. Convert camera input (m → mm) ---
        cam_x_mm = init_x_m * 1000.0
        cam_y_mm = init_y_m * 1000.0

        # --- 2. Camera → TCP offset 
        offset_x = 183.3  # mm 
        offset_y = 0.0     
        offset_z = -20.0   # need to adjust based on camera height vs TCP***

        # Convert to robot (TCP) coordinates
        target_x_mm = cam_x_mm + offset_x
        target_y_mm = cam_y_mm + offset_y

        # --- 3. Heights ---
        box_height_mm = 130.0
        safety_margin = 100.0

        z_hover_tcp = box_height_mm + safety_margin + offset_z + 50
        z_catch_tcp = box_height_mm - 15.0 + offset_z + 50

        print(f"Tracking... Hover Z={z_hover_tcp}mm, Catch Z={z_catch_tcp}mm")

        # --- 4. Conveyor tracking ---
        belt_speed_mms = self.belt_speed * 1000.0

        # Trigger when TCP aligns with box
        trigger_grab_x = target_x_mm

        self.gripper_open()
        # start_time = time.time()

        # --- 5. Hover tracking loop ---

        # # Read current TCP pose
        pose = self._read_actual_tcp_pose()
        # if pose is None:
        #     continue

        cur_x_mm = pose[0] * 1000.0
        # cur_y_mm = pose[1] * 1000.0
        cur_z_mm = pose[2] * 1000.0
        # print(f"Current TCP pose: x={cur_x_mm:.1f}mm, y={cur_y_mm:.1f}mm, z={cur_z_mm:.1f}mm | Target X: {current_target_x_mm:.1f}mm")

        # Compute RELATIVE correction (tool-based)
        dx = target_x_mm 
        dy = target_y_mm 
        dz = z_hover_tcp - cur_z_mm

        print(f"dx: {dx}, dy: {dy}, dz: {dz}")
        self.move_rel(dx, dy, dz, 0, 0, 0, wait=True)

        # --- 6. Plunge ---
        pose = self._read_actual_tcp_pose()
        cur_z_mm = pose[2] * 1000.0

        dz_down = z_catch_tcp - cur_z_mm

        self.move_rel(0, 0, dz_down, 0, 0, 0, wait=True)

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

    # Move to safe start position
    print("Moving to start position...")
    my_arm.go_to_start()

    success = my_arm.hover_and_catch(0,0)

    #my_arm.gripper_open()
    time.sleep(0.5)

            # Return to start
    my_arm.go_to_start()


 
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
