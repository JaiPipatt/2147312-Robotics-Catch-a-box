import socket
import time


class arm:
    def __init__(self, belt_speed=0.02):  # belt_speed in m/s
        self.belt_speed = belt_speed
        # Default connection parameters (from lab setup)
        self.gripper_ip = "10.10.0.60"
        self.gripper_port = 63352
        self.robot_ip = "10.10.0.60"
        self.robot_port = 30003
        self.vision_ip = "10.10.1.60"
        self.vision_port = 2025
        self.start_pose = (116, -300, 200)  # mm
        self.start_rot = (0, -180, 0)  # degree
        
        # Default movement parameters
        self.velocity = 0.25  # m/s
        self.acceleration = 1.2  # m/s^2

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

    def connect(self)->None:
        self.g = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.g.connect((self.gripper_ip, self.gripper_port))
        self.r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.r.connect((self.robot_ip, self.robot_port))
        self.v = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.v.connect((self.vision_ip, self.vision_port)) 
    def test_connection(self)->bool:
        return (
            hasattr(self, 'g') and self.g is not None and
            hasattr(self, 'r') and self.r is not None and
            hasattr(self, 'v') and self.v is not None
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
    def move(self, x: float, y: float, z: float, rx: float, ry: float, rz: float, v: float = None, a: float = None) -> None:
        if v is None:
            v = self.velocity
        if a is None:
            a = self.acceleration
        command = f"movel(pose_add(get_actual_tcp_pose(), p[{x}, {y}, {z}, {rx}, {ry}, {rz}]), {v}, {a}, 2, 0)\n"
        self.r.send(command.encode("utf-8"))
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

def main():
    my_arm = arm()  # Example belt speed
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
    
if __name__ == '__main__':
    main()