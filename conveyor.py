import socket
import time


class ConveyorController:
    def __init__(self, host='10.10.0.98', port=2002):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.addr = None

    def start_server(self):
        self.server.bind((self.host, self.port))
        print(f"Socket binded to {self.port}")
        self.server.listen(1)
        print("Socket is listening...")

        self.conn, self.addr = self.server.accept()
        print(f"Connected by {self.addr}")

    def send_cmd(self, cmd: str):
        if self.conn:
            self.conn.sendall(cmd.encode())
        else:
            raise RuntimeError("Connection not established")

    # ===== Conveyor Commands =====
    def activate_tcp(self):
        self.send_cmd('activate,tcp,0.0\n')
        time.sleep(1)

    def power_on(self):
        self.send_cmd('pwr_on,conv,0\n')
        time.sleep(1)

    def power_off(self):
        self.send_cmd('pwr_off,conv,0\n')
        time.sleep(1)

    def set_velocity(self, vel: float):
        self.send_cmd(f'set_vel,conv,{vel}\n')
        time.sleep(1)

    def jog_forward(self):
        self.send_cmd('jog_fwd,conv,0\n')

    def jog_backward(self):
        self.send_cmd('jog_bwd,conv,0\n')

    def stop(self):
        self.send_cmd('jog_stop,conv,0\n')

    # ===== High-level control =====
    def start_conveyor(self, velocity=50):
        self.activate_tcp()
        self.power_on()
        self.set_velocity(velocity)
        self.jog_forward()
        print("Conveyor running... Press Ctrl+C to stop")

    def shutdown(self):
        print("\nStopping conveyor...")
        self.stop()
        time.sleep(1)
        self.power_off()
        print("Conveyor stopped safely")

    def run_forever(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def close(self):
        if self.conn:
            self.conn.close()
        self.server.close()


# ===== Usage =====
if __name__ == "__main__":
    conveyor = ConveyorController()

    conveyor.start_server()
    conveyor.start_conveyor(velocity=50)

    conveyor.run_forever()
    conveyor.close()