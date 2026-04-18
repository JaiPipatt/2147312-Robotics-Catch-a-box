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
- Lift it to the same level as the starting point
- If can successfully grab
    - success!
- If not successfully
    - Open the gripper
    - Go to state 1
'''
import os
import socket
import subprocess
import sys
import time
from enum import Enum
from arm import arm
from conveyor import ConveyorController

VISION_HOST = "localhost"
VISION_PORT = 2025
VISION_START_TIMEOUT_S = 20.0
VISION_POLL_TIMEOUT_S = 0.15
STATE1_POLL_SLEEP_S = 0.02


def _vision_server_ready(host: str = VISION_HOST, port: int = VISION_PORT, timeout: float = 1.0) -> bool:
    """Return True if a cap! request can be served by the local vision server."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as vision_sock:
            vision_sock.sendall(b"cap!")
            reply = vision_sock.recv(255).decode("utf-8", errors="ignore").strip()
            return len(reply) > 0
    except OSError:
        return False


def _start_vision_process() -> subprocess.Popen:
    """Launch boxbox_yolo.py as a child process using the same Python interpreter."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "boxbox_yolo_trial.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Could not find vision script: {script_path}")
    return subprocess.Popen([sys.executable, script_path, "--silent"], cwd=script_dir)


def _ensure_vision_server_running(timeout: float = VISION_START_TIMEOUT_S):
    """
    Ensure local boxbox_yolo vision server is reachable.
    Returns a Popen handle only if this function started the process.
    """
    if _vision_server_ready():
        print("[VISION] Vision server already running.")
        return None

    print("[VISION] Starting boxbox_yolo.py...")
    proc = _start_vision_process()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if _vision_server_ready():
            print("[VISION] Vision server is ready.")
            return proc

        if proc.poll() is not None:
            raise RuntimeError(f"boxbox_yolo_trial.py exited early with code {proc.returncode}")

        time.sleep(0.2)

    try:
        proc.terminate()
    except Exception:
        pass
    raise TimeoutError(f"Vision server did not become ready within {timeout:.1f}s")


def _stop_vision_process(proc) -> None:
    """Stop auto-started vision process if it is still alive."""
    if proc is None or proc.poll() is not None:
        return

    print("[VISION] Stopping auto-started vision process...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

# ── 1. State Machine Enum ─────────────────────────────────────────────────────
class RobotState(Enum):
    STATE_0 = 0     # Initial position, Open gripper
    STATE_1 = 1     # Detect position, Wait for FULL box, Save data
    STATE_2 = 2     # Hover with -X prediction (2cm in 1sec) & Tilt
    STATE_3 = 3     # Lower, Close, Check Grab -> Lift (Success) OR Open & Retry (Fail)

# ── 2. Main Pipeline ──────────────────────────────────────────────────────────
def main_pipeline():
    my_arm = None
    vision_proc = None

    try:
        vision_proc = _ensure_vision_server_running()    
        my_arm = arm()

        if not my_arm.test_connection():
            print("Failed to connect to the robot, gripper, or vision server. Exiting.")
            return

        current_state = RobotState.STATE_0
        box_data = None
        vision_delay = 0.0
        state1_positioned = False
        state1_poll_count = 0

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
                state1_positioned = False
                current_state = RobotState.STATE_1
                input("Press Enter to continue...")

            # =========================================================
            # STATE 1: Detect Box
            # =========================================================
            elif current_state == RobotState.STATE_1:
                if not state1_positioned:
                    print("\n[STATE 1] Move to the position to detect box")
                    my_arm.move_abs(
                        my_arm.cam_read_pose[0], my_arm.cam_read_pose[1], my_arm.cam_read_pose[2], 
                        my_arm.cam_read_rot[0], my_arm.cam_read_rot[1], my_arm.cam_read_rot[2], 
                        wait=True
                    )
                    state1_positioned = True
                    state1_poll_count = 0
                    print("-> Wait until see FULL box...")

                t_start = time.monotonic()
                

            
                # If YOLO sees a partial box, get_coordinates() returns None
                coor = my_arm.get_coordinates(timeout_s=VISION_POLL_TIMEOUT_S)
                poll_elapsed = time.monotonic() - t_start
                
                if coor is not None:
                    # If center of box is detected (and is fully in frame)
                    # vision_delay = time.time() - t_start
                    vision_delay = poll_elapsed
                    
                    x, y, ceta = coor
                    vx = -2.0 # cm/s
                    
                    # Save the x, y, ceta, vx
                    box_data = (x, y, ceta, vx)
                    print(f"-> Box detected! Saved X:{x:.1f}, Y:{y:.1f}, Ceta:{ceta:.1f}, Vx:{vx}")
                    print(f"  Vision check: coor={coor}, elapsed={poll_elapsed:.3f}s")
                    
                    # Stop using the camera / Go to state 2
                    print("-> Stop using the camera. Go to state 2")
                    state1_positioned = False
                    current_state = RobotState.STATE_2
                else:
                    # If cant detect -> Still in state 1
                    state1_poll_count += 1
                    if state1_poll_count % 10 == 0:
                        print(f"  Vision polling... coor=None, last elapsed={poll_elapsed:.3f}s")
                    time.sleep(STATE1_POLL_SLEEP_S)

            # =========================================================
            # STATE 2: Predictive Hover
            # =========================================================
            elif current_state == RobotState.STATE_2:
                print("\n[STATE 2] Using function hover by...")
                x, y, ceta, vx = box_data

                # debug: print current arm pose
                current_pose = my_arm._read_actual_tcp_pose()
                print(f"  Current arm pose: {current_pose}")


                
                # Using hover function (predicts -X direction and tilts gripper)
                hover_start = time.monotonic()
                hover_ok = my_arm.hover(x, y, ceta, vision_delay, max_wait_s=3.0, belt_vx_mm_s=20.0)
                hover_elapsed = time.monotonic() - hover_start
                if hover_ok:
                    print(f"-> Hovering above the predicted box position with tilt applied in {hover_elapsed:.2f}s")
                else:
                    print(f"-> Hover wait timed out after {hover_elapsed:.2f}s; continuing to grasp")
                
                # Go to state 3
                current_state = RobotState.STATE_3

            # =========================================================
            # STATE 3: Grab and Evaluate
            # =========================================================
            elif current_state == RobotState.STATE_3:
                print("\n[STATE 3] Attempting to grab...")
                
                # my_arm.move_rel(0, 0, -115, 0, 0, 0, wait=True) # Move down a bit to ensure better grip
                
                # Close the gripper
                print("-> Close the gripper")
                my_arm.gripper_close(wait=True)
                
                # # If can successfully grab
                # if True:
                #     # time.sleep(0.5) # Wait a moment to ensure gripper has fully closed and box is secured
                #     print("-> Success! Box grabbed.")
                    

                #     # !!! UNFIX: CHANGE TO PLACE THE BOX AT SOME FIXED LOCATION AND GO BACK TO STATE 1 INSTEAD OF LIFTING UP THEN SUCCESS 
                #     # Lift it to the same level as the starting point
                #     print("-> Lift it to the same level as the starting point")
                # my_arm.move_abs(
                #     my_arm.start_pose[0], my_arm.start_pose[1], my_arm.start_pose[2], 
                #     my_arm.start_rot[0], my_arm.start_rot[1], my_arm.start_rot[2], 
                #     wait=True
                # )

                my_arm.move_rel( 0.0, 0.0, 35.0, 0.0, 0.0, 0.0, wait=True)
                print('-> moved to same level as starting point')
                time.sleep(0.5) # Wait a moment to ensure gripper has fully closed and box is secured
                if my_arm.box_gripped():
                    print("!!! SUCCESS! Demo Complete !!!")
                    # wait for user press space to continuee
                    input("Press Enter to continue...")
                    current_state = RobotState.STATE_0 # Task finished
                else:
                    print("-> Grab failed. Box not securely held.")
                    my_arm.gripper_open(wait=True) # Open the gripper to drop the box
                    print("-> retrying...")
                    current_state = RobotState.STATE_1 # Go back to state 1 to retry
                    state1_positioned = False # Force repositioning in state 1 for better vision next time
                    

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        if my_arm is not None:
            my_arm.disconnect()
        _stop_vision_process(vision_proc)

if __name__ == '__main__':
    # conveyor = ConveyorController()
    # conveyor.start_server()
    # conveyor.shutdown()
    main_pipeline()