# 2147312 — Robotics: Catch a Box

## Project Overview
In this project, we control a **UR3 robot arm** to **detect and grab one box (out of two)** using a **camera** and a **gripper**.

- The boxes move on a conveyor belt at **~1 cm/s** to the left.
- The program must **recognize** the box and **grab it automatically** after the referee gives the start signal.

<img width="351" height="471" alt="image" src="https://github.com/user-attachments/assets/63fa66af-ba29-44c2-b8d4-6fe3a3c9a3a7" />

*Fig. 1: Robot arm grabbing a box from the conveyor belt*

---

## Setup & Initial Conditions

### Robot initial pose
The robot arm is initially placed at the midpoint of the conveyor belt.

Pose w.r.t. base (x, y, z, Rx, Ry, Rz):

- **x:** 116 mm  
- **y:** -300 mm  
- **z:** 200 mm  
- **Rx:** 0 rad  
- **Ry:** -3.143 rad  
- **Rz:** 0 rad  

### Box placement & orientation
- The **first box** starts on the **far right** of the belt with a **random position and orientation**.
- The orientation will not deviate from the belt axis by more than **±10°**.

<img width="289" height="219" alt="image" src="https://github.com/user-attachments/assets/2b9bb461-ca50-4bc6-b97d-6f97c5ced887" />

*Fig. 2: Nominal orientation of the boxes (top view)*

---

## Demo Rules
1. When the referee signals the start, the team runs the program.
2. The program must handle:
   - detecting the box
   - tracking it on the moving belt
   - grabbing it using the gripper
3. After the first box passes the midpoint of the conveyor belt, the **second box** will be placed at the far right.
4. If grabbing fails, the team may retry as long as:
   - the box has **not** reached the end of the belt.

### End condition
The demo ends when either:
- the box is successfully grabbed and lifted to at least the starting height, **or**
- both boxes reach the end of the belt.

---

## Schedule
- All teams will do a **live demo on April 24** (last day of class).  
  **This date cannot be postponed.**
- Before the demo, each team presents their work for **up to 10 minutes**.

---

## Report Requirements
The report must include:
- the full program flow (overall pipeline / logic)
- relevant calculations
- code (with comments)
- the work contributed by each team member

---

## Scoring (30 points total)
- **Successful grab:** 10 points  
- **Presentation:** 10 points  
- **Report:** 10 points  

### Extra points (speed ranking)
Extra points go to the teams with the shortest successful grab time:
- **1st:** +3 points  
- **2nd:** +2 points  
- **3rd:** +1 point  

**Important:** Grabbing must be **dynamic**:
- the robot hand must not wait idly for the box
- the grab must occur **before** the box passes the midpoint of the conveyor belt
