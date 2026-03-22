# 2147312-Robotics-Catch-a-box
Project: Catch a box
• In this year, we will control the robot (UR3) to grab a box (out of
two) using the camera and the gripper. The boxes move slowly (1
cm/s) to the left by the conveyer belt. See the picture.
<img width="351" height="471" alt="image" src="https://github.com/user-attachments/assets/63fa66af-ba29-44c2-b8d4-6fe3a3c9a3a7" />
Fig. 1: Robot arm grabbing a box out of the conveyer belt
The robot arm is initially placed at the middle point of the
conveyer belt, where the exact coordinates (x,y,z,Rx,Ry,Rz) with respect
to the base is (116mm, -300mm, 200mm, 0rad, -3.143rad, 0rad). The first
box is initially placed properly on a belt at the far right with random
position and orientation. Actual orientation will not be deviated from the
conveyer belt axis more than ±10°. See picture 2.
<img width="289" height="219" alt="image" src="https://github.com/user-attachments/assets/2b9bb461-ca50-4bc6-b97d-6f97c5ced887" />
Fig. 2: Nominal orientation of the boxes viewing from the top
Once the referee signals the starting time, the team executes the
program. The program takes care of recognizing and grabbing the box
automatically. After the first box has passed the midpoint of the
conveyer belt, the second box will be placed at the far right. If the
grabbing fails, the team can retry grabbing again and again as long as the
boxes have not come to the end of the belt yet. The demo ends once the
box is grabbed and lifted up to at least the same height as the starting, or
the boxes have come to the end of the belt.
• All teams will do a live demo on April 24th, which is the last day of
the class. The day cannot be postponed.
• Before the demo, the team will present the work for 10 minutes
maximally.
• The report should include at least the flow of the whole program,
relevant calculation, and the code with comments. The report must
also state the work each member has done.
Scoring (30 points)
• Grab the box successfully (10 points)
• Presentation (10 points)
• Report (10 points)
• Extra points for the first place who spent the shortest time to grab
the box, the second, and the third place (3, 2, 1 point respectively).
Grabbing must be dynamic (the hand does not wait idly for the box
to come and grab) and must occur before the box has passed the
middle point of the conveyer belt.
