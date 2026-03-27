"""
camera image -> detect box (pixel position and orientation) -> convert pixel to 
camera coordinates -> transform camera to robot base coordinates -> predict motion -> send pick pose to robot
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#1. Detect box
img = cv2.imread(r'C:\Users\Asus\Downloads\Percogrobot_group\2147312-Robotics-Catch-a-box\test_img\box1_nocrop.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                        threshold=50, 
                        minLineLength=50, 
                        maxLineGap=10)

# Debug: Check if lines were detected
print(f"Lines detected: {len(lines) if lines is not None else 0}")

if lines is None or len(lines) == 0:
    print("No lines detected! Adjusting parameters...")
    # Try with less strict parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=30, 
                            minLineLength=30, 
                            maxLineGap=20)
    print(f"Lines detected with adjusted params: {len(lines) if lines is not None else 0}")

# choose longest line
best_line = None
max_len = 0

if lines is not None:
    # Filter lines: ignore edges (top 5%, bottom 5%, left 5%, right 5%)
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    for line in lines:
        x1,y1,x2,y2 = line[0]
        length = np.hypot(x2-x1, y2-y1)
        
        # Skip lines in top 5% / bottom 5%
        if (y1 > img_height * 0.95 and y2 > img_height * 0.95) or \
           (y1 < img_height * 0.05 and y2 < img_height * 0.05):
            print(f"Skipping horizontal edge: y1={y1}, y2={y2}")
            continue
        
        # Skip lines in left 5% / right 5%
        if (x1 < img_width * 0.05 and x2 < img_width * 0.05) or \
           (x1 > img_width * 0.95 and x2 > img_width * 0.95):
            print(f"Skipping vertical edge: x1={x1}, x2={x2}")
            continue
        
        # Calculate angle and prefer more horizontal edges (box lid edge)
        angle = np.arctan2(y2-y1, x2-x1)
        angle_deg = abs(np.degrees(angle))
        
        # Prefer lines that are more horizontal (0-45 degrees or 135-180 degrees)
        angle_score = min(angle_deg, 180 - angle_deg)  # 0-90 range
        
        # Longer lines with more horizontal angle are preferred
        score = length if angle_score < 45 else length * 0.5
        
        if score > max_len:
            max_len = score
            best_line = (x1,y1,x2,y2)
            print(f"New best line: length={length:.1f}, angle={np.degrees(angle):.1f}deg, x1={x1}, x2={x2}, y1={y1}, y2={y2}, score={score:.1f}")

if best_line is None:
    print("ERROR: No box line detected!")
    # Display edges for debugging
    plt.figure(figsize=(10, 8))
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection - No lines found")
    plt.show()
else:
    # compute angle and position
    x1,y1,x2,y2 = best_line
    angle = np.arctan2(y2-y1, x2-x1)

    # Calculate center position (pixel coordinates)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Convert angle to degrees
    angle_deg = np.degrees(angle)

    # Print results
    print(f"Box Position (pixels): ({center_x:.1f}, {center_y:.1f})")
    print(f"Box Orientation: {angle_deg:.2f} degrees")
    print(f"Line endpoints: ({x1}, {y1}) to ({x2}, {y2})")

    # Visualize with line drawn on image
    img_copy = img.copy()
    cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line
    cv2.circle(img_copy, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # Red center point

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Box Detection - Position: ({center_x:.1f}, {center_y:.1f}) | Orientation: {angle_deg:.2f} degrees")
    plt.axis('on')
    plt.tight_layout()
    plt.show()

