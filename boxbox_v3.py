import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Detect box
img = cv2.imread(r'C:\Users\Asus\Downloads\Percogrobot_group\2147312-Robotics-Catch-a-box\test_img\box1_nocrop.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                        threshold=50,
                        minLineLength=50,
                        maxLineGap=10)

print(f"Lines detected: {len(lines) if lines is not None else 0}")

if lines is None or len(lines) == 0:
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=30,
                            minLineLength=30,
                            maxLineGap=20)

best_line = None
max_len = 0

if lines is not None:
    img_height, img_width = img.shape[:2]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2-x1, y2-y1)

        # skip image borders
        if (y1 > img_height * 0.95 and y2 > img_height * 0.95) or \
           (y1 < img_height * 0.05 and y2 < img_height * 0.05):
            continue

        if (x1 < img_width * 0.05 and x2 < img_width * 0.05) or \
           (x1 > img_width * 0.95 and x2 > img_width * 0.95):
            continue

        angle = np.arctan2(y2-y1, x2-x1)
        angle_deg = abs(np.degrees(angle))
        angle_score = min(angle_deg, 180 - angle_deg)

        score = length if angle_score < 45 else length * 0.5

        if score > max_len:
            max_len = score
            best_line = (x1, y1, x2, y2)

if best_line is None:
    print("ERROR: No box line detected!")
else:
    x1, y1, x2, y2 = best_line
    angle = np.arctan2(y2-y1, x2-x1)
    angle_deg = np.degrees(angle)

    # midpoint of detected edge
    edge_center_x = (x1 + x2) / 2
    edge_center_y = (y1 + y2) / 2

    # perpendicular direction
    nx = np.sin(angle)
    ny = -np.cos(angle)

    # estimated box width in pixels
    box_width_pixels = 200

    # shift to box center
    center_x = edge_center_x + nx * (box_width_pixels/2)
    center_y = edge_center_y + ny * (box_width_pixels/2)

    print(f"Box Position (pixels): ({center_x:.1f}, {center_y:.1f})")
    print(f"Box Orientation: {angle_deg:.2f} degrees")

    # ---- Visualization ----
    img_copy = img.copy()

    # detected edge
    cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # edge midpoint
    cv2.circle(img_copy, (int(edge_center_x), int(edge_center_y)), 5, (0,0,255), -1)

    # perpendicular direction
    p2 = (int(edge_center_x + nx*100), int(edge_center_y + ny*100))
    cv2.line(img_copy,
             (int(edge_center_x), int(edge_center_y)),
             p2,
             (255,0,0), 2)

    # estimated box center
    cv2.circle(img_copy, (int(center_x), int(center_y)), 6, (0,255,255), -1)

    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Box Detection - Position: ({center_x:.1f}, {center_y:.1f}) | Orientation: {angle_deg:.2f} deg")
    plt.axis('on')
    plt.show()