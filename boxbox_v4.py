import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────

blur_kernel       = (5, 5)
canny_low         = 10
canny_high        = 100

hough_threshold_1 = 25
hough_min_len_1   = 60
hough_gap_1       = 20

hough_threshold_2 = 20
hough_min_len_2   = 40
hough_gap_2       = 30

perp_tolerance    = 0.5
corner_proximity  = 20
border_margin     = 0.05

# ── Geometry helpers ──────────────────────────────────────────────────────────

def points_close(p1, p2, threshold=20):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1]) < threshold

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    return (x1 + t*(x2-x1), y1 + t*(y2-y1))

def _is_border_line(x1, y1, x2, y2, img_height, img_width):
    return (
        (y1 > img_height*(1-border_margin) and y2 > img_height*(1-border_margin)) or
        (y1 < img_height*border_margin      and y2 < img_height*border_margin)      or
        (x1 < img_width*border_margin       and x2 < img_width*border_margin)       or
        (x1 > img_width*(1-border_margin)   and x2 > img_width*(1-border_margin))
    )

# ── Detection pipeline ────────────────────────────────────────────────────────

def detect_edges(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    return cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

def detect_lines(edges):
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=hough_threshold_1,
        minLineLength=hough_min_len_1,
        maxLineGap=hough_gap_1
    )
    if lines is None or len(lines) == 0:
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=hough_threshold_2,
            minLineLength=hough_min_len_2,
            maxLineGap=hough_gap_2
        )
    return lines

def merge_parallel_connected_lines(line, lines, angle_tol=2, proximity=50):
    lx1, ly1, lx2, ly2 = line
    line_angle = np.arctan2(ly2-ly1, lx2-lx1)
    endpoints  = [(lx1, ly1), (lx2, ly2)]
    all_points = [(lx1, ly1), (lx2, ly2)]

    for candidate in lines:
        cx1, cy1, cx2, cy2 = candidate[0]

        # Skip if it IS the seed line
        if cx1 == lx1 and cy1 == ly1 and cx2 == lx2 and cy2 == ly2:
            continue

        cand_angle = np.arctan2(cy2-cy1, cx2-cx1)
        angle_diff = abs(np.degrees(cand_angle - line_angle)) % 180
        angle_diff = min(angle_diff, 180 - angle_diff)

        if angle_diff > angle_tol:
            continue

        # Accept if any endpoint of the candidate is near any endpoint of the line
        for ep in endpoints:
            if points_close((cx1, cy1), ep, proximity) or points_close((cx2, cy2), ep, proximity):
                all_points.append((cx1, cy1))
                all_points.append((cx2, cy2))
                break

    if len(all_points) > 2:
        all_points = np.array(all_points)
        # Project all points onto the line direction to find true extremes
        dx = np.cos(line_angle)
        dy = np.sin(line_angle)
        projections = [p[0]*dx + p[1]*dy for p in all_points]
        p_min = all_points[np.argmin(projections)]
        p_max = all_points[np.argmax(projections)]
        return (int(p_min[0]), int(p_min[1]), int(p_max[0]), int(p_max[1]))

    return line

def find_best_pair(lines, img_height, img_width):
    best_pair, best_score = None, -1

    for i, line_a in enumerate(lines):
        ax1, ay1, ax2, ay2 = line_a[0]
        if _is_border_line(ax1, ay1, ax2, ay2, img_height, img_width):
            continue
        len_a   = np.hypot(ax2-ax1, ay2-ay1)
        angle_a = np.arctan2(ay2-ay1, ax2-ax1)

        for line_b in lines[i+1:]:
            bx1, by1, bx2, by2 = line_b[0]
            if _is_border_line(bx1, by1, bx2, by2, img_height, img_width):
                continue
            len_b   = np.hypot(bx2-bx1, by2-by1)
            angle_b = np.arctan2(by2-by1, bx2-bx1)

            # Perpendicularity check
            angle_diff = abs(np.degrees(angle_a - angle_b)) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)
            if abs(angle_diff - 90) > perp_tolerance:
                continue

            # Closest endpoint-pair distance
            ep_pairs = [
                ((ax1, ay1), (bx1, by1)),
                ((ax1, ay1), (bx2, by2)),
                ((ax2, ay2), (bx1, by1)),
                ((ax2, ay2), (bx2, by2)),
            ]
            min_dist = min(np.hypot(p[0]-q[0], p[1]-q[1]) for p, q in ep_pairs)

            if min_dist > corner_proximity * 6:
                continue

            # Score: reward total length, penalise endpoint gap
            score = (len_a + len_b) / (1 + min_dist)
            if score > best_score:
                best_score = score
                best_pair  = (tuple(line_a[0]), tuple(line_b[0]))

    return best_pair

# ── Main detect function ──────────────────────────────────────────────────────

def detect(img_path_or_array, visualize=False, verbose=True):
    img = cv2.imread(img_path_or_array) if isinstance(img_path_or_array, str) else img_path_or_array
    if img is None:
        print("ERROR: Could not load image!")
        return None

    img_height, img_width = img.shape[:2]
    edges = detect_edges(img)
    lines = detect_lines(edges)

    if verbose:
        print(f"Lines detected: {len(lines) if lines is not None else 0}")
    if lines is None or len(lines) == 0:
        print("ERROR: No lines detected!")
        return None

    pair = find_best_pair(lines, img_height, img_width)
    if pair is None:
        print("ERROR: No perpendicular corner pair found!")
        return None

    best_line, best_perp_line = pair

    # Merge both lines before using them
    best_line      = merge_parallel_connected_lines(best_line,      lines)
    best_perp_line = merge_parallel_connected_lines(best_perp_line, lines)

    x1,  y1,  x2,  y2  = best_line
    px1, py1, px2, py2 = best_perp_line
    corner = line_intersection(best_line, best_perp_line)

    if corner is None:
        print("ERROR: Lines do not intersect!")
        return None

    corner_x, corner_y = corner
    edge1_mid_x, edge1_mid_y = (x1+x2)/2,   (y1+y2)/2
    edge2_mid_x, edge2_mid_y = (px1+px2)/2, (py1+py2)/2
    dist1 = np.hypot(edge1_mid_x-corner_x, edge1_mid_y-corner_y)
    dist2 = np.hypot(edge2_mid_x-corner_x, edge2_mid_y-corner_y)

    dir1_x = (edge1_mid_x-corner_x) / dist1 if dist1 > 0 else 0
    dir1_y = (edge1_mid_y-corner_y) / dist1 if dist1 > 0 else 0
    dir2_x = (edge2_mid_x-corner_x) / dist2 if dist2 > 0 else 0
    dir2_y = (edge2_mid_y-corner_y) / dist2 if dist2 > 0 else 0

    center_x = corner_x + dir1_x*dist1 + dir2_x*dist2
    center_y = corner_y + dir1_y*dist1 + dir2_y*dist2

    edge1_length = np.hypot(x2-x1,   y2-y1)
    edge2_length = np.hypot(px2-px1, py2-py1)
    edge_ratio   = max(edge1_length, edge2_length) / min(edge1_length, edge2_length)
    box_state    = 'stand' if edge_ratio > 1.35 else 'lie'
    angle_deg    = -np.degrees(np.arctan2(y2-y1, x2-x1)) if edge1_length < edge2_length else -np.degrees(np.arctan2(py2-py1, px2-px1))

    result = {
        'corner':       (corner_x, corner_y),
        'center':       (center_x, center_y),
        'orientation':  angle_deg,
        'edge1_length': edge1_length,
        'edge2_length': edge2_length,
        'best_line':    best_line,
        'perp_line':    best_perp_line,
        'img':          img,
        'all_lines':    lines,
        'box_state':    box_state,
    }

    if verbose:
        print(f"Corner detected at:    ({corner_x:.1f}, {corner_y:.1f})")
        print(f"Box center (pixels):   ({center_x:.1f}, {center_y:.1f})")
        print(f"Box orientation:       {angle_deg:.2f} deg")
        print(f"Edge lengths:          {edge1_length:.1f}px, {edge2_length:.1f}px")
        print(f"Box state:             {box_state}")

    if visualize:
        visualize_result(result)

    return result

# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_result(result):
    img      = result['img']
    img_copy = img.copy()
    x1,  y1,  x2,  y2  = result['best_line']
    px1, py1, px2, py2 = result['perp_line']
    corner_x, corner_y = result['corner']
    center_x, center_y = result['center']
    angle_deg           = result['orientation']

    np.random.seed(42)
    for line in result['all_lines']:
        lx1, ly1, lx2, ly2 = line[0]
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        cv2.line(img_copy, (lx1, ly1), (lx2, ly2), color, 1)

    cv2.line(img_copy,   (x1,  y1),  (x2,  y2),  (0, 255, 0),   3)
    cv2.line(img_copy,   (px1, py1), (px2, py2), (255, 0, 255),  3)
    cv2.circle(img_copy, (int(corner_x), int(corner_y)),           8, (255, 0,   0),   -1)
    cv2.circle(img_copy, (int((x1+x2)/2),   int((y1+y2)/2)),       5, (0,   0,   255), -1)
    cv2.circle(img_copy, (int((px1+px2)/2), int((py1+py2)/2)),     5, (0,   0,   255), -1)
    cv2.circle(img_copy, (int(center_x), int(center_y)),           6, (0,   255, 255), -1)

    edges = detect_edges(img)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis('on')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny edges")
    plt.axis('on')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Corner ({corner_x:.1f}, {corner_y:.1f})  |  "
        f"Center ({center_x:.1f}, {center_y:.1f})  |  "
        f"{angle_deg:.1f} deg"
    )
    plt.axis('on')

    plt.tight_layout()
    plt.show()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = detect(r'D:\Data\Me\CHULA\Study\Robotics\2147312-Robotics-Catch-a-box\test_img_crop\box4_crop.jpg')