import cv2
import matplotlib.pyplot as plt
import numpy as np

blur_kernel = (5, 5)
canny_low, canny_high = 15, 100
hough_threshold_1, hough_min_len_1, hough_gap_1 = 25, 50, 30
hough_threshold_2, hough_min_len_2, hough_gap_2 = 20, 40, 30
perp_tolerance, corner_proximity, border_margin = 5.0, 200, 0.05

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

def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    return cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold_1, minLineLength=hough_min_len_1, maxLineGap=hough_gap_1)
    if lines is None or len(lines) == 0:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold_2, minLineLength=hough_min_len_2, maxLineGap=hough_gap_2)
    return lines

def find_best_line(lines, img_height, img_width):
    best_line, max_score = None, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2-x1, y2-y1)
        if ((y1 > img_height*(1-border_margin) and y2 > img_height*(1-border_margin)) or (y1 < img_height*border_margin and y2 < img_height*border_margin) or
            (x1 < img_width*border_margin and x2 < img_width*border_margin) or (x1 > img_width*(1-border_margin) and x2 > img_width*(1-border_margin))):
            continue
        angle = np.arctan2(y2-y1, x2-x1)
        angle_score = min(abs(np.degrees(angle)), 180 - abs(np.degrees(angle)))
        score = length if angle_score < 45 else length * 0.5
        if score > max_score:
            max_score, best_line = score, (x1, y1, x2, y2)
    return best_line

def find_perp_line(best_line, lines):
    bx1, by1, bx2, by2 = best_line
    best_angle = np.arctan2(by2-by1, bx2-bx1)
    best_endpoints = [(bx1, by1), (bx2, by2)]
    best_perp_line, best_perp_score = None, -1
    
    for line in lines:
        if np.array_equal(line[0], best_line):
            continue
        x1, y1, x2, y2 = line[0]
        current_angle = np.arctan2(y2-y1, x2-x1)
        angle_diff = min(abs(np.degrees(current_angle - best_angle)), 180 - abs(np.degrees(current_angle - best_angle)))
        
        if abs(angle_diff - 90) <= perp_tolerance:
            for best_ep in best_endpoints:
                for curr_ep in [(x1, y1), (x2, y2)]:
                    if points_close(best_ep, curr_ep, corner_proximity):
                        length = np.hypot(x2-x1, y2-y1)
                        if length > best_perp_score:
                            best_perp_score, best_perp_line = length, (x1, y1, x2, y2)
                        break
    return best_perp_line

def merge_parallel_connected_lines(perp_line, lines):
    if perp_line is None:
        return perp_line
    
    px1, py1, px2, py2 = perp_line
    perp_angle = np.arctan2(py2-py1, px2-px1)
    perp_endpoints = [(px1, py1), (px2, py2)]
    all_points = [(px1, py1), (px2, py2)]
    
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        line_angle = np.arctan2(ly2-ly1, lx2-lx1)
        angle_diff = min(abs(np.degrees(line_angle - perp_angle)), 180 - abs(np.degrees(line_angle - perp_angle)))
        
        if angle_diff <= 2:
            for ep in perp_endpoints:
                if points_close((lx1, ly1), ep, 50) or points_close((lx2, ly2), ep, 50):
                    all_points.append((lx1, ly1))
                    all_points.append((lx2, ly2))
                    break
    
    if len(all_points) > 2:
        all_points = np.array(all_points)
        distances = [np.hypot(p[0] - all_points[0, 0], p[1] - all_points[0, 1]) for p in all_points]
        idx_min, idx_max = np.argmin(distances), np.argmax(distances)
        return tuple(all_points[idx_min]) + tuple(all_points[idx_max])
    
    return perp_line

def detect(img_path_or_array, visualize=True, verbose=True):
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
    
    best_line = find_best_line(lines, img_height, img_width)
    if best_line is None:
        print("ERROR: No box line detected!")
        return None
    
    best_perp_line = find_perp_line(best_line, lines)
    if best_perp_line is None:
        print("ERROR: No perpendicular corner line found!")
        return None
    
    best_perp_line = merge_parallel_connected_lines(best_perp_line, lines)
    
    x1, y1, x2, y2 = best_line
    px1, py1, px2, py2 = best_perp_line
    angle_deg = np.degrees(np.arctan2(y2-y1, x2-x1))
    corner = line_intersection(best_line, best_perp_line)
    
    if corner is None:
        print("ERROR: Lines do not intersect!")
        return None
    
    corner_x, corner_y = corner
    edge1_mid_x, edge1_mid_y = (x1+x2)/2, (y1+y2)/2
    edge2_mid_x, edge2_mid_y = (px1+px2)/2, (py1+py2)/2
    dist1 = np.hypot(edge1_mid_x - corner_x, edge1_mid_y - corner_y)
    dist2 = np.hypot(edge2_mid_x - corner_x, edge2_mid_y - corner_y)
    
    dir1_x = (edge1_mid_x - corner_x) / dist1 if dist1 > 0 else 0
    dir1_y = (edge1_mid_y - corner_y) / dist1 if dist1 > 0 else 0
    dir2_x = (edge2_mid_x - corner_x) / dist2 if dist2 > 0 else 0
    dir2_y = (edge2_mid_y - corner_y) / dist2 if dist2 > 0 else 0
    
    center_x = corner_x + dir1_x * dist1 + dir2_x * dist2
    center_y = corner_y + dir1_y * dist1 + dir2_y * dist2
    
    edge1_length = np.hypot(x2-x1, y2-y1)
    edge2_length = np.hypot(px2-px1, py2-py1)
    
    result = {
        'corner': (corner_x, corner_y),
        'center': (center_x, center_y),
        'orientation': angle_deg,
        'edge1_length': edge1_length,
        'edge2_length': edge2_length,
        'best_line': best_line,
        'perp_line': best_perp_line,
        'img': img,
        'all_lines': lines
    }
    
    if verbose:
        print(f"Corner detected at: ({corner_x:.1f}, {corner_y:.1f})")
        print(f"Box Position (pixels): ({center_x:.1f}, {center_y:.1f})")
        print(f"Box Orientation: {angle_deg:.2f} degrees")
        print(f"Edge 1 length: {edge1_length:.1f}px, Edge 2 length: {edge2_length:.1f}px")
    
    if visualize:
        visualize_result(result)
    
    return result

def visualize_result(result):
    img_copy = result['img'].copy()
    x1, y1, x2, y2 = result['best_line']
    px1, py1, px2, py2 = result['perp_line']
    corner_x, corner_y = result['corner']
    center_x, center_y = result['center']
    angle_deg = result['orientation']
    all_lines = result['all_lines']
    
    np.random.seed(42)
    for line in all_lines:
        lx1, ly1, lx2, ly2 = line[0]
        color = tuple(int(x) for x in np.random.randint(0, 256, 3))
        cv2.line(img_copy, (lx1, ly1), (lx2, ly2), color, 1)
    
    cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.line(img_copy, (px1, py1), (px2, py2), (255, 0, 255), 3)
    cv2.circle(img_copy, (int(corner_x), int(corner_y)), 8, (255, 0, 0), -1)
    cv2.circle(img_copy, (int((x1+x2)/2), int((y1+y2)/2)), 5, (0, 0, 255), -1)
    cv2.circle(img_copy, (int((px1+px2)/2), int((py1+py2)/2)), 5, (0, 0, 255), -1)
    cv2.circle(img_copy, (int(center_x), int(center_y)), 6, (0, 255, 255), -1)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Box Detection - Corner: ({corner_x:.1f}, {corner_y:.1f}) | Center: ({center_x:.1f}, {center_y:.1f}) | Orientation: {angle_deg:.2f} deg")
    plt.axis('on')
    plt.show()

if __name__ == "__main__":
    result = detect(r'D:\Data\Me\CHULA\Study\Robotics\2147312-Robotics-Catch-a-box\test_img\box6_nocrop.jpg')