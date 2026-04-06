import cv2
import numpy as np

# ── Test image path ───────────────────────────────────────────────────────────
test_image_path = r"D:\Data\Me\CHULA\Study\Robotics\2147312-Robotics-Catch-a-box\test_img_crop\box4_crop.jpg"

# ── Load image ────────────────────────────────────────────────────────────────
img = cv2.imread(test_image_path)
if img is None:
    print("ERROR: Could not load image!")
    exit()

img_height, img_width = img.shape[:2]
print(f"Image loaded: {img_width}x{img_height}")

# ── Create window and trackbars ───────────────────────────────────────────────
window_name = "Pink Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 600)

# Default HSV values for pink
h_lower_default = 130
s_lower_default = 100
v_lower_default = 100
h_upper_default = 180
s_upper_default = 255
v_upper_default = 255

def nothing(x):
    pass

# Create trackbars for HSV adjustment
cv2.createTrackbar("H_Lower", window_name, h_lower_default, 179, nothing)
cv2.createTrackbar("S_Lower", window_name, s_lower_default, 255, nothing)
cv2.createTrackbar("V_Lower", window_name, v_lower_default, 255, nothing)

cv2.createTrackbar("H_Upper", window_name, h_upper_default, 179, nothing)
cv2.createTrackbar("S_Upper", window_name, s_upper_default, 255, nothing)
cv2.createTrackbar("V_Upper", window_name, v_upper_default, 255, nothing)

print("=" * 60)
print("Pink Detection with HSV Slider Adjustment")
print("=" * 60)
print("Adjust the sliders to fine-tune pink detection")
print("Press 'q' to quit")
print("=" * 60)

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    # Get trackbar values
    h_lower = cv2.getTrackbarPos("H_Lower", window_name)
    s_lower = cv2.getTrackbarPos("S_Lower", window_name)
    v_lower = cv2.getTrackbarPos("V_Lower", window_name)
    
    h_upper = cv2.getTrackbarPos("H_Upper", window_name)
    s_upper = cv2.getTrackbarPos("S_Upper", window_name)
    v_upper = cv2.getTrackbarPos("V_Upper", window_name)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create result image - only show pink regions
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Add info text
    info_text = f"H: [{h_lower}-{h_upper}]  S: [{s_lower}-{s_upper}]  V: [{v_lower}-{v_upper}]"
    cv2.putText(result, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Count pink pixels
    pink_pixels = np.count_nonzero(mask)
    total_pixels = img_height * img_width
    percentage = (pink_pixels / total_pixels) * 100
    count_text = f"Pink: {pink_pixels} pixels ({percentage:.2f}%)"
    cv2.putText(result, count_text, (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display result
    cv2.imshow(window_name, result)
    
    # Wait for key press
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        print("\nClosing...")
        print(f"Final HSV range:")
        print(f"  Lower: H={h_lower}, S={s_lower}, V={v_lower}")
        print(f"  Upper: H={h_upper}, S={s_upper}, V={v_upper}")
        break

cv2.destroyAllWindows()