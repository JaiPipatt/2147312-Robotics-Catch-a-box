import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 1. SETUP: Load images
image_folder = r'C:\Users\Asus\Downloads\Percogrobot_group\2147312-Robotics-Catch-a-box\Image'
reference_descriptors = []
orb = cv2.ORB_create(nfeatures=1000) # Increased features for better accuracy

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".webp", ".png")):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, 0) # Load as grayscale directly
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            reference_descriptors.append((kp, des, filename))

# 2. DETECTION: Function to find the best match in a live frame
def find_box(live_frame):
    live_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
    kp_live, des_live = orb.detectAndCompute(live_gray, None)
    
    if des_live is None: return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_count = 0
    best_info = None

    for kp_ref, des_ref, name in reference_descriptors:
        matches = bf.match(des_live, des_ref)
        # Filter for "good" matches based on distance
        good_matches = [m for m in matches if m.distance < 50] 
        
        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_info = (kp_ref, kp_live, good_matches, name)

    return best_info

# 3. EXECUTION: Test it on a new image or camera feed
test = cv2.imread(r'C:\Users\Asus\Downloads\Percogrobot_group\2147312-Robotics-Catch-a-box\Image\LINE_ALBUM_box box_260324_1.jpg')
result = find_box(test)

# 4. VISUALIZATION: Display the results
if result is not None:
    kp_ref, kp_live, good_matches, name = result
    
    # Load the reference image in color for visualization
    ref_img_path = os.path.join(image_folder, name)
    ref_img = cv2.imread(ref_img_path)
    
    # Draw matches
    final_img = cv2.drawMatches(ref_img, kp_ref, test, kp_live, good_matches[:20], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display with matplotlib
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Best Match: {name} ({len(good_matches)} matches)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No match found!")