import cv2
import numpy as np
import scipy.optimize
import torch
import math
import argparse
import sys
import os
from geocalib import GeoCalib

##############################################################################
# MODULE 1: GeoCalib (AI Extrinsics & Focal Length)
##############################################################################

def run_geocalib_inference(img_path, img_shape):
    print("\n--- MODULE 1: GEOCALIB AI EXTRINSICS ---")
    print("Loading GeoCalib (pinhole weights)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GeoCalib(weights='pinhole').to(device)
    
    print("Analyzing image for vanishing points...")
    img_tensor = model.load_image(img_path).to(device)
    res = model.calibrate(img_tensor)
    
    camera = res['camera']
    gravity = res['gravity']
    
    # Extract AI's refined Focal Length from the flat image
    # Note: camera.f gives a single scalar focal length normalized by image width, or pixel coordinates depending on internal GeoCalib format.
    # In GeoCalib, camera is usually represented in normalized coordinates, so we can extract FOV first.
    h_fov_rad = camera.hfov.item()
    v_fov_rad = camera.vfov.item()
    h, w = img_shape[:2]
    
    # Convert FOV to Focal Length in pixels
    f_geo_x = (w / 2.0) / math.tan(h_fov_rad / 2.0)
    f_geo_y = (h / 2.0) / math.tan(v_fov_rad / 2.0)
    f_geo = (f_geo_x + f_geo_y) / 2.0 # Average them
    
    # GeoCalib's gravity vector points UP towards the ceiling.
    # We must invert it so it points DOWN towards the floor.
    up_vec = gravity.vec3d.squeeze().cpu().numpy()
    down_vec = -up_vec
    
    pitch = math.degrees(math.asin(down_vec[2])) # Z component
    print(f"GeoCalib Extracted Focal Length: {f_geo:.2f}px")
    print(f"GeoCalib Extracted Gravity Vector: {down_vec}")
    print(f"Estimated Pitch: {pitch:.2f} deg")
    
    return f_geo, down_vec

##############################################################################
# MODULE 2: The 3D Ray-Casting Engine
##############################################################################

clicked_points = []
img_display = None
img_original = None
f_geo_global = None
down_vec_global = None
pitch_global = 0.0
CAMERA_HEIGHT = 5.0

def intersect_ray_with_floor(ray_cam, down_vec, height):
    """Finds where the 3D ray hits the ground plane."""
    # To rotate the ray to world coordinates, we build a rotation matrix where down_vec is the new Z-axis (or Y-axis).
    # However, a simpler mathematical approach is to use the dot product.
    # The dot product of the ray and the gravity vector tells us the "downward" component of the ray.
    v_down = np.dot(ray_cam, down_vec)
    
    if v_down < 1e-6:
        return None # Ray points above horizon
        
    scale = height / v_down
    hit_point_3d = ray_cam * scale
    return hit_point_3d

def compute_distance(p1, p2):
    """Executes the 3D Ray-Casting Algorithm on two raw pixels."""
    pts = [p1, p2]
    floor_pts = []
    
    h, w = img_original.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    
    for j in range(2):
        u = pts[j][0]
        v = pts[j][1]
        
        # Normalize to Rays using GeoCalib's Focal Length
        X = (u - cx) / f_geo_global
        Y = (v - cy) / f_geo_global
        Z = 1.0
        
        ray_cam = np.array([X, Y, Z], dtype=np.float32)
        
        # Normalize the ray vector
        ray_cam /= np.linalg.norm(ray_cam)
        
        # Apply GeoCalib's Pitch and Intersect the Floor Plane
        hit_pt = intersect_ray_with_floor(ray_cam, down_vec_global, CAMERA_HEIGHT)
        if hit_pt is None:
            return None
            
        floor_pts.append(hit_pt)
        
    # Calculate Euclidean Distance
    return float(np.linalg.norm(floor_pts[0] - floor_pts[1]))

def redraw():
    global img_display
    img_display = img_original.copy()
    
    for pt in clicked_points:
        cv2.circle(img_display, pt, 5, (0, 0, 255), -1)
        
    for i in range(0, len(clicked_points) - 1, 2):
        p1 = clicked_points[i]
        p2 = clicked_points[i+1]
        cv2.line(img_display, p1, p2, (0, 255, 0), 2)
        
        dist = compute_distance(p1, p2)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        if dist is not None:
            text = f"{dist:.2f}m"
            color = (0, 255, 255)
        else:
            text = "Above Horizon"
            color = (0, 0, 255)
            
        cv2.putText(img_display, text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img_display, text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    cv2.imshow("Module 2: Standard Ray-Casting Engine", img_display)

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points:
            clicked_points.pop()
            redraw()

def main():
    global img_original, img_display
    global f_geo_global, down_vec_global, pitch_global, CAMERA_HEIGHT
    
    parser = argparse.ArgumentParser(description="Standard CCTV GeoCalib Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--height", type=float, default=5.0, help="Camera height in meters")
    args = parser.parse_args()
    
    global CAMERA_HEIGHT
    CAMERA_HEIGHT = args.height

    img_original = cv2.imread(args.image)
    if img_original is None:
        print(f"Error: Could not read image at {args.image}")
        return

    print("=========================================================")
    print("         STANDARD GEOCALIB RAY-CASTING PIPELINE          ")
    print("=========================================================")

    # Run Module 1 (GeoCalib)
    f_geo_global, down_vec_global = run_geocalib_inference(args.image, img_original.shape)
    
    # Store initial pitch
    pitch_global = math.degrees(math.asin(down_vec_global[2]))
    
    # Run Module 2 (GUI)
    print("\n--- MODULE 2: 3D RAY-CASTING ENGINE ---")
    print("Opening CCTV frame...")
    print(f"Camera height locked at {CAMERA_HEIGHT} meters.")
    print("1. Click points to form pairs and measure distance.")
    print("2. Right-click to undo.")
    print("3. Press 'W' to tilt camera DOWN (increases pitch, shrinks distances).")
    print("4. Press 'S' to tilt camera UP (decreases pitch, stretches distances).")
    print("5. Press 'q' to exit.")
    
    cv2.namedWindow("Module 2: Standard Ray-Casting Engine", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Module 2: Standard Ray-Casting Engine", mouse_callback)
    
    redraw()
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            pitch_global += 1.0
            print(f"Pitch increased to {pitch_global:.2f} deg")
            pitch_rad = math.radians(pitch_global)
            down_vec_global = np.array([0.0, math.cos(pitch_rad), math.sin(pitch_rad)], dtype=np.float32)
            redraw()
        elif key == ord('s'):
            pitch_global -= 1.0
            print(f"Pitch decreased to {pitch_global:.2f} deg")
            pitch_rad = math.radians(pitch_global)
            down_vec_global = np.array([0.0, math.cos(pitch_rad), math.sin(pitch_rad)], dtype=np.float32)
            redraw()
            
    cv2.destroyAllWindows()
    print("Exiting pipeline. Thank you!")

if __name__ == "__main__":
    main()
