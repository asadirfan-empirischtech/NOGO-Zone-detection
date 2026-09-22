import cv2
import numpy as np
import argparse
import os
import sys
import math
import json
import requests
import base64
import re
import time
from openai import OpenAI

QWEN_API_URL = "http://localhost:8000/v1"
QWEN_MODEL = "Qwen/Qwen3.8-27B" 
SAM3_API_URL = "http://localhost:8001/v1/segment"

client = OpenAI(base_url=QWEN_API_URL, api_key="secret-123")

# Ensure local imports work regardless of where the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from master_pipeline import run_segmentation, call_sam3_for_image, decode_mask, call_qwen
from hybrid_plumb_geocalib_pipeline import run_geocalib_inference, intersect_ray_with_floor

def calc_3d_distance_geocalib(p1, p2, f_geo, down_vec, camera_height, h, w):
    """Calculates 3D physical distance using GeoCalib ray-casting logic."""
    cx = w / 2.0
    cy = h / 2.0
    
    floor_pts = []
    for pt in [p1, p2]:
        u, v = pt
        X = (u - cx) / f_geo
        Y = (v - cy) / f_geo
        Z = 1.0
        
        ray_cam = np.array([X, Y, Z], dtype=np.float32)
        ray_cam /= np.linalg.norm(ray_cam)
        
        hit_pt = intersect_ray_with_floor(ray_cam, down_vec, camera_height)
        if hit_pt is None:
            return None # Ray points above horizon
        floor_pts.append(hit_pt)
        
    return float(np.linalg.norm(floor_pts[0] - floor_pts[1]))

def calc_2d_distance_topdown(p1, p2, camera_height, fov, w):
    """Calculates physical distance using exact 2D top-down similar triangles."""
    fov_rad = math.radians(fov)
    focal_length_px = (w / 2.0) / math.tan(fov_rad / 2.0)
    meters_per_pixel = camera_height / focal_length_px
    
    pix_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return pix_dist * meters_per_pixel

def find_nearest_black_pixel(feet_pt, boundary_mask):
    """Finds the nearest 2D pixel coordinate in the boundary mask to the feet point."""
    pts = np.column_stack(np.where(boundary_mask > 0))
    if len(pts) == 0:
        return None
    
    pts_xy = pts[:, ::-1] # Convert (y, x) to (x, y)
    
    # Calculate squared distance to avoid expensive sqrt for all points
    diff = pts_xy - np.array(feet_pt)
    dist_sq = np.sum(diff**2, axis=1)
    
    nearest_idx = np.argmin(dist_sq)
    nearest_pt = pts_xy[nearest_idx]
    return tuple(nearest_pt)


def detect_people_ppe(image_path: str) -> list:
    """Use SAM 3 to detect all people in the image."""
    print("  [2/4] Detecting people using SAM 3...")
    detections = []
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        payload = {"labels": "human"} 
        time.sleep(1)
        response = requests.post(SAM3_API_URL, files=files, data=payload)
        
    if response.status_code == 200:
        detections = response.json().get("detections", [])
    else:
        print(f"  -> SAM 3 API Error: {response.status_code} - {response.text}")
        
    print(f"  -> Found {len(detections)} people.")
    return detections

def check_person_ppe(image_path: str, person_box: list, required_ppe: list) -> bool:
    """Use SAM 3 to check if the person in the bounding box is wearing at least 50% of the required PPE."""
    status = {}
    
    for item in required_ppe:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            payload = {
                "labels": item,
                "boxes": json.dumps([person_box])
            }
            time.sleep(1)
            response = requests.post(SAM3_API_URL, files=files, data=payload)
            
        if response.status_code == 200:
            dets = response.json().get("detections", [])
            status[item] = len(dets) > 0
        else:
            status[item] = False
            
    print(f"      SAM 3 PPE evaluation: {status}")
    
    items_checked = len(required_ppe)
    items_worn = sum(1 for item in required_ppe if status.get(item, False))
    
    if items_checked == 0: 
        return True
        
    percentage = items_worn / items_checked
    is_compliant = percentage >= 0.25
    print(f"      Compliance: {percentage*100:.0f}% -> {'PASS' if is_compliant else 'FAIL'}")
    return is_compliant

def main():
    parser = argparse.ArgumentParser(description="Unified Scalable System for Hazard and Distance Tracking")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--height", type=float, default=5.0, help="Camera height in meters")
    parser.add_argument("--is_topdown", action="store_true", help="Use 2D Top-Down math (FOV=90) instead of GeoCalib")
    parser.add_argument("--mode", choices=["segment", "distance", "ppe", "all"], default="distance", help="Pipeline mode (segmentation, distance, ppe, or all)")
    args = parser.parse_args()

    print(f"\n=============================================")
    print(f"      SCALABLE SYSTEM PIPELINE EXECUTING     ")
    print(f"=============================================")
    print(f"Mode: {args.mode.upper()}")
    
    canvas = None
    boundary_mask = None
    floor_mask = None
    required_ppe = []
    
    # ---------------------------------------------------------
    # STEP 1: SEGMENTATION
    # ---------------------------------------------------------
    if args.mode in ["segment", "distance", "all"]:
        print(f"Top-Down Mode: {'ENABLED (FOV=90)' if args.is_topdown else 'DISABLED (Using GeoCalib)'}")
        print("\n[STEP 1] Running hazard segmentation pipeline...")
        canvas, boundary_mask, floor_mask, required_ppe = run_segmentation(args.image)
        
        if canvas is None:
            print("❌ Failed to run segmentation.")
            return

        # If the user only wants segmentation, exit early.
        if args.mode == "segment":
            print("\n✅ Mode is 'segment'. Ending pipeline successfully.")
            return
    else:
        # For pure PPE mode without segment
        canvas = cv2.imread(args.image)
        print("\n[STEP 1] Asking Qwen for required PPE...")
        try:
            qwen_data = call_qwen(args.image)
            required_ppe = qwen_data.get("required_ppe", ["hard hat", "high-visibility vest"])
            print(f"  -> Required PPE identified: {required_ppe}")
            time.sleep(3)
        except Exception as e:
            print(f"❌ Qwen Pipeline Failed: {e}")
            required_ppe = ["hard hat", "high-visibility vest"]

    # ---------------------------------------------------------
    # STEP 2: PPE COMPLIANCE
    # ---------------------------------------------------------
    if args.mode in ["ppe", "all"]:
        print("\n[PPE COMPLIANCE CHECK]")
        if not required_ppe:
            print("❌ No PPE requirements could be determined. Skipping PPE.")
        else:
            people_detections = detect_people_ppe(args.image)
            if not people_detections:
                print("❌ No people detected. Skipping PPE.")
            else:
                print(f"  Evaluating PPE compliance for {len(people_detections)} people...")
                for i, det in enumerate(people_detections):
                    mask_bytes = base64.b64decode(det['mask_b64'])
                    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
                    mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
                    
                    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    person_box = [x, y, x+w, y+h]
                    
                    print(f"    Person {i+1}/{len(people_detections)} at {person_box}:")
                    is_compliant = check_person_ppe(args.image, person_box, required_ppe)
                    
                    color = (255, 0, 0) if is_compliant else (0, 0, 255) 
                    
                    bool_mask = mask_img > 127
                    canvas[bool_mask] = (
                        canvas[bool_mask].astype(np.float32) * 0.5 + 
                        np.array(color, dtype=np.float32) * 0.5
                    ).astype(np.uint8)
                    
                    cv2.drawContours(canvas, contours, -1, color, 2)
                    label = "PPE: PASS" if is_compliant else "PPE: FAIL"
                    cv2.putText(canvas, label, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        if args.mode == "ppe":
            out_name = f"ppe_output_{os.path.basename(args.image)}"
            out_path = os.path.join("PipelineVer2", "outputs", out_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, canvas)
            print(f"  [4/4] Saved PPE annotated image to: {out_path}")
            return

    # ---------------------------------------------------------
    # STEP 3: HUMAN DETECTION & DISTANCE
    # ---------------------------------------------------------
    if args.mode in ["distance", "all"]:
        print("\n[DISTANCE CALCULATION]")
        # We pass the original image to SAM3 to avoid it getting confused by the blue/red overlays.
        img_original = cv2.imread(args.image)
        detections = call_sam3_for_image(img_original, ["person", "worker"])
        
        human_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        
        if detections:
            for det in detections:
                if det.get("score", 0.0) >= 0.6:
                    mask_b64 = det.get("mask_b64")
                    if mask_b64:
                        full_mask = decode_mask(mask_b64)
                        human_mask[full_mask > 0] = 255
                        
        num_labels, labels = cv2.connectedComponents(human_mask)
        feet_points = []
        
        for i in range(1, num_labels): 
            pts = np.where(labels == i)
            if len(pts[0]) < 50: 
                continue
            
            max_y_idx = np.argmax(pts[0])
            feet_y = pts[0][max_y_idx]
            feet_x = pts[1][max_y_idx]
            feet_points.append((feet_x, feet_y))
            
        print(f" -> Found {len(feet_points)} valid human(s) in the image.")
        if len(feet_points) == 0:
            print(" -> No humans found. Exiting distance calculation phase.")
        else:
            print("\nInitializing geometric distance engine...")
            f_geo, down_vec = None, None
            if not args.is_topdown:
                print(" -> Using Hybrid GeoCalib Ray-Casting (Extracting Vanishing Points).")
                f_geo, down_vec = run_geocalib_inference(args.image, canvas.shape)
            else:
                print(" -> Using exact 2D Top-Down Mode (Assumed FOV = 90).")

            print("\nCalculating distances to nearest hazard boundary...")
            h, w = canvas.shape[:2]
            
            for feet_pt in feet_points:
                nearest_bnd_pt = find_nearest_black_pixel(feet_pt, boundary_mask)
                nearest_floor_pt = find_nearest_black_pixel(feet_pt, floor_mask)
                
                if nearest_bnd_pt is None and nearest_floor_pt is None:
                    print(f" -> No boundary or floor found for human at {feet_pt}. Skipping.")
                    continue
                    
                dist_to_bnd = None
                if nearest_bnd_pt is not None:
                    if args.is_topdown:
                        dist_to_bnd = calc_2d_distance_topdown(feet_pt, nearest_bnd_pt, args.height, 90.0, w)
                    else:
                        dist_to_bnd = calc_3d_distance_geocalib(feet_pt, nearest_bnd_pt, f_geo, down_vec, args.height, h, w)
                
                dist_to_floor = None
                if nearest_floor_pt is not None:
                    if args.is_topdown:
                        dist_to_floor = calc_2d_distance_topdown(feet_pt, nearest_floor_pt, args.height, 90.0, w)
                    else:
                        dist_to_floor = calc_3d_distance_geocalib(feet_pt, nearest_floor_pt, f_geo, down_vec, args.height, h, w)
                        
                # Determine status
                status_text = "SAFE"
                color = (0, 255, 0) # Green
                
                if dist_to_floor is not None and dist_to_floor < 0.10:
                    status_text = "DANGER: INSIDE HAZARD"
                    color = (0, 0, 255) # Red
                elif dist_to_bnd is not None and dist_to_bnd < 2.0:
                    status_text = "CAUTION"
                    color = (0, 165, 255) # Orange (BGR)
                    
                # Draw connections to black boundary if it exists
                if nearest_bnd_pt is not None:
                    cv2.circle(canvas, feet_pt, 6, color, -1) 
                    cv2.circle(canvas, nearest_bnd_pt, 6, (255, 0, 0), -1) 
                    cv2.line(canvas, feet_pt, nearest_bnd_pt, color, 2) 
                    
                    mid_x = (feet_pt[0] + nearest_bnd_pt[0]) // 2
                    mid_y = (feet_pt[1] + nearest_bnd_pt[1]) // 2
                    
                    if dist_to_bnd is not None:
                        text = f"{dist_to_bnd:.2f}m ({status_text})"
                        print(f" -> Distance to Hazard: {dist_to_bnd:.2f}m. Status: {status_text}")
                        cv2.putText(canvas, text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                        cv2.putText(canvas, text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    else:
                        print(f" -> Distance to Hazard: Above Horizon (invalid ray).")
                        cv2.putText(canvas, "Above Horizon", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # If we only have floor distance (edge case)
                    if dist_to_floor is not None:
                        print(f" -> Distance to floor: {dist_to_floor:.2f}m. Status: {status_text}")
                        cv2.putText(canvas, status_text, (feet_pt[0], feet_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
        out_name = f"scalable_system_output_{os.path.basename(args.image)}"
        out_path = os.path.join("PipelineVer2", "outputs", out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, canvas)
        print(f"\n✅ Scalable System complete! Saved final annotated image to {out_path}")
    
if __name__ == "__main__":
    main()
