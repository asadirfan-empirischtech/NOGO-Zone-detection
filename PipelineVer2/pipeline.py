import cv2
import requests
import json
import base64
import numpy as np
import os
import re
import argparse
from config import QWEN_API_URL, QWEN_API_KEY, SAM3_API_URL, QWEN_MODEL, QWEN_MAX_TOKENS, QWEN_TEMPERATURE, COLOR_FLOOR, COLOR_HURDLE, FLOOR_KEYWORDS, COLOR_WARNING

QWEN_SYSTEM_PROMPT = """You are an expert safety and scene analyzer. Use thinking in this process to deeply understand the scene context. Your task is to find hazard areas, restricted zones, or enclosed spaces.

Analyze the image very carefully. Multiple hazard zones can exist in a single image. Look for ALL of them. 

Look specifically for:
- Stripped boundaries on the floor.
- Cones covering an area.
- Barricades.
- Hurdles around a machine.
- Police-style strips around crime scenes or broken roads.
- Note: The boundary object might just be flooring colors or painted lines! Be dynamic and reason like a human.

The area must be forming a closed space or covering something. It does not matter if the area inside is completely empty; if it is barricaded or marked off, it is a hazard zone.

Return your analysis STRICTLY as a JSON object with a list of zones. For every zone you find, you must provide:
- zone_id: A unique identifier.
- visibility_rating: An integer from 0 to 100 rating how visible and clear this hazard zone is.
- large_bounding_box: A single bounding box [ymin, xmin, ymax, xmax] (normalized 0-1000 coordinates). Keep the bounding box strictly CONFINED and tight around the hazard zone (enclosing the outer edges of the hazard tape or cones).
- sam3_objects: A list of object dictionaries. You MUST generate EXACTLY these 2 object_types for EVERY zone:
  1. "boundary": The hazard tape, cones, chairs, or fences marking the zone. Provide 4-6 phrases.
  2. "floor": The actual empty flat ground surface inside the zone. Provide 4-6 phrases. CRITICAL: Use highly specific phrases like "flat empty concrete floor", "bare ground", or "empty floor tiles" to ensure machines/pillars are NOT included in the floor mask!

OUTPUT FORMAT:
- First, do your spatial reasoning BRIEFLY inside <think>...</think> tags. Iterate your reasoning at least twice to ensure accuracy before finalizing the JSON.
- Then, IMMEDIATELY after the closing </think> tag, output ONLY the raw JSON object.
- Do NOT include any text, explanation, or markdown outside the JSON structure after </think>.
"""

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_qwen(image_path: str) -> dict:
    b64_image = encode_image_base64(image_path)
    url = f"{QWEN_API_URL}/chat/completions"
    
    payload = {
        "model": QWEN_MODEL,
        "max_tokens": QWEN_MAX_TOKENS,
        "temperature": QWEN_TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [
            {
                "role": "system",
                "content": QWEN_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image according to the system prompt and return the JSON."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}"
    }
    
    print(f"Calling Qwen-VL at {url}...")
    response = requests.post(url, json=payload, headers=headers, timeout=300)
    response.raise_for_status()
    
    content = response.json()["choices"][0]["message"]["content"]
    
    # Try to extract JSON from <think> block or code blocks
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
        
    # Find the first { and last } to extract just the JSON object
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        json_str = content[start_idx:end_idx+1]
    else:
        json_str = content
        
    json_str = re.sub(r"```json\s*", "", json_str)
    json_str = re.sub(r"```\s*", "", json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON. Raw content:\n{content}")
        raise e

def decode_mask(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    mask_bytes = base64.b64decode(b64_str)
    mask_array = np.frombuffer(mask_bytes, np.uint8)
    return cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

def call_sam3_for_image(image: np.ndarray, phrases: list) -> list:
    _, buffer = cv2.imencode('.png', image)
    files = {"file": ("full_image.png", buffer.tobytes(), "image/png")}
    
    phrase_str = ", ".join(phrases)
    data = {"labels": phrase_str}
    
    print(f" -> Sending full image to SAM3 with phrases: {phrase_str}")
    response = requests.post(SAM3_API_URL, files=files, data=data, timeout=120)
    response.raise_for_status()
    
    return response.json().get("detections", [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found.")
        return

    try:
        qwen_data = call_qwen(args.image)
        
        json_out_path = os.path.join("PipelineVer2", "outputs", f"qwen_output_{os.path.basename(args.image)}.json")
        with open(json_out_path, "w") as f:
            json.dump(qwen_data, f, indent=4)
        print(f"✅ Saved Qwen JSON to {json_out_path}")
            
        zones = qwen_data.get("zones", [])
        if not zones:
            print("Qwen found no zones.")
            return
            
        valid_zones = []
        for z in zones:
            if not isinstance(z, dict):
                print(f"Skipping malformed zone entry (not a dict): {z}")
                continue
                
            rating = z.get("visibility_rating", 100)
            if rating >= 60:
                valid_zones.append(z)
            else:
                print(f"Skipping Zone {z.get('zone_id')} due to low rating ({rating} < 60)")
                
        if not valid_zones:
            print("No zones passed the 60% visibility threshold.")
            return
            
        print(f"✅ Qwen identified {len(valid_zones)} valid hazard zones.")
    except Exception as e:
        print(f"❌ Qwen Pipeline Failed: {e}")
        return

    img = cv2.imread(args.image)
    H, W = img.shape[:2]
    
    final_overlay = np.zeros_like(img)
    blended_mask = np.zeros(img.shape[:2], dtype=bool)
    boundary_combined = np.zeros(img.shape[:2], dtype=bool)
    raw_floor_combined = np.zeros(img.shape[:2], dtype=bool)
    
    # We will pass the FULL image to SAM3 for each zone to avoid clipping boundary lines
    # Then we mask the results based on Qwen's bounding box.
    for idx, zone in enumerate(valid_zones):
        zone_id = zone.get("zone_id", f"Zone_{idx+1}")
        bbox = zone.get("large_bounding_box", [])
        sam3_objects_raw = zone.get("sam3_objects", [])
        
        # Robustly parse sam3_objects in case Qwen returned a dictionary instead of a list
        sam3_objects = []
        if isinstance(sam3_objects_raw, dict):
            for k, v in sam3_objects_raw.items():
                if isinstance(v, list):
                    sam3_objects.append({"object_type": k, "phrases": v})
                elif isinstance(v, dict) and "phrases" in v:
                    sam3_objects.append({"object_type": k, "phrases": v["phrases"]})
        elif isinstance(sam3_objects_raw, list):
            sam3_objects = sam3_objects_raw
        
        if len(bbox) != 4:
            print(f"Skipping {zone_id} due to invalid bbox format.")
            continue
            
        ymin, xmin, ymax, xmax = bbox
        
        y1 = max(0, int((ymin / 1000.0) * H))
        x1 = max(0, int((xmin / 1000.0) * W))
        y2 = min(H, int((ymax / 1000.0) * H))
        x2 = min(W, int((xmax / 1000.0) * W))
        
        print(f"\nProcessing {zone_id}: Confined BBox [{y1}:{y2}, {x1}:{x2}]")
        
        # We process BOUNDARY first, then use its shape to clip the FLOOR
        boundary_masks_to_apply = []
        floor_masks_to_apply = []
        
        # Separate boundary and floor objects
        boundary_objs = [o for o in sam3_objects if o.get("object_type", "").lower() == "boundary"]
        floor_objs = [o for o in sam3_objects if o.get("object_type", "").lower() == "floor"]
        
        pad = 20
        c_y1 = max(0, y1 - pad)
        c_x1 = max(0, x1 - pad)
        c_y2 = min(H, y2 + pad)
        c_x2 = min(W, x2 + pad)
        
        # ── Step 1: Detect boundary objects ──
        for obj in boundary_objs:
            phrases = obj.get("phrases", [])
            if not phrases:
                continue
            try:
                detections = call_sam3_for_image(img, phrases)
            except Exception as e:
                print(f"❌ SAM3 Failed for {zone_id} boundary: {e}")
                continue
            if not detections:
                print(f" -> No detections found for boundary phrases.")
                continue
                
            best_label = ""
            highest_valid_score = -1.0
            found_valid_mask = False
            for det in detections:
                score = det.get(    "score", 0.0)
                if score < 0.75:
                    continue
                
                mask_b64 = det.get("mask_b64")
                if not mask_b64: continue
                full_mask = decode_mask(mask_b64)
                overlap = full_mask[c_y1:c_y2, c_x1:c_x2]
                if not np.any(overlap > 0):
                    continue
                found_valid_mask = True
                label = det.get("label", "")
                if score > highest_valid_score:
                    highest_valid_score = score
                    best_label = label
                # Clip boundary mask to bounding box with generous padding
                # so tape at edges isn't cut. We use a massive pad (300) so Qwen's 
                # occasionally short bounding boxes don't chop off the North area!
                bpad = 300
                b_y1 = max(0, y1 - bpad)
                b_x1 = max(0, x1 - bpad)
                b_y2 = min(H, y2 + bpad)
                b_x2 = min(W, x2 + bpad)
                clipped_boundary = np.zeros_like(full_mask)
                clipped_boundary[b_y1:b_y2, b_x1:b_x2] = full_mask[b_y1:b_y2, b_x1:b_x2]
                boundary_masks_to_apply.append(clipped_boundary)
                
            if found_valid_mask:
                print(f" -> Best mask applied for boundary was '{best_label}' with score {highest_valid_score:.2f} (and possibly other overlapping pieces)")
            else:
                print(f" -> No overlapping masks found for boundary inside the bounding box.")
        
        # ── Step 2: Build convex hull from boundary masks for floor clipping ──
        interior_clip = None
        if boundary_masks_to_apply:
            combined_boundary = np.zeros(img.shape[:2], dtype=np.uint8)
            for m in boundary_masks_to_apply:
                combined_boundary[m > 0] = 255
            
            points = np.column_stack(np.where(combined_boundary > 0))
            if len(points) > 2:
                hull_points = points[:, ::-1]  # flip to (x, y)
                hull = cv2.convexHull(hull_points.astype(np.float32))
                
                # Shrink hull 5% inward so floor doesn't bleed past cones/tape
                M = cv2.moments(hull)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = int(np.mean(hull[:, 0, 0]))
                    cy = int(np.mean(hull[:, 0, 1]))
                
                # Set shrink factor to 0.0 so the convex hull perfectly traces 
                # the boundary without creating gaps in the black region.
                shrink_factor = 0.0
                shrunk_hull = hull.copy().astype(np.float32)
                for i in range(len(shrunk_hull)):
                    shrunk_hull[i, 0, 0] = hull[i, 0, 0] + (cx - hull[i, 0, 0]) * shrink_factor
                    shrunk_hull[i, 0, 1] = hull[i, 0, 1] + (cy - hull[i, 0, 1]) * shrink_factor
                
                interior_clip = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(interior_clip, shrunk_hull.astype(np.int32), 255)
                print(f" -> Built shrunk convex hull ({len(hull)} vertices) for floor clipping")
        
        # ── Step 3: Detect floor objects via SAM3, then clip to the shrunk hull ──
        for obj in floor_objs:
            phrases = obj.get("phrases", [])
            if not phrases:
                continue
            try:
                detections = call_sam3_for_image(img, phrases)
            except Exception as e:
                print(f"❌ SAM3 Failed for {zone_id} floor: {e}")
                continue
            if not detections:
                print(f" -> No detections found for floor phrases.")
                continue
                
            best_label = ""
            highest_valid_score = -1.0
            found_valid_mask = False
            
            for det in detections:
                score = det.get("score", 0.0)
                if score < 0.75:
                    continue
                
                mask_b64 = det.get("mask_b64")
                if not mask_b64: continue
                full_mask = decode_mask(mask_b64)
                overlap = full_mask[c_y1:c_y2, c_x1:c_x2]
                if not np.any(overlap > 0):
                    continue
                found_valid_mask = True
                label = det.get("label", "")
                if score > highest_valid_score:
                    highest_valid_score = score
                    best_label = label
                
                # Clip SAM3's floor mask to the shrunk boundary hull
                # Save raw floor mask for the red warning zone
                raw_floor_combined[full_mask > 0] = True
                
                if interior_clip is not None:
                    full_mask = cv2.bitwise_and(full_mask, interior_clip)
                else:
                    # Fallback: clip to bounding box if no boundary was detected
                    localized_mask = np.zeros_like(full_mask)
                    localized_mask[c_y1:c_y2, c_x1:c_x2] = full_mask[c_y1:c_y2, c_x1:c_x2]
                    full_mask = localized_mask
                    
                floor_masks_to_apply.append(full_mask)
                
            if found_valid_mask:
                print(f" -> Best mask applied for floor was '{best_label}' with score {highest_valid_score:.2f}")
            else:
                print(f" -> No overlapping masks found for floor inside the bounding box.")

        # Apply Floor masks (semi-transparent blue overlay)
        for m in floor_masks_to_apply:
            final_overlay[m > 0] = COLOR_FLOOR
            blended_mask[m > 0] = True
            
        # Accumulate Boundary masks — PURE SOLID BLACK
        for m in boundary_masks_to_apply:
            boundary_combined[m > 0] = True
            
    # Blend floor overlay with original image (semi-transparent)
    alpha = 0.6
    blended_img = cv2.addWeighted(img, 1.0 - alpha, final_overlay, alpha, 0)
    
    # Apply floor blend where blended_mask is True BUT boundary is not
    floor_only = blended_mask & (~boundary_combined)
    img[floor_only] = blended_img[floor_only]
    
    # --- RED WARNING ZONE LOGIC ---
    restricted_area = blended_mask | boundary_combined
    restricted_area_uint8 = np.zeros(img.shape[:2], dtype=np.uint8)
    restricted_area_uint8[restricted_area] = 255
    
    # DILATION FACTOR: Controls how many pixels the red warning area extends outward
    red_zone_thickness = 60  # <-- CHANGE THIS VALUE TO MAKE RED ZONE BIGGER/SMALLER
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (red_zone_thickness, red_zone_thickness))
    expanded_area = cv2.dilate(restricted_area_uint8, kernel, iterations=1)
    
    # Warning ring is the expanded area MINUS the restricted area, strictly on the floor
    warning_ring = (expanded_area > 0) & (~restricted_area)
    warning_only = warning_ring & raw_floor_combined
    
    warning_overlay = np.zeros_like(img)
    warning_overlay[warning_only] = COLOR_WARNING
    warning_blended = cv2.addWeighted(img, 1.0 - alpha, warning_overlay, alpha, 0)
    img[warning_only] = warning_blended[warning_only]
    
    # Stamp boundary pixels as PURE SOLID BLACK directly (no alpha)
    img[boundary_combined] = (0, 0, 0)
    
    out_name = f"PipelineVer2_output_{os.path.basename(args.image)}"
    out_path = os.path.join("PipelineVer2", "outputs", out_name)
    cv2.imwrite(out_path, img)
    print(f"\n✅ PipelineVer2 complete! Saved to {out_path}")

if __name__ == "__main__":
    main()
