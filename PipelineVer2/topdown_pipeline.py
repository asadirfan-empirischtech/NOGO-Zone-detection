import cv2
import math
import argparse
import numpy as np

# Global variables for GUI
clicked_points = []
img_display = None
img_original = None
meters_per_pixel = 0.0

def compute_distance(p1, p2):
    # Standard Euclidean distance in pixels
    pix_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Convert directly to meters using the top-down scaling factor
    real_dist = pix_dist * meters_per_pixel
    return real_dist

def redraw():
    global img_display
    img_display = img_original.copy()
    
    # Draw points
    for pt in clicked_points:
        cv2.circle(img_display, pt, 5, (0, 0, 255), -1)
        
    # Draw lines and distances
    for i in range(0, len(clicked_points) - 1, 2):
        p1 = clicked_points[i]
        p2 = clicked_points[i+1]
        cv2.line(img_display, p1, p2, (0, 255, 0), 2)
        
        dist = compute_distance(p1, p2)
        
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        text = f"{dist:.2f}m"
        # Draw text with outline for visibility
        cv2.putText(img_display, text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img_display, text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow("Top-Down Measurement Engine", img_display)

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        if len(clicked_points) % 2 == 0:
            p1 = clicked_points[-2]
            p2 = clicked_points[-1]
            dist = compute_distance(p1, p2)
            print(f"Segment Measured: {dist:.2f} meters")
        redraw()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(clicked_points) > 0:
            clicked_points.pop()
            print("Undo last click.")
            redraw()

def main():
    global img_original, img_display, meters_per_pixel
    
    parser = argparse.ArgumentParser(description="Measurement Engine for Perfect Top-Down Views")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--height", type=float, default=5.0, help="Camera height in meters (default: 5.0)")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera horizontal FOV in degrees (default: 60.0)")
    args = parser.parse_args()

    # Load Image
    img_original = cv2.imread(args.image)
    if img_original is None:
        print(f"Error: Could not load image at {args.image}")
        return
        
    h, w = img_original.shape[:2]
    
    # 1. Calculate focal length in pixels using the known FOV
    fov_rad = math.radians(args.fov)
    focal_length_px = (w / 2.0) / math.tan(fov_rad / 2.0)
    
    # 2. Calculate the direct scaling factor (meters per pixel)
    # Because it's a perfect top down view (Pitch = 90 deg), similar triangles apply perfectly!
    meters_per_pixel = args.height / focal_length_px
    
    print("\n=========================================")
    print("      TOP-DOWN MEASUREMENT ENGINE        ")
    print("=========================================")
    print(f"Image Resolution: {w}x{h}")
    print(f"Camera Height: {args.height} meters")
    print(f"Assumed FOV: {args.fov} degrees")
    print(f"Focal Length: {focal_length_px:.2f} px")
    print(f"Scale: 1 pixel = {meters_per_pixel:.4f} meters")
    print("-----------------------------------------")
    print("1. LEFT CLICK two points to measure distance.")
    print("2. RIGHT CLICK to undo.")
    print("3. Press 'q' to exit.")
    print("=========================================\n")

    img_display = img_original.copy()
    cv2.namedWindow("Top-Down Measurement Engine", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Top-Down Measurement Engine", mouse_callback)
    
    redraw()
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Exiting pipeline. Thank you!")

if __name__ == "__main__":
    main()
