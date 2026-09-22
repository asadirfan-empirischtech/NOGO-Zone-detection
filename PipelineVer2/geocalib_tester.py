import torch
import argparse
import sys
import ssl
import urllib.request
import cv2
import numpy as np

# Fix macOS SSL Certificate error for torch.hub downloading weights
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from geocalib import GeoCalib
except ImportError:
    print("Error: GeoCalib not installed. Please run: pip install git+https://github.com/cvg/GeoCalib")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automated GeoCalib AI: Fisheye Correction & Pitch/FOV Extraction")
    parser.add_argument("--image", required=True, help="Path to input CCTV image")
    args = parser.parse_args()

    print(f"Loading GeoCalib AI (Distortion Model) to analyze vanishing points in '{args.image}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model with 'distorted' weights to automatically handle Fisheye lenses!
    model = GeoCalib(weights='distorted').to(device)

    # Analyze the CCTV frame and infer the radial fisheye curve
    img_tensor = model.load_image(args.image).to(device)
    result = model.calibrate(img_tensor, camera_model='simple_radial')

    print("\n✅ Calibration Complete!")
    print("\n--- GeoCalib Reverse-Engineered Parameters ---")
    
    # Convert FOV from radians to degrees
    v_fov = torch.rad2deg(result["camera"].vfov).item()
    h_fov = torch.rad2deg(result["camera"].hfov).item()
    
    pitch = torch.rad2deg(result["gravity"].pitch).item()
    roll = torch.rad2deg(result["gravity"].roll).item()

    print(f"Camera Field of View (Vertical):   {v_fov:.2f} degrees")
    print(f"Camera Field of View (Horizontal): {h_fov:.2f} degrees")
    print(f"Downward Pitch Angle:              {pitch:.2f} degrees")
    print(f"Camera Roll (Tilt):                {roll:.2f} degrees")
    print("----------------------------------------------")
    
    print("\n-> Automatically undistorting Fisheye lens...")
    # Add a batch dimension [1, 3, H, W] for grid_sampler
    img_tensor_batch = img_tensor.unsqueeze(0)
    
    # Let GeoCalib's camera math flatten the image perfectly
    undistorted_tensor = result["camera"].undistort_image(img_tensor_batch)
    
    # Convert tensor back to an OpenCV BGR image and save it
    img_np = (undistorted_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    out_path = "auto_undistorted.jpg"
    cv2.imwrite(out_path, img_np)
    print(f"✅ Saved perfectly flat (pinhole) image to {out_path}!")
    
    print("\nNow you can run the multi-point tester automatically:")
    print(f"python3 PipelineVer2/multi_point_tester.py --image '{out_path}' --pitch {pitch:.2f} --fov {h_fov:.2f}")

if __name__ == "__main__":
    main()
