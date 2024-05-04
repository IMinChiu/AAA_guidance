import cv2
import json
import argparse
from pathlib import Path


# get the image from arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# check if the image does exist
img_p = Path(args["image"])
assert img_p.exists(), "Image does not exist"

# Read the image
img = cv2.imread(args["image"])

# Select the ROI from the image
ROI = cv2.selectROI("Image", img, False, False)

# Append the ROI coordinates to the csv file
# header: filename, ori_width, ori_height, roi_x, roi_y, roi_width, roi_height
with open("roi.csv", "a") as f:
    # if no file exists, create a new one with the header
    if f.tell() == 0:
        f.write("filename,ori_width,ori_height,roi_x,roi_y,roi_width,roi_height\n")
    ori_w, ori_h = img.shape[1], img.shape[0]
    f.write(f"{img_p.name},{ori_w},{ori_h},{ROI[0]},{ROI[1]},{ROI[2]},{ROI[3]}\n")

# Display cropped image
cropped = img[ROI[1] : ROI[1] + ROI[3], ROI[0] : ROI[0] + ROI[2]]
cv2.imshow("Cropped Image", cropped)
cv2.waitKey(0)
