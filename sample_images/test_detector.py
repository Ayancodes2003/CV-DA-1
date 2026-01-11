import os
import glob
import cv2
import sys
# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shape_detector import ShapeDetector

IMG_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(IMG_DIR, 'annotated')
os.makedirs(OUT_DIR, exist_ok=True)

sd = ShapeDetector()

for img_path in glob.glob(os.path.join(IMG_DIR, '*.png')):
    img = cv2.imread(img_path)
    if img is None:
        continue
    annotated, edges, detected = sd.detect_shapes(img, canny_thresh1=50, canny_thresh2=150, min_area=100)
    print(f"{os.path.basename(img_path)} -> {len(detected)} objects detected")
    for d in detected:
        print('  ', d)
    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(OUT_DIR, f'{base}_annotated.png'), annotated)
    cv2.imwrite(os.path.join(OUT_DIR, f'{base}_edges.png'), edges)

print('Done running detector on sample images. Annotated outputs are in sample_images/annotated/')
