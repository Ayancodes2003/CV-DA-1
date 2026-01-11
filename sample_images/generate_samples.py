"""Generate simple synthetic images containing geometric shapes for testing."""
import os
import cv2
import numpy as np

os.makedirs(os.path.dirname(__file__), exist_ok=True)

W, H = 800, 600
BG = 255

def save(img, name):
    path = os.path.join(os.path.dirname(__file__), name)
    cv2.imwrite(path, img)
    print(f"Saved {path}")

img1 = np.full((H, W, 3), BG, dtype=np.uint8)
pts = np.array([[100, 150], [40, 300], [160, 300]])
cv2.drawContours(img1, [pts], 0, (0, 0, 0), -1)
center = (300, 220)
box = cv2.boxPoints((center, (120, 120), 30)).astype(int)
cv2.drawContours(img1, [box], 0, (0, 0, 0), -1)
cv2.rectangle(img1, (420, 140), (620, 260), (0, 0, 0), -1)
pent = np.array([[100, 420], [60, 480], [100, 540], [180, 540], [220, 480]])
cv2.drawContours(img1, [pent], 0, (0, 0, 0), -1)
cv2.circle(img1, (520, 440), 70, (0, 0, 0), -1)
save(img1, 'shapes_simple.png')

img2 = np.full((H, W, 3), BG, dtype=np.uint8)
for x in range(50, 750, 100):
    for y in range(50, 550, 100):
        r = (x + y) % 40 + 10
        cv2.circle(img2, (x, y), r, (0, 0, 0), -1)
save(img2, 'shapes_many.png')

img3 = np.full((H, W, 3), BG, dtype=np.uint8)
cv2.rectangle(img3, (100, 100), (450, 400), (0, 0, 0), -1)
cv2.circle(img3, (300, 250), 140, (255, 255, 255), -1)
cv2.ellipse(img3, (600, 350), (120, 80), 30, 0, 360, (0, 0, 0), -1)
save(img3, 'shapes_overlap.png')
print('Done generating sample images.')
