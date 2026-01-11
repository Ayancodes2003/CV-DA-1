"""shape_detector.py (restored)

Module implementing a ShapeDetector class that detects geometric shapes
from an image using classical OpenCV contour-based methods.
"""

from typing import List, Tuple, Dict
import cv2
import numpy as np


class ShapeDetector:
    """Detect and classify simple geometric shapes in an image using contours."""

    def __init__(self) -> None:
        pass

    def detect_shapes(
        self,
        image: np.ndarray,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
        min_area: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """Detect shapes in the provided BGR image.

        Returns annotated image, edge image, and list of detected objects with
        'shape', 'area', and 'perimeter'.
        """
        annotated = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected: List[Dict[str, float]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = len(approx)

            shape_name = "Unidentified"
            if vertices == 3:
                shape_name = "Triangle"
            elif vertices == 4:
                rect = cv2.minAreaRect(approx)
                (width, height) = rect[1]
                if width == 0 or height == 0:
                    aspect_ratio = 0
                else:
                    aspect_ratio = float(width) / float(height) if width >= height else float(height) / float(width)
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_name = "Square"
                else:
                    shape_name = "Rectangle"
            elif vertices == 5:
                shape_name = "Pentagon"
            else:
                shape_name = "Circle"

            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cX = x + w // 2
                cY = y + h // 2

            label_shape = f"{shape_name}"
            label_stats = f"A:{area:.0f} P:{perimeter:.0f}"
            cv2.putText(annotated, label_shape, (cX - 40, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(annotated, label_stats, (cX - 40, cY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

            detected.append({
                "shape": shape_name,
                "area": round(area, 2),
                "perimeter": round(perimeter, 2),
            })

        detected.sort(key=lambda x: x["area"], reverse=True)
        return annotated, edges, detected


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python shape_detector.py <image_path>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Could not read image")
        sys.exit(1)
    sd = ShapeDetector()
    annotated, edges, detected = sd.detect_shapes(img)
    print(detected)
    cv2.imshow('Annotated', annotated)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
