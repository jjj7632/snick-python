def fallback_process_image(self, frame_number, image_data):
    import cv2
    import numpy as np

    # Camera constants
    FOCAL_PX = 10.0 / 0.006
    CX       = 1920 / 2.0
    CY       = 1080 / 2.0
    BASELINE = 0.10

    # Images arrive sequentially, cache the left frame until the right frame arrives
    if not hasattr(self, "_fallback_left_frame"):
        self._fallback_left_frame = (frame_number, image_data)
        return {"frame_number": frame_number, "x": 0.0, "y": 0.0, "z": 0.0, "reasonable": False, "likely_out": False}

    left_frame_number, left_image = self._fallback_left_frame
    right_image = image_data
    del self._fallback_left_frame

    # Convert to HSV and threshold to isolate tennis ball yellow-green pixels
    hsv_lower = np.array([25, 80, 80],   dtype=np.uint8)
    hsv_upper = np.array([65, 255, 255], dtype=np.uint8)
    left_mask  = cv2.inRange(cv2.cvtColor(left_image,  cv2.COLOR_RGB2HSV), hsv_lower, hsv_upper)
    right_mask = cv2.inRange(cv2.cvtColor(right_image, cv2.COLOR_RGB2HSV), hsv_lower, hsv_upper)

    # Find the most circular contour in the mask and return its pixel centroid
    def get_centroid(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area < 50 or perimeter == 0:       # raised from 20 to 50
                continue
            circularity = (4.0 * np.pi * area) / (perimeter ** 2)
            if circularity > best_score:
                best, best_score = c, circularity
        if best is None or best_score < 0.4:      # lowered from 0.6 to 0.4
            return None
        m = cv2.moments(best)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    left_centroid  = get_centroid(left_mask)
    right_centroid = get_centroid(right_mask)

    # Ball not detected in one or both frames
    if left_centroid is None or right_centroid is None:
        return {"frame_number": left_frame_number, "x": 0.0, "y": 0.0, "z": 0.0, "reasonable": False, "likely_out": False}

    # Compute real world x/y/z
    u_left, v_left = left_centroid
    u_right        = right_centroid[0]
    disparity      = u_left - u_right

    if abs(disparity) < 1.0:
        return {"frame_number": left_frame_number, "x": 0.0, "y": 0.0, "z": 0.0, "reasonable": False, "likely_out": False}

    z = (FOCAL_PX * BASELINE) / disparity
    x = (u_left - CX) * z / FOCAL_PX
    y = (v_left - CY) * z / FOCAL_PX

    return {
        "frame_number": left_frame_number,
        "x": x,
        "y": y,
        "z": z,
        "reasonable": True,
        "likely_out": False
    }