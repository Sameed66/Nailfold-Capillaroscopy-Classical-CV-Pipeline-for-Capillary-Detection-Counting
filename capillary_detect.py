
import cv2
def calculate_luminance(image):
    return image.mean()

def analyze_images(image_batch, corresponding_paths):
    analysis_summary = {}

    for idx, (gray_img, img_path) in enumerate(zip(image_batch, corresponding_paths)):
        low_thresh = 60
        high_thresh = 90

        brightness = calculate_luminance(gray_img)
        img_title = os.path.basename(img_path)
        print(f"Luminance for {img_title}: {brightness}")

        if brightness < 88:
            equalized = cv2.equalizeHist(gray_img)
            clahe_obj = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
            enhanced = clahe_obj.apply(equalized)
            thresholded = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 125, 20)
        else:
            equalized = cv2.equalizeHist(gray_img)
            clahe_obj = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(6, 6))
            enhanced = clahe_obj.apply(gray_img)
            thresholded = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 125, 50)

        opened_1 = binary_opening(thresholded, square(4)).astype(np.uint8) * 255

        kernel_small = np.ones((3, 3), np.uint8)
        dilated_1 = cv2.dilate(opened_1, kernel_small, iterations=1)

        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened_2 = binary_opening(dilated_1, ellipse_kernel).astype(np.uint8) * 255

        hysteresis_output = filters.apply_hysteresis_threshold(~opened_2, low_thresh, high_thresh)
        opened_2 = hysteresis_output.astype(np.uint8) * 255

        grown = cv2.dilate(opened_2, ellipse_kernel, iterations=1)

        erosion_kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(grown, erosion_kernel, iterations=1)

        morph_closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, erosion_kernel)
        morph_opened = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        fully_dilated = cv2.dilate(morph_opened, np.ones((3, 3), np.uint8), iterations=2)

        label_count, label_map = cv2.connectedComponents(dilated_1)
        print(label_count)
        overlay_img = cv2.cvtColor(fully_dilated.copy(), cv2.COLOR_GRAY2BGR)

        total_capillaries = 0
        max_capillaries = 10
        min_area = 260
        merge_distance = 20.0

        def get_slope_dist(cnt1, cnt2):
            M1 = cv2.moments(cnt1)
            M2 = cv2.moments(cnt2)
            cx1 = int(M1["m10"] / M1["m00"])
            cy1 = int(M1["m01"] / M1["m00"])
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])
            slope_val = (cy2 - cy1) / (cx2 - cx1)
            dist = np.hypot(cx2 - cx1, cy2 - cy1)
            return slope_val, dist

        def is_horizontal_shape(cnt):
            return len(cnt) >= 5 and abs(cv2.fitEllipse(cnt)[-1] - 90) < 15

        def is_tube_like(cnt):
            a = cv2.contourArea(cnt)
            p = cv2.arcLength(cnt, True)
            return a / p > 0.5

        for label in range(1, label_count):
            segment = np.zeros_like(morph_opened)
            segment[label_map == label] = 255
            cnts, _ = cv2.findContours(segment, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            filtered_cnts = [c for c in cnts if cv2.contourArea(c) > min_area and not is_horizontal_shape(c)]

            if total_capillaries >= max_capillaries:
                filtered_cnts = sorted(filtered_cnts, key=cv2.contourArea, reverse=True)[:max_capillaries]

            merged = []
            for i in range(len(filtered_cnts)):
                for j in range(i + 1, len(filtered_cnts)):
                    slope, dist = get_slope_dist(filtered_cnts[i], filtered_cnts[j])
                    if 0.5 < abs(slope) < 1.5 and dist < merge_distance:
                        merged.append(filtered_cnts[i])
                        break
                else:
                    merged.append(filtered_cnts[i])

            for cnt in merged:
                if is_tube_like(cnt):
                    total_capillaries += 1
                    cv2.drawContours(overlay_img, [cnt], -1, (0, 255, 0), 2)
                if total_capillaries >= max_capillaries:
                    break
            if total_capillaries >= max_capillaries:
                break

        analysis_summary[img_title] = total_capillaries

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title(f"Original: {img_title}")

        axes[1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Detected Capillaries: {total_capillaries}")

        plt.tight_layout()
        plt.show()

    return analysis_summary
