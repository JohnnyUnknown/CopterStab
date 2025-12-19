import cv2
import numpy as np
import time
from collections import deque

min_kp, max_kp = 100, 300
min_match, max_match = 6, max_kp

def match_descriptors(desc1, desc2, threshold_matcher, is_sift=True):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    norm = cv2.NORM_L2 if is_sift else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in matches if m.distance < threshold_matcher * n.distance]
    

def compute_relative_angle(src, dst):
    M, inliers_mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers_mask.sum() < 10:
        return 0.0
    angle_rad = np.arctan2(M[1, 0], M[0, 0])
    angle = -np.degrees(angle_rad)
    return angle if abs(angle) >= 0.01 else 0


def reinitialize_SIFT(contrast, len_kp):
    if len_kp < min_kp:
        contrast -= 0.01
    if len_kp > max_kp:
        contrast += 0.01
    return cv2.SIFT_create(contrastThreshold=contrast), contrast


def new_threshold_matcher(threshold, len_match, max_match):
    if len_match < min_match:
        threshold += 0.05
        if threshold > 0.7: 
            return 0.7
    if len_match > max_match:
        threshold -= 0.05
        if threshold < 0.4: 
            return 0.4
    return threshold


def filter_outlier_shift(dx, dy, history, max_deviation=30.0):
    hist_x = [p[0] for p in history]
    hist_y = [p[1] for p in history]
    mean_x, mean_y = np.mean(hist_x), np.mean(hist_y)
    is_outlier = False

    if abs(dx - history[2][0]) > max_deviation:
        print("outlier dx", dx, history[2][0], mean_x)
        is_outlier = True
        dx = history[2][0]  
    if abs(dy - history[2][1]) > max_deviation:
        print("outlier dy", dy, history[2][1], mean_y)
        is_outlier = True
        dy = history[2][1]
    return dx, dy, is_outlier


def main():
    video_path = r"C:\My\Projects\images\stab_test.mp4"
    img_h = 300
    img_w = int(img_h / 9 * 16)

    method_name = "SIFT"
    is_sift = method_name == "SIFT"
    contrast = 0.04
    threshold_matcher = 0.5
    detector = cv2.SIFT_create(contrastThreshold=contrast) if is_sift else cv2.ORB_create(nfeatures=200)

    #---------------------Получение кадра--------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Не удалось открыть видео")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Видео пустое")
    frame = cv2.resize(frame, (img_w, img_h))
    #---------------------------------------------------------------------

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)

    H, W = prev_gray.shape
    first_center = np.array([[[W / 2, H / 2]]], dtype=np.float32)
    shifts_buffer = deque([(0, 0)] * 3, maxlen=3)
    H_cumulative = np.eye(3, dtype=np.float32)
    # angle_accum = 0.0

    # === Цикл обработки ===
    start_time = time.perf_counter()
    frame_count = 0
    fps = 0

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #---------------------Получение кадра--------------------------------
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_w, img_h))
        #---------------------------------------------------------------------

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)

        if len(curr_kp) < min_kp or len(curr_kp) > max_kp:
            detector, contrast = reinitialize_SIFT(contrast, len(curr_kp))
            curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)

        max_match = int(min(len(prev_kp), len(curr_kp)) * 0.99)
        matches = match_descriptors(prev_desc, curr_desc, threshold_matcher, is_sift)

        if len(matches) < min_match or len(matches) > max_match:
            threshold_matcher = new_threshold_matcher(threshold_matcher, len(matches), max_match)
            matches = match_descriptors(prev_desc, curr_desc, threshold_matcher, is_sift)

        # Инициализация
        center = None

        if matches and len(matches) >= min_match:
            # Гомография для центра
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
            H_local, inliers_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if H_local is not None and inliers_mask.sum() >= 10:
                H_total = H_local @ H_cumulative
                try:
                    proj = cv2.perspectiveTransform(first_center, H_total)
                    center = (proj[0, 0, 0], proj[0, 0, 1])
                    dx = center[0] - W / 2
                    dy = H / 2 - center[1]
                    dx, dy, is_outlier = filter_outlier_shift(dx, dy, shifts_buffer)
                    shifts_buffer.append((dx, dy))
                    if not is_outlier:
                        H_cumulative = H_total
                except:
                    print("Трекинг потерян")

            # # Угол между соседними кадрами
            # rel_angle = compute_relative_angle(prev_kp, curr_kp, matches)
            # angle_accum = (angle_accum + rel_angle) % 360

            prev_gray, prev_kp, prev_desc = curr_gray, curr_kp, curr_desc
            
        # FPS
        frame_count += 1
        if time.perf_counter() - start_time >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = time.perf_counter()

        # Визуализация
        display = frame.copy()
        if center and not is_outlier:
            cv2.circle(display, (int(W/2+dx), int(H/2-dy)), 2, (0, 0, 255), 2)
            # text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, angle:{angle_accum:.1f} | FPS: {fps}"
            text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, FPS: {fps}"
            cv2.putText(display, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # ------------ Сетка 50x50, начинающаяся от центра ---------------
            grid_size = 50
            center_x, center_y = int(W / 2), int(H / 2)
            # Вертикальные линии сетки от центра
            x = center_x
            while x >= 0:
                cv2.line(display, (x, 0), (x, img_h), (100, 100, 100), 1)
                x -= grid_size
            x = center_x + grid_size
            while x < img_w:
                cv2.line(display, (x, 0), (x, img_h), (100, 100, 100), 1)
                x += grid_size
            # Горизонтальные линии сетки от центра
            y = center_y
            while y >= 0:
                cv2.line(display, (0, y), (img_w, y), (100, 100, 100), 1)
                y -= grid_size
            y = center_y + grid_size
            while y < img_h:
                cv2.line(display, (0, y), (img_w, y), (100, 100, 100), 1)
                y += grid_size

            # Центральные оси (выделенные синие линии)
            cv2.line(display, (center_x, 0), (center_x, img_h), (255, 0, 0), 1)
            cv2.line(display, (0, center_y), (img_w, center_y), (255, 0, 0), 1)
            # -----------------------------------------------------------------

            print(f"{text} | {contrast=:.2f}, {threshold_matcher=:.2f}, kp1={len(prev_kp)}, kp2={len(curr_kp)}, matches={len(matches)}")
        else:
            print("❌ Трекинг потерян")

        cv2.imshow("Tracking", display)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()