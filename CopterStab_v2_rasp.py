import cv2
import numpy as np
from libcamera import controls
from picamera2 import Picamera2
import time
from collections import deque

camera = Picamera2()
curr_fps = 120
img_size = 300
video_config = camera.create_video_configuration(
    main={"size": (img_size, img_size), "format": 'RGB888'},
    controls={"FrameRate": curr_fps,
            "AfMode": controls.AfModeEnum.Manual, 
            "LensPosition": 0.0,
            "AfSpeed": controls.AfSpeedEnum.Fast}
)

camera.configure(video_config)
camera.start()

min_kp = 80
max_kp = min_kp*6
min_match, max_match = 20, max_kp


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
        contrast -= 0.005
    if len_kp > max_kp:
        contrast += 0.005
    return cv2.SIFT_create(contrastThreshold=contrast), contrast


def new_threshold_matcher(threshold, len_match, max_match):
    if len_match < min_match:
        threshold += 0.05
        if threshold > 0.7: 
            return 0.7
    if len_match > max_match:
        threshold -= 0.05
        if threshold < 0.3: 
            return 0.3
    return threshold


def level_deviation(history, cnt_outliers=0, altitude = 1):
    correction = 1 + cnt_outliers * 0.8
    max_dev = max((history[2][0]-history[1][0]), (history[2][1]-history[1][1]))
    if max_dev < 15:
        max_dev = 15 
    lev_dev = max_dev * correction
    print(f"{lev_dev=}, {correction=}, {max_dev=}, {history[2]}, {history[1]}")
    return lev_dev


def filter_outlier_shift(dx, dy, history, max_deviation=20.0):
    is_outlier = False
    if abs(dx - history[2][0]) > max_deviation:
        print("outlier dx", dx, history[2][0])
        is_outlier = True
        dx = history[2][0]  
    if abs(dy - history[2][1]) > max_deviation:
        print("outlier dy", dy, history[2][1])
        is_outlier = True
        dy = history[2][1]
    return dx, dy, is_outlier



def main():
    method_name = "SIFT"
    is_sift = method_name == "SIFT"
    contrast = 0.04
    threshold_matcher = 0.5
    detector = cv2.SIFT_create(contrastThreshold=contrast) if is_sift else cv2.ORB_create(nfeatures=200)

    while True:
        #---------------------Получение кадра--------------------------------
        frame = camera.capture_array()
        #---------------------------------------------------------------------

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 1)
        prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)

        if (len(prev_kp) < min_kp or len(prev_kp) > max_kp) and contrast > 0.01 and contrast < 0.1:
            detector, contrast = reinitialize_SIFT(contrast, len(prev_kp))
            curr_kp, curr_desc = detector.detectAndCompute(prev_gray, None)
        else:
            break
    
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)

    # H, W = prev_gray.shape
    first_center = np.array([[[img_size / 2, img_size / 2]]], dtype=np.float32)
    shifts_buffer = deque([(0, 0)] * 3, maxlen=3)
    H_cumulative = np.eye(3, dtype=np.float32)
    # angle_accum = 0.0

    start_time = time.perf_counter()
    frame_count = 0
    fps = 0
    frame_errors_cnt = 0

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #---------------------Получение кадра--------------------------------
        frame = camera.capture_array()
        #---------------------------------------------------------------------

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 1)
        curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)

        if (len(curr_kp) < min_kp or len(curr_kp) > max_kp) and contrast > 0.01 and contrast < 0.1:
            detector, contrast = reinitialize_SIFT(contrast, len(curr_kp))
            curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)

        # Коррекция max_match. 0.99 для небольшого диапазона.
        max_match = int(min(len(prev_kp), len(curr_kp)) * 0.99)
        matches = match_descriptors(prev_desc, curr_desc, threshold_matcher, is_sift)

        if len(matches) < min_match or len(matches) > max_match:
            threshold_matcher = new_threshold_matcher(threshold_matcher, len(matches), max_match)
            matches = match_descriptors(prev_desc, curr_desc, threshold_matcher, is_sift)

        center = None
        prev_len_kp = len(prev_kp)

        if matches and len(matches) >= 6:
            try:
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
                H_local, inliers_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            except cv2.error:
                frame_errors_cnt += 1
                # continue

            if H_local is not None and inliers_mask.sum() >= 10:
                H_total = H_local @ H_cumulative
                try:
                    proj = cv2.perspectiveTransform(first_center, H_total)
                    center = (proj[0, 0, 0], proj[0, 0, 1])
                    dx = center[0] - img_size / 2
                    dy = img_size / 2 - center[1]
                    deviation = level_deviation(history=shifts_buffer,
                                                cnt_outliers=frame_errors_cnt, 
                                                altitude = 1)
                    dx, dy, is_outlier = filter_outlier_shift(dx=dx, 
                                                              dy=dy, 
                                                              history=shifts_buffer, 
                                                              max_deviation=deviation)
                    shifts_buffer.append((dx, dy))
                    if not is_outlier:
                        H_cumulative = H_total
                        prev_len_kp = len(prev_kp)  # Информация о кол-ве точек в опорном кадре
                        prev_gray, prev_kp, prev_desc = curr_gray, curr_kp, curr_desc
                        frame_errors_cnt = 0
                except Exception as e:
                    print(f"\nТип исключения: {type(e).__name__}, сообщение: {str(e)}\n")

                # prev_gray, prev_kp, prev_desc = curr_gray, curr_kp, curr_desc

            # # Угол между соседними кадрами
            # rel_angle = compute_relative_angle(prev_kp, curr_kp, matches)
            # angle_accum = (angle_accum + rel_angle) % 360
            
        # FPS
        frame_count += 1
        if time.perf_counter() - start_time >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = time.perf_counter()

        # Визуализация
        display = frame.copy()
        if center and not is_outlier:
            cv2.circle(display, (int(img_size / 2 + dx), int(img_size / 2 - dy)), 2, (0, 0, 255), 2)
            # text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, angle:{angle_accum:.1f} | FPS: {fps}"
            text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, FPS: {fps}"
            cv2.putText(display, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Центральные оси (выделенные синие линии)
            center_x, center_y = int(img_size / 2), int(img_size / 2)
            cv2.line(display, (center_x, 0), (center_x, img_size), (255, 0, 0), 1)
            cv2.line(display, (0, center_y), (img_size, center_y), (255, 0, 0), 1)

            #print(f"{text} | contr={contrast:.3f}, thres={threshold_matcher:.2f}, kp1={prev_len_kp}, kp2={len(curr_kp)}, mat={len(matches)}")
        else:
            frame_errors_cnt += 1
            print(f"Трекинг потерян: {contrast=:.3f}, {threshold_matcher=:.2f}, kp1={prev_len_kp}, kp2={len(curr_kp)}, mat={len(matches)}")
            
        # shifts_buffer.append((dx, dy))

        cv2.imshow("Tracking", display)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
