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

# cv2.namedWindow("position", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("position", 500, 500)

camera.configure(video_config)
camera.start()
# plot_size = 1500
# center_x = plot_size // 2
# center_y = plot_size // 2
# scale = 2.0
# dx = 0.0
# dy = 0.0

min_kp, max_kp = 100, 500
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


def new_treshold_matcher(threshold, len_match, max_match):
    if len_match < min_match:
        threshold += 0.05
        if threshold > 0.7: 
            return 0.7
    if len_match > max_match:
        threshold -= 0.05
        if threshold < 0.4: 
            return 0.4
    return threshold


def check_shifts(dx, dy, shifts):
    #print("Before", dx, dy)
    if abs(shifts[2][0] - dx) > 15:
        dx = np.mean([shifts[0][0], shifts[1][0], shifts[2][0], dx])
        #print("After dx", dx)
    if abs(shifts[2][1] - dy) > 15:
        dy = np.mean([shifts[0][1], shifts[1][1], shifts[2][1], dy])
        #print("After dy", dy)
    return dx, dy


def draw_plot():
    plot = np.zeros((plot_size, plot_size, 3), dtype=np.uint8)
    plot[:] = (30, 30, 30)
    cv2.line(plot, (0, center_y), (plot_size-1, center_y), (80,80,80), 2)
    cv2.line(plot, (center_x, 0), (center_x, plot_size-1), (80,80,80), 2)
    for i in range(0, plot_size, 50):
        cv2.line(plot, (i, 0), (i, plot_size-1), (50,50,50), 1)
        cv2.line(plot, (0, i), (plot_size-1, i), (50,50,50), 1)
    cv2.circle(plot, (center_x, center_y), 5, (0,255,0), -1)
    px = int(center_x + dx * scale)
    py = int(center_y - dy * scale)
    cv2.circle(plot, (px, py), 12, (0,0,255), -1)
    cv2.circle(plot, (px, py), 12, (255,255,255), 3)
    cv2.putText(plot, f"X: {dx:+8.1f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(plot, f"Y: {dy:+8.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("position", plot)

# draw_plot()


def main():
    method_name = "SIFT"
    is_sift = method_name == "SIFT"
    contrast = 0.04
    threshold_matcher = 0.5
    detector = cv2.SIFT_create(contrastThreshold=contrast) if is_sift else cv2.ORB_create(nfeatures=200)

    # Первый кадр
    frame = camera.capture_array()
    
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)

    H, W = prev_gray.shape
    first_center = np.array([[[W / 2, H / 2]]], dtype=np.float32)
    shifts_buffer = deque([(0, 0), (0, 0), (0, 0)], maxlen=3)
    H_cumulative = np.eye(3, dtype=np.float32)
    # angle_accum = 0.0

    # === Цикл обработки ===
    start_time = time.perf_counter()
    frame_count = 0
    fps = 0

    while True:
        frame = camera.capture_array()

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)

        if len(curr_kp) < min_kp or len(curr_kp) > max_kp:
            detector, contrast = reinitialize_SIFT(contrast, len(curr_kp))
            prev_kp, prev_desc = detector.detectAndCompute(curr_gray, None)
            continue

        max_match = int(min(len(prev_kp), len(curr_kp)) * 0.99)
        matches = match_descriptors(prev_desc, curr_desc, threshold_matcher, is_sift)

        if len(matches) < min_match or len(matches) > max_match:
            threshold_matcher = new_treshold_matcher(threshold_matcher, len(matches), max_match)
            matches = match_descriptors(prev_desc, curr_desc, threshold_matcher, is_sift)
            # print(threshold_matcher, len(matches))

        # Инициализация
        center = None
        tracking_ok = False

        if matches and len(matches) >= min_match:
            # Гомография для центра
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
            H_local, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if H_local is not None:
                H_total = H_local @ H_cumulative
                try:
                    proj = cv2.perspectiveTransform(first_center, H_total)
                    center = (proj[0, 0, 0], proj[0, 0, 1])
                    tracking_ok = True
                except:
                    tracking_ok = False

            # # Угол между соседними кадрами
            # rel_angle = compute_relative_angle(prev_kp, curr_kp, matches)
            # angle_accum = (angle_accum + rel_angle) % 360

            # Обновление опорного кадра при успешном трекинге
            if tracking_ok:
                prev_gray, prev_kp, prev_desc = curr_gray, curr_kp, curr_desc
                H_cumulative = H_total
            
        # FPS
        frame_count += 1
        if time.perf_counter() - start_time >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = time.perf_counter()

        # Визуализация
        display = curr_gray.copy()
        if center:
            dx = center[0] - W / 2
            dy = H / 2 - center[1]
            dx, dy = check_shifts(dx, dy, shifts_buffer)
            #shifts_buffer.append((dx, dy))
        else:
            #dx, dy = check_shifts(None, None, shifts_buffer)
            print("❌ Трекинг потерян")
            
        shifts_buffer.append((dx, dy))
            
        cv2.circle(display, (int(proj[0, 0, 0]), int(proj[0, 0, 1])), 2, (0, 0, 255), 2)
        # text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, angle:{angle_accum:.1f} | FPS: {fps}"
        text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, FPS: {fps}"
        cv2.putText(display, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        center_x, center_y = int(W / 2), int(H / 2)

        # Центральные оси (выделенные синие линии)
        cv2.line(display, (center_x, 0), (center_x, img_size), (255, 0, 0), 1)
        cv2.line(display, (0, center_y), (img_size, center_y), (255, 0, 0), 1)
        # -----------------------------------------------------------------

        print(f"{text} | {contrast=:.2f}, {threshold_matcher=:.2f}, kp1={len(prev_kp)}, kp2={len(curr_kp)}, matches={len(matches)}")
        

        cv2.imshow("Tracking", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
