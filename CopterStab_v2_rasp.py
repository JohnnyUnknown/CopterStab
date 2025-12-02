import cv2
import numpy as np
from libcamera import controls
from picamera2 import Picamera2
import time

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

cv2.namedWindow("position", cv2.WINDOW_NORMAL)
cv2.resizeWindow("position", 500, 500)

camera.configure(video_config)
camera.start()
plot_size = 1500
center_x = plot_size // 2
center_y = plot_size // 2
scale = 2.0
dx = 0.0
dy = 0.0

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



def detect_and_compute(frame, detector):
    return detector.detectAndCompute(frame, None)

def match_descriptors(desc1, desc2, is_sift=True):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    norm = cv2.NORM_L2 if is_sift else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in matches if m.distance < 0.5 * n.distance]

def compute_relative_angle(kp1, kp2, matches):
    if len(matches) < 4:
        return 0.0
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return 0.0
    angle_rad = np.arctan2(M[1, 0], M[0, 0])
    return -np.degrees(angle_rad)  # CW = positive


draw_plot()


def main():
    method_name = "SIFT"
    is_sift = method_name == "SIFT"
    detector = cv2.SIFT_create(nOctaveLayers=3) if is_sift else cv2.ORB_create(nfeatures=200)

    # Первый кадр
    frame = camera.capture_array()
    
    frame = cv2.resize(frame, (int(img_size / 9 * 16), img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(frame, (5, 5), 0)
    prev_kp, prev_desc = detect_and_compute(prev_gray, detector)

    H, W = prev_gray.shape
    first_center = np.array([[[W / 2, H / 2]]], dtype=np.float32)
    H_cumulative = np.eye(3, dtype=np.float32)
    angle_accum = 0.0

    # === Цикл обработки ===
    start_time = time.perf_counter()
    frame_count = 0
    fps = 0

    while True:
        frame = camera.capture_array()

        # Обработка кадра
        frame = cv2.resize(frame, (int(img_size / 9 * 16), img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(frame, (5, 5), 0)
        curr_kp, curr_desc = detect_and_compute(curr_gray, detector)

        matches = match_descriptors(prev_desc, curr_desc, is_sift)

        # Инициализация
        center = None
        tracking_ok = False

        if matches and len(matches) >= 6:
            # Гомография для центра
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
            H_local, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if H_local is not None:
                H_total = H_local @ H_cumulative
                try:
                    proj = cv2.perspectiveTransform(first_center, H_total)
                    center = (int(proj[0, 0, 0]), int(proj[0, 0, 1]))
                    tracking_ok = True
                except:
                    tracking_ok = False

            # Угол между соседними кадрами
            rel_angle = compute_relative_angle(prev_kp, curr_kp, matches)
            angle_accum = (angle_accum + rel_angle) % 360

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
            cv2.circle(display, center, 2, (0, 255, 0), 2)
            dx = center[0] - W / 2
            dy = H / 2 - center[1]
            text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, angle:{angle_accum:.1f} | FPS: {fps}"
            cv2.putText(display, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(text)
        else:
            print("❌ Трекинг потерян")

        cv2.imshow("Tracking", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
