import cv2
import numpy as np
import time

def detect_and_compute(frame, detector):
    return detector.detectAndCompute(frame, None)

def match_descriptors(desc1, desc2, is_sift=True):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    norm = cv2.NORM_L2 if is_sift else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in matches if m.distance < 0.5 * n.distance]

def estimate_affine_params(kp1, kp2, matches):
    if len(matches) < 6:
        return 0.0, 1.0
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=10.0)
    if M is None:
        return 0.0, 1.0
    a, c = M[0, 0], M[1, 0]
    angle_rad = np.arctan2(c, a)
    angle_deg = -np.degrees(angle_rad)  # CW = +
    scale_x = np.sqrt(a**2 + M[0,1]**2)
    scale_y = np.sqrt(c**2 + M[1,1]**2)
    scale = (scale_x + scale_y) / 2.0
    return angle_deg, scale


def main():
    video_path = r"C:\My\Projects\images\djelga_car2.mp4"
    img_size = 300
    method_name = "SIFT"
    is_sift = method_name == "SIFT"
    detector = cv2.SIFT_create(nOctaveLayers=3) if is_sift else cv2.ORB_create(nfeatures=200)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Не удалось открыть видео")

    # === Первый кадр ===
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Видео пустое")

    frame = cv2.resize(frame, (int(img_size / 9 * 16), img_size))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    first_kp, first_desc = detect_and_compute(first_gray, detector)

    H, W = first_gray.shape
    first_center = np.array([[[W / 2, H / 2]]], dtype=np.float32)
    H_cumulative = np.eye(3, dtype=np.float32)

    # === Опорный кадр (начинаем с первого) ===
    ref_gray = first_gray.copy()
    ref_kp, ref_desc = first_kp, first_desc

    # === Глобальная поправка угла: угол от первого кадра до текущего опорного ===
    global_angle_offset = 0.0

    # === Пороги масштаба ===
    SCALE_MIN, SCALE_MAX = 0.7, 1.4

    # === FPS ===
    start_time = time.perf_counter()
    frame_count = 0
    fps = 0

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (int(img_size / 9 * 16), img_size))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        curr_kp, curr_desc = detect_and_compute(curr_gray, detector)

        # === Совпадения с ОПОРНЫМ кадром ===
        matches_ref = match_descriptors(ref_desc, curr_desc, is_sift)
        center = None
        tracking_ok = False
        current_scale = 1.0
        angle_from_ref_to_curr = 0.0

        if matches_ref and len(matches_ref) >= 6:
            # Гомография для центра
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches_ref])
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches_ref])
            H_local, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if H_local is not None:
                H_total = H_local @ H_cumulative
                try:
                    proj = cv2.perspectiveTransform(first_center, H_total)
                    center = (int(proj[0, 0, 0]), int(proj[0, 0, 1]))
                    tracking_ok = True
                except:
                    tracking_ok = False

            # Угол и масштаб от опоры до текущего
            angle_from_ref_to_curr, current_scale = estimate_affine_params(ref_kp, curr_kp, matches_ref)

        # === Решение: обновлять опорный кадр? ===
        should_update_ref = False
        if tracking_ok:
            if current_scale < SCALE_MIN or current_scale > SCALE_MAX:
                should_update_ref = True

        # === Обновление опорного кадра и поправки ===
        if should_update_ref:
            # Вычисляем угол между СТАРЫМ опорным и НОВЫМ (текущим)
            # Он уже в `angle_from_ref_to_curr`
            global_angle_offset += angle_from_ref_to_curr
            global_angle_offset %= 360

            # Обновляем опору
            ref_gray, ref_kp, ref_desc = curr_gray, curr_kp, curr_desc
            H_cumulative = H_total
            print(f"🔄 Обновление опоры | scale={current_scale:.2f}, offset={global_angle_offset:.1f}°")

        # === Глобальный угол от ПЕРВОГО кадра до ТЕКУЩЕГО ===
        global_angle = (global_angle_offset + angle_from_ref_to_curr) % 360

        # === FPS ===
        frame_count += 1
        if time.perf_counter() - start_time >= 1.0:
            fps = frame_count
            frame_count = 0
            start_time = time.perf_counter()

        # === Визуализация ===
        display = curr_gray.copy()
        if center:
            cv2.circle(display, center, 2, (0, 255, 0), 2)
            dx = center[0] - W / 2
            dy = H / 2 - center[1]
            text = f"dx:{dx:+.1f}, dy:{dy:+.1f}, angle:{global_angle:.1f} | FPS: {fps}"
            cv2.putText(display, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(text)
        else:
            print("❌ Трекинг потерян")

        cv2.imshow("Tracking", display)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()