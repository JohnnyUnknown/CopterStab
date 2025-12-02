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
camera.configure(video_config)
camera.start()


def detect_and_compute(frame, method, name_method="SIFT"):
    if name_method == 'SIFT':
        keypoints, descriptors = method.detectAndCompute(frame, None)
    elif name_method == 'ORB':
        keypoints, descriptors = method.detectAndCompute(frame, None)
    else:
        raise ValueError("Метод должен быть 'SIFT' или 'ORB'")
    return keypoints, descriptors


def match_descriptors(desc1, desc2, name_method='SIFT'):
    if desc1 is None or desc2 is None:
        return []
    if name_method == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            if len(good_matches) >= 40:
                return good_matches
    return good_matches

# Поиск списка координат общих КТ на кадре после pixel_mask
def location_images_2(good_matches, kp):
    matches = []
    for i in range(len(good_matches)):
        large_image_KP = list(kp[good_matches[i].trainIdx].pt)
        large_image_KP[0] = int(large_image_KP[0])
        large_image_KP[1] = int(large_image_KP[1])
        matches.append(large_image_KP)
    return matches


def compute_angle(kp1, kp2, matches):
    """
    Оценивает угол поворота между двумя кадрами.
    
    Возвращает:
        angle_deg — угол поворота в диапазоне [0, 360) градусов.
    """
    if len(matches) < 4:
        # print("Недостаточно совпадений (< 4)")
        return None

    # Координаты точек
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Оценка частичного аффинного преобразования
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    if M is None:
        # print("Не удалось оценить преобразование")
        return None

    # Извлечение угла поворота
    a, c = M[0, 0], M[1, 0]
    angle_rad = np.arctan2(c, a)          # диапазон: [-π, π]
    angle_deg = np.degrees(angle_rad)     # диапазон: [-180, 180]

    # Нормализация к [0, 360)
    angle_deg = angle_deg % 360
    if angle_deg < 0:
        angle_deg += 360  # хотя % 360 уже даёт [0, 360), это страховка

    return angle_deg

# Поиск списка координат общих КТ на главном изображении
def find_area(good_matches, kp1):
    matches = []
    for i in range(len(good_matches)):
        dmatch = good_matches[i]
        # Поиск найденных КТ для обеих изображений в списке КТ главного изображения
        large_image_KP = list(kp1[dmatch.queryIdx].pt)
        large_image_KP[0] = int(large_image_KP[0])
        large_image_KP[1] = int(large_image_KP[1])
        # Добавление в список КТ главного изображения, совпадающих с КТ искомого
        matches.append(large_image_KP)
    return matches

# Вычисление матрицы преобразования координат
def transformation_matrix(main_matches, matches_2):
    # Массивы с точками соответствия
    pts1 = np.float32([m for m in matches_2]).reshape(-1, 1, 2)
    pts2 = np.float32([m for m in main_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H

# Определение положения на опорном кадре с помощью матрицы преобразования
def true_center(img, frame_matches, matches):
    crop_center = np.array([[img.shape[1] / 2, img.shape[0] / 2]], dtype='float32').reshape(-1, 1, 2)
    H = transformation_matrix(frame_matches, matches)
    try:
        find_center = cv2.perspectiveTransform(crop_center, H)
        true_center = []
        true_center.append(int(find_center[0][0][0]))
        true_center.append(int(find_center[0][0][1]))
        return true_center
    except cv2.error:
        print("Ошибка матрицы гомографии.\n")
        return None


def print_points(img, matches, center):
    for point in matches:
        img = cv2.circle(img, point, radius=1,color=(0,255,0),thickness=1)
    return cv2.circle(img, center, radius=2,color=(0,255,0),thickness=2)
    

def main():
    name_method = "SIFT"
    method = None
    if name_method == 'SIFT':
        method = cv2.SIFT_create(nOctaveLayers=3)
    elif name_method == 'ORB':
        method = cv2.ORB_create(nfeatures=200)

    cnt, FPS, angle = 0, 0, 0

    prev_frame = camera.capture_array()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    #prev_frame = prev_frame[:img_size, :img_size]
    prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)
    kp1, desc1 = detect_and_compute(prev_frame, method, name_method)

    start = time.perf_counter()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        cnt += 1
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = frame[:img_size, :img_size]
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if len(kp1) < 20:
            #print(f"{len(kp1)=}")
            prev_frame = frame
            kp1, desc1 = detect_and_compute(prev_frame, method, name_method)
            continue
            
        kp2, desc2 = detect_and_compute(frame, method, name_method)
        if len(kp2) < 10: 
            #print(f"{len(kp2)=}")
            continue

        matches = match_descriptors(desc1, desc2, name_method)


        if matches and len(matches) >= 6:
            angle = compute_angle(kp1, kp2, matches)
            main_matches = find_area(matches, kp1)
            matches = location_images_2(matches, kp2)
            center = true_center(prev_frame, matches, main_matches)
            if center:
                display_frame = frame.copy()
                display_frame = print_points(display_frame, matches, center)
                #display_frame = cv2.circle(display_frame, center, radius=2,color=(0,255,0),thickness=2)

                x, y = center
                dx = round(x - img_size / 2, 1)
                dy = round(img_size / 2 - y, 1)
                text = f"{dx:.1f}, {dy:.1f}, {angle:.1f}"
                print(text, f"{len(matches)=}, {FPS=}")

                finish = time.perf_counter()
                seconds = (finish - start) % 60
                if 1 <= seconds:
                    FPS = cnt 
                    cnt = 0
                    start = finish

                cv2.putText(display_frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Video with shift", display_frame)
            else:
                print("Центр не найден!!!")
        else:
            text = "Нет общих точек."
            #print(f"{len(matches)=}")
            #prev_frame = frame

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
