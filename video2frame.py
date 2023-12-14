import cv2
import os

def save_all_frames(video_path, dir_path, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)

    n = 0

    while True:
        ret, frame = cap.read() 
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ret:
            if n % 2 == 0:
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                cv2.imwrite(f'{dir_path}/{str(n//2).zfill(3)}.{ext}', frame)
            n += 1
        else:
            return

save_all_frames('../dataset/raw_video/3.mp4', '../dataset/raw_video/3_15fps_resized')
