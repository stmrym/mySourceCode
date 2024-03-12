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
            # if n % 2 == 0:
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            frame = frame[180:900, 320:1600, :]
            cv2.imwrite(f'{dir_path}/{str(n).zfill(5)}.{ext}', frame)
            n += 1
        else:
            return

if __name__ == '__main__':
    save_all_frames('/mnt/d/Chronos/vid_2025-03-06_22-25-03.mp4', '/mnt/d/Chronos/004')
