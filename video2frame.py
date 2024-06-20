import cv2
import glob
import os
from tqdm import tqdm

def save_frames(video_path: str, dir_path: str, ext: str = 'png'):
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
            # frame = frame[180:900, 320:1600, :]
            cv2.imwrite(f'{dir_path}/{str(n).zfill(5)}.{ext}', frame)
            n += 1
        else:
            return


def save_all_videos(video_dir: str, save_dataset_dir:str):

    seqs = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    for seq in tqdm(seqs):
        seq_name = os.path.splitext(os.path.basename(seq))[0]
        seq_dir = os.path.join(save_dataset_dir, seq_name)

        if not os.path.isdir(seq_dir):
            os.makedirs(seq_dir, exist_ok=True)
        
        save_frames(
            video_path = seq,
            dir_path = seq_dir,
            ext = 'png')


if __name__ == '__main__':
    
    save_all_videos(
        video_dir = '/mnt/d/Chronos_HS/test', 
        save_dataset_dir = '/mnt/d/Chronos_HS/test'
        )
