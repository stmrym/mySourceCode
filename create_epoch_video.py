import cv2
import numpy as np
from pathlib import Path
from put_text_in_image import put_text_in_image
from PIL import Image





def search_path_list(base_path, pattern):
    base_path = Path(base_path)
    seq_l = []
    path_l = sorted(base_path.glob(str(pattern)))
    
    assert path_l != []

    for match in path_l:
        parts = match.relative_to(base_path).parts
        seq_part = parts[0]
        seq_l.append(seq_part)

    return path_l, seq_l




def create_gif(path_l, seq_l, save_name, duration=100):

    frame_l = []
    for seq, path in zip(seq_l, path_l):
        frame = cv2.imread(str(path))
        frame = put_text_in_image(frame, seq, place='top-left', size=0.9, color='green')
        frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))        
        frame_l.append(frame_rgb)

    gif_save_name = str(save_name) + '.gif'
    frame_l[0].save(gif_save_name, save_all=True, append_iamges=frame_l[1:], loop=0, duration=duration)
    print(f'{gif_save_name} wrote.')



def create_mp4(path_l, seq_l, save_name, fps=1):
    
    first_frame = cv2.imread(str(path_l[0]))
    H, W, _ = first_frame.shape 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_save_name = str(save_name) + '.mp4'
    video_writer = cv2.VideoWriter(video_save_name, fourcc, fps, (W, H))


    for seq, path in zip(seq_l, path_l):
        frame = cv2.imread(str(path))
        frame = put_text_in_image(frame, seq, place='top-left', size=0.9, color='red')
        video_writer.write(frame)
    print(f'{video_save_name} wrote.')
    video_writer.release()




if __name__ == '__main__':

    # dataset = 'BSD_2ms16ms_comp'
    dataset = 'Mi11Lite'
    # seq = '121'
    seq = 'VID_20240523_165150'
    # frame = '00000006.png'
    frame = '00045.png'
    # exp = '2024-11-21T192622_C_ESTDAN_v3_BSD_3ms24ms_GOPRO'
    # exp = '2024-11-18T170527_CT_ESTDAN_v3_BSD_3ms24ms_GOPRO'
    exp = '2024-11-26T024206_C_ESTDAN_v3_BSD_3ms24ms_GOPRO'

    input_path = Path('../dataset') / dataset / 'test' / seq / frame
    video_base_path = Path('../STDAN_modified/exp_log/train/') / exp / 'visualization'
    pattern = Path('epoch-*_'+ dataset + '_output') / seq / frame

    output_dir = ''


    path_l, seq_l = search_path_list(video_base_path, pattern)
    
    del_index_l = []
    for i in range(len(path_l)):
        epoch = seq_l[i].split('epoch-')[-1].split('_')[0]
        if int(epoch) % 100 != 0:
            del_index_l.append(i)
    
    del_index_l.sort(reverse=True)
    for index in del_index_l:
        del path_l[index], seq_l[index]

    path_l.insert(0, input_path)
    seq_l.insert(0, 'Input')

    for path, seq in zip(path_l, seq_l):
        print(path, seq)

    save_name = Path(output_dir) / '-'.join(Path(str(pattern).replace('epoch-*_', '')).with_suffix('').parts + (video_base_path.parts[-2], ) )


    create_mp4(path_l, seq_l, save_name, fps=1)
    # create_gif(path_l, seq_l, save_name, duration=100)



