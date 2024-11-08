import cv2
import glob
import os
import yaml










if __name__ == '__main__':
    
    file_tmpl = '../dataset/BSD_3ms24ms/test/%s/Blur/RGB'
    base_dir = file_tmpl.split('%s')[0]

    with open('create_val_transformed_dataset.yml', mode='r') as f:
        opt = yaml.safe_load(f)

    seqs = sorted([f for f in os.listdir(base_dir) if os.path.join(base_dir, f)])
    for seq in seqs:


        print(seqs)
        exit()
        file_paths = sorted(glob.glob(os.path.join(base_dir, '*/*.png'), recursive=True))
        save_dir = '../dataset/BSD_3ms24ms_comp/test'



