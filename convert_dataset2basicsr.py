import glob
import os
import shutil

src = '../dataset/BSD_3ms24ms_original/test/*'
dst = '../dataset/BSD_3ms24ms/test_basic'

src_seq_list = sorted(glob.glob(src))

for src_seq in src_seq_list:
    seq = os.path.basename(src_seq)
    
    copy_src = os.path.join(src_seq, 'Blur', 'RGB')
    copy_dst = os.path.join(dst, 'blur', seq)
    shutil.copytree(copy_src, copy_dst)

    print(f'{copy_src} -> {copy_dst}')

    copy_src = os.path.join(src_seq, 'Sharp', 'RGB')
    copy_dst = os.path.join(dst, 'GT', seq)
    shutil.copytree(copy_src, copy_dst)

    print(f'{copy_src} -> {copy_dst}')

