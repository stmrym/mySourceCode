import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
plot SSIM graph

[Input] SSIM .csv files of each video (./SSIM_csv/xxx.csv)

[Output] SSIM graph of each video (./SSIM_graph_woSIRR/xxx.png)

'''

exp_name = 'STDAN_Stack_BSD_3ms24ms_best-ckpt'
exp_path = os.path.join(os.environ['HOME'], 'STDAN', 'exp_log', 'test', exp_name)

csv_list = sorted(glob.glob(os.path.join(exp_path, 'SSIM_csv', '*.csv')))
for csv in csv_list:

    df = pd.read_csv(csv)

    frame = df['frame']
    output = df['output']

    output_LPF = df['output_LPF']


    ### plot SSIM graph ###


    fig = plt.figure(figsize=(7,4), dpi=300)
    ax = fig.add_subplot()

    ax.set_xlim([0,99])
    #ax.set_ylim([0.945,0.99])
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_xticks(np.arange(0,100, step=10))


    ax.plot(frame, output_LPF, c='tab:red', ls='-', lw=0.5, alpha=0.4)
    ax.plot(frame, output, c='tab:red', ls='-', lw=1.5, alpha=1, label='STDAN')



    #ax.legend(loc='lower center', ncol=4) 
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3)
    plt.tight_layout()

    savepath = os.path.join(exp_path, 'SSIM_graph')
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    seq = os.path.splitext(os.path.basename(csv))[0]
    plt.savefig(os.path.join(savepath, seq + '.png'), transparent=False, dpi=300, bbox_inches='tight')
