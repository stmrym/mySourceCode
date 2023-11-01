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
csv_list = sorted(glob.glob(os.path.join('SSIM_csv', '*.csv')))
edvr_cdv_list = sorted(glob.glob(os.path.join('edvr_SSIM_csv', '*.csv')))
for csv, edvr_csv in zip(csv_list, edvr_cdv_list):

    df = pd.read_csv(csv)
    edvr_df = pd.read_csv(edvr_csv)

    frame = df['frame']
    output = df['output']
    edvr = edvr_df['EDVR']
    edvr = edvr[2:-2]
    wopd_tsa = edvr_df['woPD_woTSA']
    wopd_tsa = wopd_tsa[2:-2]

    output_LPF = df['output_LPF']
    edvr_LPF = edvr_df['EDVR_LPF']
    edvr_LPF = edvr_LPF[2:-2]
    wopd_tsa_LPF = edvr_df['woPD_woTSA_LPF']
    wopd_tsa_LPF = wopd_tsa_LPF[2:-2]


    ### plot SSIM graph ###


    fig = plt.figure(figsize=(7,4), dpi=300)
    ax = fig.add_subplot()

    ax.set_xlim([0,99])
    #ax.set_ylim([0.945,0.99])
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_xticks(np.arange(0,100, step=10))

    
    #ax.plot(frame, sirr_LPF, c='black', ls='-', lw=1, alpha=0.5) 
    ax.plot(frame, edvr_LPF, c='tab:blue', ls='-', lw=0.5, alpha=0.4)
    ax.plot(frame, edvr, c='tab:blue', ls='-', lw=1.5, alpha=1, label='EDVR')

    ax.plot(frame, wopd_tsa_LPF, c='tab:green', ls='-', lw=0.5, alpha=0.4)
    ax.plot(frame, wopd_tsa, c='tab:green', ls='-', lw=1.5, alpha=1, label='woPD_TSA')

    ax.plot(frame, output_LPF, c='tab:red', ls='-', lw=0.5, alpha=0.4)
    ax.plot(frame, output, c='tab:red', ls='-', lw=1.5, alpha=1, label='Zhang et al.')



    #ax.legend(loc='lower center', ncol=4) 
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3)
    plt.tight_layout()

    savepath = 'SSIM_graph'
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    seq = os.path.splitext(os.path.basename(csv))[0]
    plt.savefig(os.path.join(savepath, seq + '.png'), transparent=False, dpi=300, bbox_inches='tight')
