import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

'''
plot SSIM graph

[Input] SSIM .csv files of each video (./SSIM_csv/xxx.csv)

[Output] SSIM graph of each video (./SSIM_graph_woSIRR/xxx.png)

'''

class dataFrame():
    def __init__(self, exp_path=''):
        self.exp_path = exp_path
        self.csv_list = sorted(glob.glob(os.path.join(self.exp_path, '*.csv')))
        self.num_csv = len(self.csv_list)

    def get_seq(self, id):
        self.seq = os.path.splitext(os.path.basename(self.csv_list[id]))[0]
        return self.seq
    
    def get_frame(self, id):
        self.frame = pd.read_csv(self.csv_list[id])['frame']
        return self.frame
    
    def plot(self, id=0, frame=0, col_name='', color='tab:red', label='sample'):
        self.df = pd.read_csv(self.csv_list[id])
        ax.plot(frame, self.df[col_name +  '_LPF'], c=color, ls='-', lw=0.5, alpha=0.4)
        ax.plot(frame, self.df[col_name], c=color, ls='-', lw=1.5, alpha=1, label=label)

        
# wo_edge = dataFrame(exp_path = '../../STDAN_modified/exp_log/train/WO_Motion_small_2024-02-08T161225_STDAN_Stack_BSD_3ms24ms_GOPRO/visualization/epoch-0900_output/SSIM_csv')
# w_edge = dataFrame(exp_path = '../../STDAN_modified/exp_log/train/F_2024-03-12T112710_ESTDAN_light_Stack_BSD_3ms24ms_GOPRO/visualization/epoch-0900_output/SSIM_csv')

wo_edge = dataFrame(exp_path = '../../STDAN_modified/exp_log/test/WO_Motion_small_2024-02-08T161225_STDAN_Stack_BSD_3ms24ms_GOPRO_ckpt-epoch-0900/SSIM_csv')
w_edge = dataFrame(exp_path = '../../STDAN_modified/exp_log/test/F_2024-03-12T112710_ESTDAN_light_Stack_BSD_3ms24ms_GOPRO_ckpt-epoch-0900/SSIM_csv')

savepath = '../../STDAN_modified/debug_results/20240322'


if not os.path.isdir(savepath):
    os.makedirs(savepath, exist_ok=True)


for id in range(wo_edge.num_csv):

    frame = wo_edge.get_frame(id)
    ### plot SSIM graph ###


    fig = plt.figure(figsize=(11,4), dpi=300)
    ax = fig.add_subplot()

    # ax.set_xlim([0,149])
    #ax.set_ylim([0.945,0.99])
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)

    wo_edge.plot(id=id, frame=frame, col_name='output', color='tab:blue', label='STDANet')
    w_edge.plot(id=id, frame=frame, col_name='output', color='tab:red', label='ESTDANet')


    # ax.legend(loc='lower left', ncol=4) 
    #ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3)
    ax.minorticks_on()
    ax.grid(which = "both", axis="x")
    ax.grid(axis='y')
    ax.legend() 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    
    seq = wo_edge.get_seq(id)
    plt.savefig(os.path.join(savepath, seq + '.png'), transparent=False, dpi=300, bbox_inches='tight')
    plt.close()
