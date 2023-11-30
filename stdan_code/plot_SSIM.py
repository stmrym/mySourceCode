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

        
# vdtr = dataFrame(exp_path = os.path.join('..', '..', 'VDTR_result', 'csv_SSIM'))
stdan = dataFrame(exp_path = os.path.join('..', '..', 'STDAN_modified', 'exp_log', 'test', '20231129_STDAN_Stack_BSD_3ms24ms_ckpt-epoch-0905', 'SSIM_csv'))

savepath = os.path.join('..', '..', 'STDAN_modified', 'exp_log', 'test', '20231129_STDAN_Stack_BSD_3ms24ms_ckpt-epoch-0905', 'SSIM_graph')

if not os.path.isdir(savepath):
    os.makedirs(savepath, exist_ok=True)


for id in range(stdan.num_csv):

    frame = stdan.get_frame(id)
    ### plot SSIM graph ###


    fig = plt.figure(figsize=(7,3), dpi=300)
    ax = fig.add_subplot()

    ax.set_xlim([0,149])
    #ax.set_ylim([0.945,0.99])
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_xticks(np.arange(0,150, step=10))


    # vdtr.plot(id=id, frame=frame, col_name='VDTR', color='black', label='VDTR')
    # vdtr.plot(id=id, frame=frame, col_name='C_T10K', color='tab:blue', label='VDTR + Temopral (Proposed)')
    stdan.plot(id=id, frame=frame, col_name='output', color='tab:red', label='STDAN')


    #ax.legend(loc='lower center', ncol=4) 
    #ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3)
    ax.legend() 
    plt.tight_layout()

    
    seq = stdan.get_seq(id)
    plt.savefig(os.path.join(savepath, seq + '.png'), transparent=False, dpi=300, bbox_inches='tight')
