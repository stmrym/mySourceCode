import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

'''
plot SSIM graph

[Input] SSIM .csv files of each video (./SSIM_csv/xxx.csv)

[Output] SSIM graph of each video (./SSIM_graph_woSIRR/xxx.png)

'''
class dataPlot():
    def __init__(self, exp_path, c=None, ls='-', lw=1.5, alpha=1, label='sample'):
        self.exp_path = exp_path
        self.color = c
        self.linestyle = ls
        self.linewidth = lw
        self.alpha = alpha
        self.label = label

    def plot(self, seq):
        self.df = pd.read_csv(os.path.join(self.exp_path, seq + '.csv'))
        ax.plot(self.df['frame'], self.df[metric_type], c=self.color, ls=self.linestyle, lw=self.linewidth, alpha=self.alpha, label=self.label)

plot_list = [
    dataPlot(
        exp_path = '../../STDAN_modified/exp_log/train/Sobel_not_fixed_2024-05-03T125515_ESTDAN_light_Stack_GOPRO/visualization/epoch-0400_output/metrics_csv',
        c = 'tab:blue',
        lw = 1.5,
        label = 'Variable Sobel weights'
        ),
    dataPlot(
        exp_path = '../../STDAN_modified/exp_log/train/Sobel_fixed_2024-04-23T042249_ESTDAN_light_Stack_GOPRO/visualization/epoch-0400_output/metrics_csv',
        c = 'tab:red',
        lw = 1.5,
        label='Fixed Sobel weights'
        )
]
metric_type = 'SSIM'
savepath = '../../STDAN_modified/debug_results'      

csv_path_list = sorted(glob.glob(os.path.join(plot_list[0].exp_path, '*.csv')))
seq_list = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_path_list]

for seq in tqdm(seq_list):
    ### plot graph ###
    fig = plt.figure(figsize=(11,4), dpi=300)
    ax = fig.add_subplot()

    # ax.set_xlim([0,149])
    #ax.set_ylim([0.945,0.99])
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel(metric_type, fontsize=12)

    for plot_obj in plot_list:
        plot_obj.plot(seq)

    # ax.legend(loc='lower left', ncol=4) 
    #ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3)
    ax.minorticks_on()
    ax.grid(which = "both", axis="x")
    ax.grid(axis='y')
    ax.legend() 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, seq + '.png'), transparent=False, dpi=300, bbox_inches='tight')
    plt.close()
