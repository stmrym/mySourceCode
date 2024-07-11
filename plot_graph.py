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
    def __init__(self, exp_path, c=None, ls='-', lw=1.5, alpha=1, metric_type='SSIM', label='sample', plot_func='plot'):
        self.exp_path = exp_path
        self.color = c
        self.linestyle = ls
        self.linewidth = lw
        self.alpha = alpha
        self.label = label
        self.metric_type = metric_type
        self.plot_func = plot_func

    def plot(self, seq):
        self.df = pd.read_csv(os.path.join(self.exp_path, seq + '.csv'))
        if self.plot_func == 'plot':
            ax.plot(self.df['frame'], self.df[self.metric_type], c=self.color, ls=self.linestyle, lw=self.linewidth, alpha=self.alpha, label=self.label)
        elif self.plot_func == 'stackplot':
            ax.stackplot(self.df['frame'], self.df['masked_SSIM'], self.df['i_masked_SSIM'], ls=self.linestyle, lw=self.linewidth, colors=['orange', 'palegreen'])


plot_list = [

    dataPlot(
        exp_path = '../STDAN_modified/exp_log/train/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
        c = 'gray',
        ls = ':',
        lw = 1.0,
        metric_type = 'SSIM',
        label = 'SSIM',
        plot_func = 'stackplot'
        ),
    dataPlot(
        exp_path = '../STDAN_modified/exp_log/train/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
        c = 'gray',
        ls = ':',
        lw = 1.0,
        metric_type = 'SSIM',
        label = 'STDAN $S$',
        plot_func = 'plot'
        ),
    dataPlot(
        exp_path = '../STDAN_modified/exp_log/train/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
        c = 'tab:red',
        ls = ':',
        lw = 1.0,
        metric_type = 'masked_SSIM',
        label='STDAN $S_m$',
        plot_func = 'plot'
        ),

    dataPlot(
        exp_path = '../STDAN_modified/exp_log/train/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
        c = 'black',
        ls = '-',
        lw = 1.0,
        metric_type = 'SSIM',
        label = 'ESTDANv2 $S$',
        plot_func = 'plot'
        ),
    dataPlot(
        exp_path = '../STDAN_modified/exp_log/train/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
        c = 'red',
        ls = '-',
        lw = 1.0,
        metric_type = 'masked_SSIM',
        label='ESTDANv2 $S_m$',
        plot_func = 'plot'
        )
]

# plot_list = [
#     dataPlot(
#         exp_path = '../STDAN_modified/exp_log/train/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
#         c = 'black',
#         ls = '-',
#         lw = 1.0,
#         metric_type = 'i_masked_SSIM',
#         label = 'STDAN'
#         ),

#     dataPlot(
#         exp_path = '../STDAN_modified/exp_log/train/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim/metrics_csv',
#         c = 'tab:red',
#         ls = '-',
#         lw = 1.5,
#         metric_type = 'i_masked_SSIM',
#         label = 'ESTDAN'
#         )
# ]

metric = 'SSIM'
savepath = '../STDAN_modified/debug_results/STDAN_ESTDAN2'      

csv_path_list = sorted(glob.glob(os.path.join(plot_list[0].exp_path, '*.csv')))
seq_list = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_path_list]

for seq in tqdm(seq_list):
    ### plot graph ###
    fig = plt.figure(figsize=(11,4), dpi=300)
    ax = fig.add_subplot()

    # ax.set_xlim([0,149])
    #ax.set_ylim([0.945,0.99])
    ax.set_xlabel('frame', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

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
