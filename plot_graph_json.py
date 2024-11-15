import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import yaml
from pathlib import Path
'''
plot SSIM graph

[Input] SSIM .csv files of each video (./SSIM_csv/xxx.csv)

[Output] SSIM graph of each video (./SSIM_graph_woSIRR/xxx.png)

'''

class dataPlot():
    def __init__(self, json_path, **kwargs):
        self.json_dict = self._read_json(json_path)
        self.kwargs = kwargs

        self.frame_l = [int(f_name.split('.')[0]) for f_name in self.json_dict[metric][seq].keys()]
        self.value_l = self.json_dict[metric][seq].values()
        print(f'Read {json_path}')



    def _read_json(self, json_path):
        with open(json_path, mode='r') as f:
            json_dict = yaml.safe_load(f)    
        return json_dict

    def plot(self, ax):
        ax.plot(self.frame_l, self.value_l, **self.kwargs)


seq = '103'
metric = 'BRISQUE'  # PSNR, SSIM, BRISQUE


if __name__ == '__main__':

    # 2024-11-12T073940__CT_
    # 2024-11-12T080924__
    # 2024-11-14T040540__STDAN_

    # epoch-0400_Mi11Lite_output.json
    # epoch-0400_BSD_1ms8ms_output.json
    # epoch-0400_BSD_1ms8ms_comp_output.json
    
    save_name = seq + '_' + metric + '_comp_graph.png'
    font_size = 12
    dpi = 200
    print(seq, metric)

    plot_list = [
        dataPlot(json_path = '../STDAN_modified/exp_log/test/2024-11-14T040540__STDAN_/epoch-0400_BSD_1ms8ms_comp_output.json',
            ls='-', lw=1.0, c='black', label = 'STDAN',
        ),

        dataPlot(json_path = '../STDAN_modified/exp_log/test/2024-11-12T080924__/epoch-0400_BSD_1ms8ms_comp_output.json',
            ls='-', lw=1.0, c='tab:blue', label = 'ESTDAN',
        ),

        dataPlot(json_path = '../STDAN_modified/exp_log/test/2024-11-12T073940__CT_/epoch-0400_BSD_1ms8ms_comp_output.json',
            ls='-', lw=1.0, c='tab:red', label = 'R-ESTDAN',
        )
    ]

  
    ### plot graph ###
    fig, ax = plt.subplots(figsize=(11,3))

    

    frame_min = plot_list[0].frame_l[0]
    frame_max = plot_list[0].frame_l[-1]
    ax.set_xlim([frame_min, frame_max])
    ax.set_ylim([40, 90])
    ax.set_xlabel('frame', fontsize=font_size)
    ax.set_ylabel(metric, fontsize=font_size)


    for plot_obj in plot_list:
        plot_obj.plot(ax)

    # ax.legend(loc='lower left', ncol=4) 
    #ax.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3)
    ax.minorticks_on()
    ax.grid(which = "both", axis="x", alpha=0.5)
    ax.grid(axis='y')
    ax.legend() 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(f'Seq.{seq}')
    plt.tight_layout()
    plt.savefig(save_name, transparent=False, dpi=dpi, bbox_inches='tight')
    plt.close()
