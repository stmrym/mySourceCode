import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import yaml
from pathlib import Path
'''
plot SSIM graph

[Input] SSIM .csv files of each video (./SSIM_csv/xxx.csv)

[Output] SSIM graph of each video (./SSIM_graph_woSIRR/xxx.png)

'''

class dataPlot():
    def __init__(self, json_path, seq, metric, **kwargs):
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




if __name__ == '__main__':

    # 2024-11-12T073940__CT_
    # 2024-11-12T080924__
    # 2024-11-14T040540__STDAN_

    # epoch-0400_Mi11Lite_output.json
    # epoch-0400_BSD_1ms8ms_output.json
    # epoch-0400_BSD_1ms8ms_comp_output.json

    seq_l = [
        # 'VID_20240523_162610',
        # 'VID_20240523_163120',
        # 'VID_20240523_163623',
        # 'VID_20240523_163625',
        # 'VID_20240523_163720',
        # 'VID_20240523_164307',
        # 'VID_20240523_164327',
        # 'VID_20240523_164408',
        # 'VID_20240523_164506',
        # 'VID_20240523_164507',
        # 'VID_20240523_164838',
        # 'VID_20240523_165048',
        # 'VID_20240523_165103',
        # 'VID_20240523_165150'
    ]

    seq_l = [
        '000',
        '002',
        '003',
        '004',
        '006',
        '010',
        '014',
        '049',
        '055',
        '074',
        '082',
        '087',
        '088',
        '091',
        '094',
        '098',
        '111',
        '113',
        '128',
        '131'
    ]

    metric = 'SSIM'  # PSNR, SSIM, BRISQUE

    for seq in seq_l:
        save_name = seq + '_' + metric + '_graph.png'
        font_size = 12
        dpi = 200
        print(seq, metric)

        plot_list = [

            # dataPlot(json_path = '../STDAN_modified/exp_log/train/ToYNU/2024-12-16T180921__STDAN_Mi11Lite/epoch-0850_Mi11Lite.json',
            #     seq=seq, metric=metric, ls='-', lw=1.0, c='black', label = 'STDAN epoch-0850',
            # ),

            # dataPlot(json_path = '../STDAN_modified/exp_log/train/ToYNU/2024-12-17T143130__ESTDAN_v3_Mi11Lite/epoch-0850_Mi11Lite.json',
            #     seq=seq, metric=metric, ls='-', lw=1.0, c='tab:blue', label = 'ESTDAN epoch-0850',
            # ),

            # # dataPlot(json_path = '../STDAN_modified/exp_log/train/ToYNU/2024-12-19T122927_VT2_ESTDAN_v3_Mi11Lite/epoch-0700_Mi11Lite.json',
            # #     seq=seq, metric=metric, ls='-', lw=1.0, c='tab:orange', label = 'ESTDAN+VT1 epoch-0700',
            # # ),

            # dataPlot(json_path = '../STDAN_modified/exp_log/train/ToYNU/2024-12-11T210210__VT3__ESTDAN_v3_BSD_3ms24ms_GOPRO/epoch-0650_Mi11Lite.json',
            #     seq=seq, metric=metric, ls='-', lw=1.0, c='tab:orange', label = 'ESTDAN+VT2 epoch-0650',
            # ),

            # dataPlot(json_path = '../STDAN_modified/exp_log/train/ToYNU/2024-12-20T090716__VT3_CM__ESTDAN_v3_BSD_3ms24ms_GOPRO/epoch-1200_Mi11Lite.json',
            #     seq=seq, metric=metric, ls='-', lw=1.0, c='tab:red', label = 'ESTDAN+VT2+CM epoch-1200',
            # ),


            dataPlot(json_path = '../STDAN_modified/exp_log/test/2025-02-01T103900_STDAN_BSD_3ms24ms_GOPRO/epoch-1200_BSD_3ms24ms.json',
                seq=seq, metric=metric, ls='-', lw=1.0, c='#4E91F0', label = 'STDANet',
            ),

            dataPlot(json_path = '../STDAN_modified/exp_log/test/2025-02-04T051928_ESTDAN_BSD_3ms24ms_GOPRO/epoch-1200_BSD_3ms24ms.json',
                seq=seq, metric=metric, ls='-', lw=1.0, c='#FF965C', label = 'Proposed 1',
            ),

            dataPlot(json_path = '../STDAN_modified/exp_log/test/2025-02-05T054151_STDAN_VT2_CM_BSD_3ms24ms_GOPRO/epoch-1200_BSD_3ms24ms.json',
                seq=seq, metric=metric, ls='-', lw=1.0, c='#0C4493', label = 'STDANet+CM+VT2',
            ),

            dataPlot(json_path = '../STDAN_modified/exp_log/test/2025-01-28T065524_ESTDAN_VT2_CM_BSD_3ms24ms_GOPRO/epoch-1200_BSD_3ms24ms.json',
                seq=seq, metric=metric, ls='-', lw=1.0, c='#EE5500', label = 'Proposed 2',
            )
        ]


        ### plot graph ###
        fig, ax = plt.subplots(figsize=(11,3))

        

        frame_min = plot_list[0].frame_l[0]
        frame_max = plot_list[0].frame_l[-1]
        ax.set_xlim([frame_min, frame_max])
        # ax.set_ylim([40, 90])
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
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.title(f'Seq.{seq}')
        plt.tight_layout()
        plt.savefig(save_name, transparent=False, dpi=dpi, bbox_inches='tight')
        plt.close()
