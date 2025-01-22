import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

def plot_pcolormesh():
    H, W = (480, 640)
    k = 0.1
    sampling = 1
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(dpi=100)
    M = np.minimum(H, W)

    x_, y_ = np.arange(-M//8,M//8 + sampling, sampling), np.arange(-M//8, M//8 + sampling, sampling)
    x, y = np.meshgrid(x_, y_)

    z = np.sqrt(x**2 + y**2)/(k*M)
    z = np.minimum(z, 1)

    im = ax.pcolormesh(x,y,z,cmap='plasma', shading='auto')
    # plt.colorbar()

    # im = ax.imshow(plot_data, cmap=cmap)
    # ax.axis("off")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('bottom', '5%', pad='10%')
    ax.set(aspect=1)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.tight_layout()
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig('pcolormesh_small.png', bbox_inches='tight', pad_inches=0.04)


class dataPlot():
    def __init__(self, csv_path, csv_avg_path, label, **kwargs):
        self.csv_path = csv_path
        self.csv_avg_path = csv_avg_path
        self.label = label
        self.kwargs = kwargs
        self.x_data = np.empty(0)
        self.y_data = np.empty(0)

    def read_csv_append_list(self):
        csv_list = sorted(glob.glob(self.csv_path))
        for path in csv_list:
            print(path)
            df = pd.read_csv(path)
            self.x_data = np.concatenate([self.x_data, df['masked_SSIM'].values])
            self.y_data = np.concatenate([self.y_data, df['i_masked_SSIM'].values])
        
        print(self.x_data, self.y_data)
        print(len(self.x_data), len(self.y_data))

    def read_avg_csv_append_list(self):
        csv = self.csv_avg_path
        df = pd.read_csv(csv, index_col=0)
        
        # self.x_data = np. lf.y_data, df['avgi_masked_SSIM'].values])

        if 'BSD' in self.label:
            self.x_data = np.concatenate([self.x_data, df['000':'131']['avgmasked_SSIM'].values])
            self.y_data = np.concatenate([self.y_data, df['000':'131']['avgi_masked_SSIM'].values])

        elif 'GoPro' in self.label:
            self.x_data = np.concatenate([self.x_data, df['GOPR0384_11_00':'GOPR0881_11_01']['avgmasked_SSIM'].values])
            self.y_data = np.concatenate([self.y_data, df['GOPR0384_11_00':'GOPR0881_11_01']['avgi_masked_SSIM'].values])


    def scatter_plot(self, ax):
        # self.x_data = np.tan(self.x_data * np.pi / 2)
        # self.y_data = np.tan(self.y_data * np.pi / 2)
        ax.scatter(self.x_data, self.y_data, label=self.label, **self.kwargs)

    def ecdf_plot(self, ax):

        ssim_data = np.sort(self.x_data + self.y_data)
        sorted_x_data = np.sort(self.x_data)
        sorted_y_data = np.sort(self.y_data)

        n = len(self.x_data)
        cdf = np.arange(1, n+1)/n

        # ax.plot(ssim_data, cdf, label=self.label, **self.kwargs)
        # ax.plot(sorted_x_data, cdf, label=self.label, linestyle='-', **self.kwargs)
        ax.plot(sorted_y_data, cdf, label=self.label, linestyle = '-', **self.kwargs)
        # ax.plot(ssim_data, cdf, label=self.label, linestyle = '-', **self.kwargs)



def plot_si_vs_sm():

    class_list = [

        # for CDF
<<<<<<< HEAD
        # dataPlot(
        #         csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
        #         csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
        #         label = 'STDANet',
        #         c = 'blue',
        #         # marker = '.',
        #         # s = 5,
        #         # linewidths = 1,
        #         # facecolor = 'None',
        #         # edgecolors = 'blue'
        #         ),


        # dataPlot(
        #         csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
        #         csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
        #         label = r'Proposed 1 (ESTDANet)',
        #         c = 'red',
        #         # marker = '.',
        #         # s = 5,
        #         # linewidths = 1,
        #         # facecolor = 'None',
        #         # edgecolors = 'red'
        #         ),  
=======
        dataPlot(
                csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
                csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
                label = 'STDAN',
                c = 'blue',
                # marker = '.',
                # s = 5,
                # linewidths = 1,
                # facecolor = 'None',
                # edgecolors = 'blue'
                ),


        dataPlot(
                csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
                csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
                label = r'ESTDAN + $\mathcal{L}_\mathrm{f}$ (Ours)',
                c = 'red',
                # marker = '.',
                # s = 5,
                # linewidths = 1,
                # facecolor = 'None',
                # edgecolors = 'red'
                ),  
>>>>>>> 8780e6bad2834e7fa17e0e3d5d8f3ebe9684fd9a

        # for scatterplot
        # dataPlot(
        #         csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
        #         csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
        #         label = 'STDANet',
        #         c = 'blue',
        #         marker = '.',
        #         s = 5,
        #         linewidths = 1,
        #         # facecolor = 'None',
        #         # edgecolors = 'blue'
        #         ),


        # dataPlot(
        #         csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
        #         csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
        #         label = 'Proposed 1 (ESTDANet)',
        #         c = 'red',
        #         marker = '.',
        #         s = 5,
        #         linewidths = 1,
        #         # facecolor = 'None',
        #         # edgecolors = 'red'
        #         ),  

        dataPlot(
                csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
                csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
                label = 'STDANet (BSD)',
                # c = 'blue',
                marker = '.',
                s = 100,
                linewidths = 1,
                facecolor = 'None',
                edgecolors = 'blue'
                ),

        dataPlot(
                csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/GOPR*.csv',
                csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-05-20T193725_STDAN_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
                label = 'STDANet (GoPro)',
                # c = 'blue',
                marker = '^',
                s = 30,
                linewidths = 1,
                facecolor = 'None',
                edgecolors = 'blue'
                ),

        dataPlot(
                csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/*.csv',
                csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
                label = 'Proposed 1 (BSD)',
                # c = 'red',
                marker = '.',
                s = 100,
                linewidths = 1,
                facecolor = 'None',
                edgecolors = 'red'
                ),  
                
        dataPlot(
                csv_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/metrics_csv/GOPR*.csv',
                csv_avg_path = '../STDAN_modified/STDANet_train_ckpt/2024-06-10T115520_F_ESTDAN_v3_BSD_3ms24ms_GOPRO/visualization/c1200_out_maskedssim_cir/avg_metrics.csv',
                label = 'Proposed 1 (GoPro)',
                # c = 'red',
                marker = '^',
                s = 30,
                linewidths = 1,
                facecolor = 'None',
                edgecolors = 'red'
                ),   
    ]


    plt.rcParams["font.size"] = 13
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    
    fig, ax = plt.subplots(dpi=200, tight_layout=True)

    ax.axline((0,1), (1,0), color='black', lw=0.8)
    for data_inst in class_list:
        # data_inst.read_csv_append_list()
        data_inst.read_avg_csv_append_list()
        data_inst.scatter_plot(ax)
        # data_inst.ecdf_plot(ax)

    ax.set_xlabel(r'Motion-Weighted $\mathrm{SSIM_m}$', fontsize=16)
    ax.set_ylabel(r'Inverse Motion-Weighted $\mathrm{SSIM_i}$', fontsize=16)

    # ax.set_xlabel(r'Motion-Weighted $\mathrm{SSIM_m}$', fontsize=16)
    # ax.set_ylabel(r'Inverse Motion-Weighted $\mathrm{SSIM_i}$', fontsize=16)


    ax.set_xticks(np.linspace(0, 1.0, 11), minor=True)
    ax.set_yticks(np.linspace(0, 1.0, 11), minor=True)
    
    # ax.set_xlim([0,1.0])
    # ax.set_ylim([0,1.0])
    # ax.set_xlabel(r'$\mathrm{SSIM_i}$', fontsize=16)
    # ax.set_ylabel(r'Empirical Distribution Function', fontsize=16)

    # ax.legend(markerscale = 4)
    ax.legend(markerscale = 1)
    ax.set_aspect('equal')
    # ax.grid(alpha=0.4)
    plt.tight_layout()
    # fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig('ecdf.png', bbox_inches='tight', pad_inches=0.04)



if __name__ == '__main__':
    # plot_pcolormesh()
    plot_si_vs_sm()