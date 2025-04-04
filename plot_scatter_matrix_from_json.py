from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def make_flatten_data(data_dict):

    flattened_data = OrderedDict()
    for metric, seqs_dict in data_dict.items():
        if metric in ['NIQE', 'LPIPS']:
            metric = '-' + metric 
        flattened_data[metric] = OrderedDict()
        for seq, values_dict in seqs_dict.items():
            if seq in ['Average', 'TotalAverage']:
                continue
            for frame, value in values_dict.items():
                new_key = f'{seq}_{frame}'
                flattened_data[metric][new_key] = value if metric in ['PSNR', 'SSIM', 'LR'] else -value
    return flattened_data


def add_seq_column(df):
    df['Seq'] = df.index.str.split('_').str[0]
    return df



def plot_plotly(df):
    df['Index'] = df.index
    fig = px.scatter_matrix(df, dimensions=['PSNR', 'SSIM', '-LPIPS', '-NIQE', 'LR'], color='Seq', hover_data={'Index': True})
    
    fig.update_layout(title='Scatter Matrix with sequences')
    # fig.show()
    fig.write_html('scatter_matrix_1200.html')


def plot_sns(df):

    # 1. plot scatter matrix
    corr = df.drop(columns=['Seq']).corr()
    plt.rcParams['axes.labelsize'] = 24
    g = sns.pairplot(df, hue='Seq')
    
    # FR-IQA to blue, NR-IQA to red
    fr_iqa_l = ['PSNR', 'SSIM', '-LPIPS']
    nr_iqa_l = ['-NIQE', 'LR']

    # Coloring on axis name
    for ax in g.axes.flatten():
        if ax.get_xlabel() in fr_iqa_l:
            ax.set_xlabel(ax.get_xlabel(), color='tab:blue')
        elif ax.get_xlabel() in nr_iqa_l:
            ax.set_xlabel(ax.get_xlabel(), color='tab:red')
        
        if ax.get_ylabel() in fr_iqa_l:
            ax.set_ylabel(ax.get_ylabel(), color='tab:blue')
        elif ax.get_ylabel() in nr_iqa_l:
            ax.set_ylabel(ax.get_ylabel(), color='tab:red')
    
    plt.savefig('scatter_matrix_1200.png', bbox_inches='tight', pad_inches=0.04, dpi=200)
    plt.close()


    # 2. plot scatter corr matrix
    g = sns.heatmap(corr, cbar = True, square = True, vmin = -1.0, vmax =  1.0, center = 0, annot = True, annot_kws={ 'size':15 },
	fmt='.2f', xticklabels = corr.columns.values, yticklabels = corr.columns.values,)
    
    # Coloring on axis name
    x_label_l = g.get_xticklabels()
    y_label_l = g.get_yticklabels()

    for x_label in x_label_l:
        if x_label.get_text() in fr_iqa_l:
            x_label.set_color('tab:blue')
        elif x_label.get_text() in nr_iqa_l:
            x_label.set_color('tab:red')
        
    for y_label in y_label_l:
        if y_label.get_text() in fr_iqa_l:
            y_label.set_color('tab:blue')
        elif y_label.get_text() in nr_iqa_l:
            y_label.set_color('tab:red')
    
    plt.savefig('scatter_corr_matrix_1200.png', bbox_inches='tight', pad_inches=0.04, dpi=300)



if __name__ == '__main__':

    # Load from json file
    json_path = '../STDAN_modified/exp_log/train/2024-12-16T121335_VT3_ESTDAN_v3_BSD_3ms24ms/epoch-1200_BSD_3ms24ms.json'
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Make dataframe
    flattened_data = make_flatten_data(data)
    flattened_df = pd.DataFrame(flattened_data)

    # Change axis order
    new_order = ['PSNR', 'SSIM', '-LPIPS', '-NIQE', 'LR']
    flattened_df = flattened_df[new_order]
    flattened_df = add_seq_column(flattened_df)

    print(flattened_df)

    # Use 'plot_plotly' or 'plot_sns'
    
    # plot_plotly(flattened_df)
    plot_sns(flattened_df)


