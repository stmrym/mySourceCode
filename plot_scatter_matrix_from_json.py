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
    fig.write_html('scatter_matrix.html')


def plot_sns(df):

    corr = df.drop(columns=['Seq']).corr()
    plt.rcParams['axes.labelsize'] = 24
    sns.pairplot(df, hue='Seq')
    plt.savefig('scatter_matrix.png', bbox_inches='tight', pad_inches=0.04, dpi=200)

    plt.close()

    sns.heatmap(corr, cbar = True, square = True, vmin = -1.0, vmax =  1.0, center = 0, annot = True, annot_kws={ 'size':15 },
	fmt='.2f', xticklabels = corr.columns.values, yticklabels = corr.columns.values,)
    plt.savefig('scatter_corr_matrix.png', bbox_inches='tight', pad_inches=0.04, dpi=300)


if __name__ == '__main__':

    json_path = '../STDAN_modified/exp_log/test/2024-12-10T130843__ESTDAN_VT3_e300_/epoch-0300_BSD_3ms24ms.json'

    with open(json_path, 'r') as file:
        data = json.load(file)
    

    flattened_data = make_flatten_data(data)
    flattened_df = pd.DataFrame(flattened_data)


    new_order = ['PSNR', 'SSIM', '-LPIPS', '-NIQE', 'LR']
    flattened_df = flattened_df[new_order]
    flattened_df = add_seq_column(flattened_df)

    print(flattened_df)

    # plot_plotly(flattened_df)
    plot_sns(flattened_df)


