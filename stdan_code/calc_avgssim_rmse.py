import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

'''
Calculate the average SSIM and RMSE for n frames

[Input] SSIM .csv files of each video (./SSIM_csv/xxx.csv)

[Output] Average SSIM .csv file and RMSE .csv file (average_SSIM.csv, RMSE.csv)

'''
# # video '000' to '019'
# exp_name = 'STDAN_Stack_BSD_3ms24ms_best-ckpt'
# exp_path = os.path.join(os.environ['HOME'], 'STDAN', 'exp_log', 'test', exp_name)

exp_name = '20231124_STDAN_Stack_BSD_3ms24ms_ckpt-epoch-0455'
exp_path = os.path.join('..', '..', 'STDAN_modified', 'exp_log', 'test', exp_name)

csv_list = sorted(glob.glob(os.path.join(exp_path, 'SSIM_csv', '*.csv')))
seq_list = [os.path.splitext(os.path.basename(f))[0] for f in csv_list]

avg_df = pd.DataFrame({
    'seq': seq_list,
    'avgSSIM': np.zeros(len(seq_list)),
    'RMSE': np.zeros(len(seq_list))
})

for row_id, csv_file in enumerate(csv_list):

    df = pd.read_csv(csv_file)

    # calculate the average SSIM
    ssim_mean_series = df.loc[: , 'output'].mean()  # calculate average SSIM from 'output' 
    avg_df.loc[row_id, 'avgSSIM'] = ssim_mean_series 

    # calculate RMSE
    rmse = np.sqrt(np.mean((df['output'] - df['output_LPF'])**2))
    avg_df.loc[row_id, 'RMSE'] = rmse

    
    
print(avg_df)

# calculate average from seqs
avg_df.loc['Average', 'avgSSIM'] = avg_df['avgSSIM'].mean()
avg_df.loc['Average', 'RMSE'] = avg_df['RMSE'].mean()

print(avg_df)

avg_df.to_csv(os.path.join(exp_path, 'average.csv'), index=False) # save to .csv