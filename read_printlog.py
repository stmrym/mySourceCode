import pandas as pd

path = '../STDAN_modified/exp_log/test/20240508_W_augmentation/print.log'
save_path = '../STDAN_modified/exp_log/test/20240508_W_augmentation/print.csv'

cols = ['chronos', 'chronos_raw']
df = pd.DataFrame()

with open(path) as f:
    eval_lines = [line.rstrip() for line in f.readlines() if 'INFO:[EVAL]' in line]
    for eval_line in eval_lines:
        if 'Epoch' in eval_line:
            epoch = eval_line.split('[Epoch ')[1].split('/')[0]
            for col_dataset_name in cols:
                if col_dataset_name in eval_line:
                    
                    if 'PSNR' in eval_line:
                        mid_metric = eval_line.split('mid:')[1].split(', out:')[0]
                        out_metric = eval_line.split('out:')[1].split('), ')[0]
                        df.at[epoch, col_dataset_name + '_PSNR_mid'] = mid_metric
                        df.at[epoch, col_dataset_name + '_PSNR_out'] = out_metric
                    
                    elif 'SSIM' in eval_line:
                        mid_metric = eval_line.split('mid:')[1].split(', out:')[0]
                        out_metric = eval_line.split('out:')[1].split('))')[0]
                        df.at[epoch, col_dataset_name + '_SSIM_mid'] = mid_metric
                        df.at[epoch, col_dataset_name + '_SSIM_out'] = out_metric

print(df)
df.to_csv(save_path)