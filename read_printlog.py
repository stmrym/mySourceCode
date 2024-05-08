import pandas as pd

path = '../STDAN_modified/exp_log/train/F_2024-04-24T095752_STDAN_Stack_GOPRO/print.log'
save_path = '../STDAN_modified/exp_log/train/F_2024-04-24T095752_STDAN_Stack_GOPRO/print.csv'

cols = ['chronos', 'GOPRO']
df = pd.DataFrame()

with open(path) as f:
    eval_lines = [line.rstrip() for line in f.readlines() if 'INFO:[EVAL]' in line]
    for eval_line in eval_lines:
        if 'Epoch' in eval_line:
            epoch = eval_line.split('[Epoch ')[1].split('/')[0]
            for col_dataset_name in cols:
                if col_dataset_name in eval_line:
                    mid_metric = eval_line.split('mid:')[1].split(', out:')[0]
                    out_metric = eval_line.split('out:')[1].split('), ')[0]

                    df.at[epoch, col_dataset_name + '_mid'] = mid_metric
                    df.at[epoch, col_dataset_name + '_out'] = out_metric

print(df)
df.to_csv(save_path)