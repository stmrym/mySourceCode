import numpy as np
import matplotlib.pyplot as plt
import glob

file_paths = glob.glob('../BSSTNet/blur_map/BSD_3ms24ms/**/*.npy', recursive=True)
arrays = [np.load(file).flatten() for file in file_paths]
combined_array = np.concatenate(arrays)
print(combined_array.shape)


file_paths = glob.glob('../BSSTNet/blur_map/GOPRO_Large/**/*.npy', recursive=True)
arrays = [np.load(file).flatten() for file in file_paths]
combined_array2 = np.concatenate(arrays)
print(combined_array2.shape)

combined_array = np.append(combined_array, combined_array2)
print(combined_array.shape)

print(combined_array.max(), combined_array.min())
# ヒストグラムの作成
plt.hist(combined_array, bins=50, alpha=0.75)
plt.title('Histogram of Combined Arrays')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('unnorm_3.png')