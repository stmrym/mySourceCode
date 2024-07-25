import numpy as np
import matplotlib.pyplot as plt

n = int(1e5)

sigma_list = np.random.normal(0.1,0.05,size=n)
sigma_list = [s if s > 0 else 0 for s in sigma_list]

print(len(sigma_list)/n)
norm1 = np.random.normal(0, 0.1, size=n)
norm2 = np.random.normal(0, 0.2, size=n)

x = np.random.normal(0, sigma_list, size=len(sigma_list))

print(min(sigma_list), max(sigma_list))
print(min(x), max(x))

print()
fig, ax = plt.subplots()
# ax.hist(norm2, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0, 0.2)')
# ax.hist(norm1, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0, 0.1)')
# ax.set_ylim([0,4000])
ax.hist(sigma_list, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0.1, 0.05)')
print(x)
ax.hist(x, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0, N(0.1, 0.05))')
ax.legend()

fig.savefig('normal.png')