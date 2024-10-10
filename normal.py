import numpy as np
import matplotlib.pyplot as plt

# n = int(1e5)

# sigma_list = np.random.normal(0.1,0.05,size=n)
# sigma_list = [s if s > 0 else 0 for s in sigma_list]

# print(len(sigma_list)/n)
# norm1 = np.random.normal(0, 0.1, size=n)
# norm2 = np.random.normal(0, 0.2, size=n)

# x = np.random.normal(0, sigma_list, size=len(sigma_list))

# print(min(sigma_list), max(sigma_list))
# print(min(x), max(x))

# print()


b = np.linspace(-5, 5, 100)

y = -3
l = 2
# f = (1 - b)**2 + 0.1*b**2
f = (y - b)**2 + l*np.abs(b)

sol = y + (l/2)







fig, ax = plt.subplots(figsize=(4,4), dpi=300)
# ax.hist(norm2, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0, 0.2)')
# ax.hist(norm1, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0, 0.1)')
# ax.set_ylim([0,4000])
# ax.hist(sigma_list, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0.1, 0.05)')

ax.plot(b, f)
ax.vlines(sol, -2, 60, 'tab:red', linewidth= 1, linestyles='dashed')
ax.set_xlabel(r'$\beta_1$')
ax.set_ylabel(r'$f(\beta_1)$')
# ax.set_xlim([-2, 4])
# ax.set_ylim([-1, 5])
ax.grid()
# print(x)
# ax.hist(x, bins='auto', alpha=0.5, histtype='stepfilled', label='N(0, N(0.1, 0.05))')
# ax.legend()

fig.savefig('fig3.png')