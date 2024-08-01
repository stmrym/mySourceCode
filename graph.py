import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

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

