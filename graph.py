import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

H, W = (480, 640)
k = 0.1
sampling = 1

fig, ax = plt.subplots(dpi=300)
x_, y_ = np.arange(-W//8,W//8 + sampling, sampling), np.arange(-H//8, H//8 + sampling, sampling)
x, y = np.meshgrid(x_, y_)

z = np.sqrt((x/(k*W))**2 + (y/(k*H))**2)
z = np.minimum(z, 1)

im = ax.pcolormesh(x,y,z,cmap='plasma', shading='auto')
# plt.colorbar()

# im = ax.imshow(plot_data, cmap=cmap)
# ax.axis("off")
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
ax.set(aspect=1)
fig.colorbar(im, cax=cax)
plt.tight_layout()
fig.patch.set_alpha(0)
plt.savefig('pcolormesh.png')

