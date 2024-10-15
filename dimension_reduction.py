import random
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from matplotlib.figure import Figure as Mplfig
from plotly.graph_objs._figure import Figure as Plotlyfig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.graph_objects as go
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from stop_watch import stop_watch

import base64
from io import BytesIO as _BytesIO
import time

DATASET_LABEL = ['GOPRO', 'BSD']

def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """
    t_start = time.time()

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    t_end = time.time()
    if verbose:
        print(f"PIL converted to b64 in {t_end - t_start:.3f} sec")

    return encoded

class DimReduction():
    def __init__(self, dir_path_l, n_sample = 100):
        self.dir_path_l = dir_path_l
        self.n_sample = n_sample

        self.path_l = []
        for dir_path in dir_path_l:
            paths = sorted([p for p in Path(dir_path).glob('**/*.png') if re.search('blur_gamma|Blur', str(p))])
            self.path_l += paths

        self.path_l = random.sample(self.path_l, n_sample)

        self.image_l = []
        self.label_l = []
        for p in tqdm(self.path_l):
            self.image_l.append(self._crop_center(Image.open(p)))
            self.label_l.append(self._find_matching_label(str(p), DATASET_LABEL))

        '''
        self.path_l: [Path(img1), Path(img2), ...]
        self.image_l: [(H, W, C), (H, W, C), ...]
        self.label_l: ['BSD', 'GOPRO', ...]
        '''
            
    def _find_matching_label(self, path, label_l):
        for label in label_l:
            if label in path:
                return label
        print(f'does not match {path} in {label_l}')
        exit()

    def _crop_center(self, pil_img, crop_size=480):
        w, h = pil_img.size
        return pil_img.crop(((w - crop_size) // 2, (h - crop_size) // 2,
                            (w + crop_size) // 2, (h + crop_size) // 2))

    @stop_watch
    def fit(self, opt):
        # (N, H, W, C)
        self.images_np = np.stack(self.image_l, axis=0)
        # images_np: (N, CHW)
        self.images_np = self.images_np.reshape(self.images_np.shape[0], -1)
        print(self.images_np.shape)

        method = opt.pop('method')  
        if method == 'tsne':
            self.tsne = TSNE(n_components=opt['n_components'], perplexity=opt['perplexity'], random_state=opt['random_state'])
            self.reduced = self.tsne.fit_transform(self.images_np)
        elif method == 'pca':
            pass

        print(self.reduced.shape)
        # (N, d)
        return self.reduced



class Graph():
    def __init__(self, opt):   
        self.graph_opt = opt['graph']
        self.figsize = self.graph_opt.pop('figsize', (8,8))
        self.dpi = self.graph_opt.pop('dpi', 100)
        if self.graph_opt != 'plotly':
            if opt['n_components'] == 2:
                self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            elif opt['n_components'] == 3:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')

    def _proj(self, X, ax1, ax2):
        #3D points in ax1 was project to 2D plot, calculate position of image of ax2
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax1.get_proj())
        return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))

    def _image(self, ax,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        im = OffsetImage(arr, zoom=0.05)
        im.image.axes = ax
        ab = AnnotationBbox(im, xy, xybox=(0,0),
                            xycoords='data', boxcoords="offset points",
                            pad=0)
        ax.add_artist(ab)

    def _imscatter(self, reduced, image_l, color_l, zoom=0.1):
        # images_np: [T, H, W, C]
        im_list = [OffsetImage(ImageOps.expand(image, border=20, fill=color), zoom=zoom) for image, color in zip(image_l, color_l)]
        x, y = np.atleast_1d(reduced[:,0], reduced[:,1])
        artists = []

        if reduced.shape[1] == 2:
            for x0, y0, im in zip(x, y, im_list):
                ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
                # artists.append(self.ax.add_artist(ab))
                self.ax.add_artist(ab)
            self.ax.update_datalim(np.column_stack([x, y]))
            self.ax.set_xlabel('PC1')
            self.ax.set_ylabel('PC2')
            self.ax.autoscale()

        elif reduced.shape[1] == 3:
            # Create a second axes to place annotations
            z = np.atleast_1d(reduced[:,2])
            
            self.ax.scatter(x, y, z, marker="o")
            self.ax2 = self.fig.add_subplot(111,frame_on=False) 
            self.ax2.axis("off")
            self.ax2.axis([0,1,0,1])

            for x0, y0, z0, im in zip(x, y, z, im_list):
                X, Y = self._proj([x0, y0, z0], self.ax, self.ax2)
                im.image.axes = self.ax2
                ab = AnnotationBbox(im, [X, Y], xybox=(0,0),xycoords='data', 
                                    boxcoords="offset points",pad=0)
                self.ax2.add_artist(ab)
            
            self.ax.set_xlabel('PC1')
            self.ax.set_ylabel('PC2')            
            self.ax.set_zlabel('PC3')
            # self.ax.autoscale()             
        return artists
    
    def _scatter(self, reduced, color_l):
        if reduced.shape[1] == 2:
            print('plt.scatter(2D) mode')
            self.ax.scatter(reduced[:,0], reduced[:,1], color=color_l, linewidths=1)
            self.ax.set_xlabel('PC1')
            self.ax.set_ylabel('PC2')
            self.ax.autoscale()

        elif reduced.shape[1] == 3:
            print('plt.scatter(3D) mode')
            self.ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], color=color_l, linewidths=1)
            self.ax.set_xlabel('PC1')
            self.ax.set_ylabel('PC2')
            self.ax.set_zlabel('PC3')
            self.ax.autoscale()
            

    def _goscatter(self, reduced, color_l, path_l):
        if reduced.shape[1] == 2:
            print('go.Scatter2d mode')
            self.fig = go.Figure(data=[go.Scatter(x=reduced[:,0], y=reduced[:,1], mode='markers', marker=dict(
                    size=5, color=color_l
                ))])
        elif reduced.shape[1] == 3:
            print('go.Scatter3d mode')
            self.fig = go.Figure(data=[go.Scatter3d(x=reduced[:,0], y=reduced[:,1], z=reduced[:,2], mode='markers', marker=dict(
                size=5, color=color_l
                ))])
            
        if self.graph_opt['add_image']:
            for x, y, path in zip(reduced[:,0], reduced[:,1], path_l):
                print(x, y, str(path))
                self.fig.add_layout_image(dict(source=pil_to_b64(Image.open(path)), xref='x', yref='y', x=x, y=y, sizex=0.5,sizey=0.5))

    def plot(self, reduced, image_l, label_l, path_l):
        '''
        graph_opt: dict
        reduced: (N, d)
        image_l: [(H, W, C), (H, W, C), ...]
        label_l: ['BSD', 'GOPRO', ...]
        '''
        plot_mode = self.graph_opt.pop('plot_mode')
        color_l = [self.graph_opt['labelcolor'][label] for label in label_l]

        if plot_mode == 'point':
            if self.graph_opt['add_image']:
                self._imscatter(reduced, image_l, color_l, self.graph_opt.pop('zoom', None))
            else:
                self._scatter(reduced, color_l)
        elif plot_mode == 'plotly':
            self._goscatter(reduced, color_l, path_l)

    def save(self):
        if isinstance(self.fig, Mplfig):
            self.fig.savefig(self.graph_opt['savename'] + '.png', bbox_inches='tight', pad_inches=0, dpi=self.dpi)
            print(f'{self.graph_opt["savename"]}.png saved')
        elif isinstance(self.fig, Plotlyfig):
            self.fig.write_html(self.graph_opt['savename'] + '.html')
            self.fig.write_image(self.graph_opt['savename'] + '.png')
            print(f'{self.graph_opt["savename"]}.html saved')


if __name__ == '__main__':

    opt = {
        'dir_path_l':       [
                            '../dataset/BSD_3ms24ms/test',
                            '../dataset/GOPRO_Large/test'
                            ],
        'seed':             0,
        'n_sample':         50,
        'method':           'tsne',
        'n_components':     2,
        'random_state':     20,
        'perplexity':       30,
        'graph':
        {
            'savename':     'tsne_debug',
            'plot_mode':    'plotly', # 'point' 'imscatter' 'plotly'
            'add_image':    True,
            'zoom':         0.05,
            'figsize':      (8, 8),
            'dpi':          200,
            'labelcolor': 
            {
                'BSD' :     'magenta',
                'GOPRO' :   'cyan'
            }
        }
    }

    random.seed(opt['seed'])
    dr = DimReduction(opt['dir_path_l'], n_sample = opt['n_sample'])
    # _ = dr.fit(opt)

    dr.reduced = np.array([[i]*opt['n_components'] for i in range(0,opt['n_sample'])])
    print(dr.reduced.shape)

    graph = Graph(opt)
    graph.plot(dr.reduced, dr.image_l, dr.label_l, dr.path_l)
    graph.save()


