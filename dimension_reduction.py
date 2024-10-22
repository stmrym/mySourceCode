import yaml
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from matplotlib.figure import Figure as Mplfig
from plotly.graph_objs._figure import Figure as Plotlyfig
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.graph_objects as go
# from MulticoreTSNE import MulticoreTSNE as TSNE
from stop_watch import stop_watch


class ImageData():
    def __init__(self, path, opt):
        ''''
        self.opt: {'key': value, ...}
        self.path: Path('../.../basename.ext')
        self.image: Image(path)
        self.label: str('dataset_name')
        self.ssim: float(ssim_value)
        '''
        self.opt = opt
        self.path = path
        self._crop_size = self.opt.get('crop_size', 480)
        self.image = self._load_data(path)
        assert self.opt['label_type'] in ['dataset', 'ssim']
        if self.opt['label_type'] == 'dataset':
            self.label = self._find_matching_label(str(path), opt['dir_path_dict'].keys())
        elif self.opt['label_type'] == 'ssim':
            self.label = self._find_ssim_from_csv(opt['ssim_csv_path'])

    def _load_data(self, path):
        if path.suffix == '.npy':
            image = self._transform(Image.fromarray(np.load(path)))
        elif path.suffix == '.png':
            image = self._transform(Image.open(path))
        return image
        

    def _transform(self, pil_img):
        if self.opt['input_type'] == 'grayscale':
            pil_img = pil_img.convert('L')
        return self._crop_center(pil_img, self._crop_size)
    
    def _crop_center(self, pil_img, crop_size):
        w, h = pil_img.size
        return pil_img.crop(((w - crop_size) // 2, (h - crop_size) // 2,
                            (w + crop_size) // 2, (h + crop_size) // 2))
    
    def _find_matching_label(self, path, label_l):
        for label in label_l:
            if label in path:
                return label
        print(f'does not match {path} in {label_l}')
        exit()

    def _find_ssim_from_csv(self, csv_path):
        for dataset_prefix in self.opt['dir_path_dict'].values():
            try:
                # p: Path('seq/.../basename.ext')
                p = self.path.relative_to(Path(dataset_prefix))
                seq = p.parts[0]
                frame = p.stem
                break
            except ValueError:
                continue

        df = pd.read_csv((Path(csv_path) / seq).with_suffix('.csv'))
        ssim = df.loc[df['frame'] == int(frame), 'SSIM'].values[0]
        return ssim


class DimReduction():
    def __init__(self, opt):
        self.opt = opt

        self.path_l = []
        for dir_path in self.opt['dir_path_dict'].values():
            seq_dir_path_l = sorted([dir for dir in Path(dir_path).iterdir() if dir.is_dir()])
            for seq_dir in seq_dir_path_l:
                # paths = sorted([p for p in Path(seq_dir).glob('**/*.png') if re.search('blur_gamma|Blur', str(p))])
                paths = sorted([p for p in Path(seq_dir).rglob('*') if (re.search('blur_gamma|Blur', str(p))) or p.suffix == '.npy'])
                self.path_l += paths[1:-1]

        self.n_sample = self.opt.pop('n_sample', None)
        if not self.n_sample == None:
            self.path_l = random.sample(self.path_l, self.n_sample)
        else:
            self.path_l = self.path_l[0:100] # 0 300 3000 3200

        self.imagedata_l = []
        for p in tqdm(self.path_l):
            self.imagedata_l.append(ImageData(p, opt))

        '''
        self.path_l: [Path(img1), Path(img2), ...]
        self.image_l: [(H, W, C), (H, W, C), ...]
        self.label_l: ['BSD', 'GOPRO', ...] or [SSIM1, SSIM2, ...]
        '''
            

    @stop_watch
    def fit(self):
        # (N, H, W, C)
        images_np = np.stack([imagedata.image for imagedata in self.imagedata_l], axis=0)
        # images_np: (N, CHW)
        images_np = images_np.reshape(images_np.shape[0], -1)
        print(images_np.shape)

        method = self.opt['method']  
        if method['name'] == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=method['n_components'], perplexity=method['perplexity'], random_state=method['random_state'])
            self.reduced = tsne.fit_transform(images_np)
        elif method['name'] == 'pca':
            pass

        print(self.reduced.shape)
        # (N, d)
        return self.reduced



class Graph():
    def __init__(self, opt):
        self.graph_opt = opt['graph']
        self.dim = opt['method']['n_components']
        self.label_type = opt['label_type']
        self.figsize = self.graph_opt.pop('figsize', (8, 8))
        self.dpi = self.graph_opt.pop('dpi', 100)
        if self.graph_opt != 'plotly':
            if self.dim == 2:
                self.fig, self.ax = plt.subplots()
            elif self.dim == 3:
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

    def _normalize(self, ndarray):
        return (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())

    def _float_rgb_to_int(self, float_rgb):
        int_rgb = tuple([int(rgb*255) for rgb in float_rgb])
        return int_rgb

    def _set_xyzlabel(self):
        self.ax.set_xlabel('PC1')
        self.ax.set_ylabel('PC2')    
        if self.dim == 3:
            self.ax.set_zlabel('PC3')        

    def _imscatter(self, reduced, imagedata_l, color_l, cmap):
        # images_np: [T, H, W, C]
        if self.label_type == 'dataset':
            rgb_color_l = color_l
        elif self.label_type == 'ssim':    
            rgb_color_l = [self._float_rgb_to_int(cmap(normalized)) for normalized in self._normalize(np.array(color_l))]
        im_list = [OffsetImage(ImageOps.expand(imagedata.image.convert('RGB'), border=20, fill=color), zoom=self.graph_opt['zoom']) for imagedata, color in zip(imagedata_l, rgb_color_l)]
        
        x, y = np.atleast_1d(reduced[:,0], reduced[:,1])
        z = np.atleast_1d(reduced[:,2]) if self.dim == 3 else None
        self._set_xyzlabel()

        if self.dim == 2:            
            scat = self.ax.scatter(x, y, s=self.graph_opt['s'], alpha=self.graph_opt['alpha'], c=color_l, cmap=cmap, marker="o")
            for x0, y0, im in zip(x, y, im_list):
                ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
                self.ax.add_artist(ab)
            self.ax.update_datalim(np.column_stack([x, y]))
            self.fig.colorbar(scat, ax=self.ax) 

        elif self.dim == 3:
            # Create a second axes to place annotations
            scat = self.ax.scatter(x, y, z, s=self.graph_opt['s'], alpha=self.graph_opt['alpha'], c=color_l, cmap=cmap, marker="o")
            self.ax2 = self.fig.add_subplot(111,frame_on=False) 
            self.ax2.axis("off")
            self.ax2.axis([0,1,0,1])

            for x0, y0, z0, im in zip(x, y, z, im_list):
                X, Y = self._proj([x0, y0, z0], self.ax, self.ax2)
                im.image.axes = self.ax2
                ab = AnnotationBbox(im, [X, Y], xybox=(0,0),xycoords='data', 
                                    boxcoords="offset points",pad=0)
                self.ax2.add_artist(ab)  
    

    def _scatter(self, reduced, color_l, cmap):
        
        print('plt.scatter mode')
        x, y = np.atleast_1d(reduced[:,0], reduced[:,1])
        z = np.atleast_1d(reduced[:,2]) if self.dim == 3 else None
        pad = 0.11 if reduced.shape[1] == 3 else 0.04

        scatter_params = {k: self.graph_opt[k] for k in ['s', 'alpha', 'marker'] if k in self.graph_opt}

        if self.dim == 2:
            scat = self.ax.scatter(x, y, c=color_l, cmap=cmap, **scatter_params)
        elif self.dim == 3:
            scat = self.ax.scatter(x, y, z, c=color_l, cmap=cmap, **scatter_params)

        self.fig.colorbar(scat, ax=self.ax, pad=pad)  
        self._set_xyzlabel()


    def _goscatter(self, reduced, color_l, cmap):

        print('go.Scatter mode')
        x, y = np.atleast_1d(reduced[:,0], reduced[:,1])
        z = np.atleast_1d(reduced[:,2]) if self.dim == 3 else None

        if self.dim == 2:
            self.fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers', marker=dict(size=5, color=color_l, colorscale=cmap))])
        elif self.dim == 3:
            self.fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color=color_l, colorscale=cmap))])

    def plot(self, reduced, imagedata_l):
        '''
        graph_opt: dict
        reduced: (N, d)
        imagedata_l: [Imagedata(path1), Imagedata(path2), ...]
        '''
        plot_mode = self.graph_opt.pop('plot_mode')

        assert self.label_type in ['dataset', 'ssim']
        if self.label_type == 'dataset':
            # ['magenta', 'cyan', ...]
            color_l = [self.graph_opt['label_color'][imagedata.label] for imagedata in imagedata_l]
            cmap = None
        elif self.label_type == 'ssim':
            # [ssim1, ssim2, ...]
            color_l = [imagedata.label for imagedata in imagedata_l]
            cmap = plt.get_cmap(self.graph_opt['ssim_color'])

        if plot_mode == 'plt':
            if self.graph_opt['add_image']:
                self._imscatter(reduced, imagedata_l, color_l, cmap)
            else:
                self._scatter(reduced, color_l, cmap)
        elif plot_mode == 'plotly':
            self._goscatter(reduced, color_l, self.graph_opt['ssim_color'])

    def save(self):
        if isinstance(self.fig, Mplfig):
            self.fig.savefig(self.graph_opt['savename'] + '.png', bbox_inches='tight', pad_inches=0.1, dpi=self.dpi)
            print(f'{self.graph_opt["savename"]}.png saved')
        elif isinstance(self.fig, Plotlyfig):
            self.fig.write_html(self.graph_opt['savename'] + '.html')
            self.fig.write_image(self.graph_opt['savename'] + '.png')
            print(f'{self.graph_opt["savename"]}.html saved')


if __name__ == '__main__':

    with open('dimension_reduction.yaml', 'r') as yml:
        opt = yaml.safe_load(yml)

    random.seed(opt['seed'])
    dr = DimReduction(opt)
    _ = dr.fit()

    # dr.reduced = np.array([[i]*opt['n_components'] for i in range(0,opt['n_sample'])])
    # print(dr.reduced.shape)

    graph = Graph(opt)
    graph.plot(dr.reduced, dr.imagedata_l)
    graph.save()


