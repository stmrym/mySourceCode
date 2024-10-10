from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from MulticoreTSNE import MulticoreTSNE as TSNE



if __name__ == '__main__':
    # fig, ax = plt.subplots(figsize=(4,4))

    fig, ax = plt.subplots(figsize=(8, 8))