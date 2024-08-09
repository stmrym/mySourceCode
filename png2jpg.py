from PIL import Image

src = '/mnt/d/results/SIP_202408/result_fig/00000020_b.png'
dst = '/mnt/d/results/SIP_202408/result_fig/00000020_b.jpg'

im = Image.open(src)
im = im.convert("RGB")
im.save(dst)