import matplotlib.pyplot as plt
from matplotlib import patches, patheffects


def show_img_with_boxes(img, gt_anno=None, pred_anno=None, ax=None, figsize=(12, 12), title=""):

    ax = show_img(img, ax=ax, figsize=figsize, title=title)

    if gt_anno is not None:
        for cat, bbox in gt_anno:
            draw_rect(ax, bbox, color='red')
            draw_text(ax, bbox[:2], cat, sz=16)

    if pred_anno is not None:
        for cat, bbox in pred_anno:
            draw_rect(ax, bbox, color='blue')
            draw_text(ax, bbox[:2], cat, sz=16)

    return ax


def show_img(im, figsize=None, ax=None, title=None, return_fig=False):
    dim = len(im.shape)
    assert (dim == 2 or dim ==
            3), "Image has to be represented by a 2D or 3D Numpy array"

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    if dim == 2:
        ax.imshow(im, cmap='gray')
    else:
        ax.imshow(im)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=18)

    if return_fig:
        return fig, ax

    return ax


def draw_rect(ax, b, color='red'):
    patch = ax.add_patch(patches.Rectangle(
        b[:2], *b[-2:], fill=False, alpha=0.5, edgecolor=color, lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14, color='white'):
    xy[1] = xy[1] - 4 * sz
    text = ax.text(*xy, txt,
                   verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
