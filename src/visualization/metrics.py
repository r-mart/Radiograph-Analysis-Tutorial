import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, target_names=None):

    accuracy = np.trace(cm) / float(np.sum(cm)) * 100
    title = f'Accuracy {accuracy:.2f} %'

    if target_names:
        target_names_y = target_names_x = target_names
    else:
        target_names_y = target_names_x = []

    fig, ax = _plot_confusion_matrix(cm, title, target_names_y, target_names_x)

    return fig, ax


def _plot_confusion_matrix(cm, title, target_names_y, target_names_x):

    font_sz_middle = 18
    font_sz_small = 16
    cmap = plt.get_cmap('Blues')

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 8))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_title(title, fontsize=font_sz_middle)

    if len(target_names_x) > 0:
        tick_marks_x = np.arange(len(target_names_x))
        tick_marks_y = np.arange(len(target_names_y))

        ax.tick_params(labelsize=font_sz_small, length=0)
        ax.set_ylim(-0.5, len(target_names_y)-0.5)
        ax.set_xlim(-0.5, len(target_names_x)-0.5)
        ax.set_xticks(tick_marks_x)
        ax.set_xticklabels(target_names_x, rotation=45)
        ax.set_yticks(tick_marks_y)
        ax.set_yticklabels(target_names_y)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=font_sz_small,
                color="white" if cm[i, j] > thresh else "black")

    # fig.tight_layout()

    ax.set_ylabel('Actual', fontsize=font_sz_middle)
    ax.set_xlabel('Predicted', fontsize=font_sz_middle)

    return fig, ax
