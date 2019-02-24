# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:55:53 2019

@author: ashima.garg
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import os
import matplotlib
matplotlib.use('Agg')
import config

def visualize(x_test, y_test):

    x_test = x_test.reshape([-1, 28, 28])
    embed = np.fromfile(os.path.join(config.OUT_DIR, 'embed.txt'), dtype=np.float32)
    embed = embed.reshape([-1, 2])
    feat = embed
    ax_min = np.min(embed,0)
    ax_max = np.max(embed,0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    colormap = plt.get_cmap('tab10')
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]
        patch_to_color = np.expand_dims(x_test[i], -1)
        patch_to_color = np.tile(patch_to_color, (1, 1, 3))
        patch_to_color = (1-patch_to_color) * (1,1,1) + patch_to_color * colormap(y_test[i]/10.)[:3]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(patch_to_color, zoom=0.5, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    plt.title('Embedding from the last layer of the network')
    plt.show()
    plt.savefig(os.path.join(config.OUT_DIR, 'embed.jpg'), format='jpeg', dpi=600, bbox_inches='tight')
    
    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)
    # two ways of visualization: leave with original scale