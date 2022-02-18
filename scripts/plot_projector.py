#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import gpt2aug
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

class MatrixPlot:
    def __init__(self, matrix):
        self.matrix = matrix
        self.display_labels = None

    def plot(
        self,
        *,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        colorbar=True,
        max=1.0,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.matrix
        n_classes = cm.shape[0]
        self.max = max
        self.min = -max
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin = self.min, vmax = self.max)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(1.0)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (self.max + self.min) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            cb = fig.colorbar(self.im_, ax=ax)
            cb.ax.tick_params(labelsize=120) 
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

def plot_projector(mtx, out_file):
    n = len(mtx)
    my = MatrixPlot(mtx)
    disp = my.plot(cmap="seismic", include_values=False, max=1.0)
    #disp.ax_.set_title("Hello")
    disp.ax_.set(
                ylabel="Y",
                xlabel="X",
            )

    #cm = ConfusionMatrixDisplay(proj)
    #disp = cm.plot(cmap="seismic", include_values=False)
    #disp.ax_.set_title("Hello")
    #disp.ax_.set(
        #ylabel="Y",
        #xlabel="X",
    #)
    disp.figure_.set_size_inches(n/4, n/4)
    plt.savefig(out_file)

print(sys.argv[1])
path = sys.argv[1]
proj = gpt2aug.from_pretrained("proj", path).projector.weight
plot_projector(np.array(proj.detach().numpy()), path + "/plot_proj.svg")
