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
import sys
import torch
import torch.linalg

path = sys.argv[1]
#path = "/mnt/ssd-1/igor/data/step2/pile-ln_f-fp16/gpt2-medium.layers.12"
#path = "/mnt/ssd-1/igor/data/step2/pile-proj-bad/gpt2-medium.layers.1"
model = gpt2aug.from_pretrained("proj", path)
for name, m in model.named_children():
    if name == "projector":
        if False:
            for i in range(len(m.weight)):
                for j in range(len(m.weight)):
                    if torch.abs(m.weight[i][j]) > 0.05:
                        print(i, j, m.weight[i][j].item())

        if False:
            # look at particular row
            row = 909
            arr = []
            for j in range(len(m.weight)):
                arr.append((m.weight[row][j].item(), j))

            arr.sort(key = lambda x: -abs(x[909]))
            for x in arr:
                print(x)

        if False:
            # look at particular row
            arr = []
            for i in range(len(m.weight)):
                arr.append((m.weight[i][0].item(), i))

            arr.sort(key = lambda x: -abs(x[0]))
            for x in arr:
                print(x)

 
        if False:
            # look at large values
            arr = []
            for i in range(len(m.weight)):
                for j in range(len(m.weight)):
                    arr.append((m.weight[i][j].item(), i, j))

            arr.sort(key = lambda x: -abs(x[0]))
            for x in arr[:len(m.weight)]:
                print(x)

        if False:
            # look at the diagonals
            arr = []
            for j in range(len(m.weight)):
                arr.append((m.weight[j][j].item(), j))

            arr.sort(key = lambda x: -x[0])
            for x in arr:
                print(x)

        if False:
            # Count values > threshold
            thres = 0.01
            mat = m.weight
            zero_tensor = torch.zeros(len(mat), len(mat))
            mat = torch.where(mat > thres, mat, zero_tensor)

            val_pos = 0
            val_zero = 0
            for i in range(len(m.weight)):
                for j in range(len(m.weight)):
                    if mat[i][j] != 0:
                        val_pos = val_pos + 1
                    else:
                        val_zero = val_zero + 1

            print(val_pos, "/", val_pos + val_zero)

        if False:
            # Count positive values whose abs(x) > abs_thres
            abs_thres = 0.2
            thres = 0
            mat = m.weight

            val_pos = 0
            val_zero = 0
            for i in range(len(m.weight)):
                for j in range(len(m.weight)):
                    if abs(mat[i][j]) < abs_thres:
                        continue
                    if mat[i][j] > thres:
                        val_pos = val_pos + 1
                    else:
                        val_zero = val_zero + 1

            print(val_pos, "/", val_pos + val_zero)


        if False:
            # Calculate row sum
            for i in range(len(m.weight)):
                print(i, m.weight[i].sum().item())

        if False:
            # Calculate col sum
            for i in range(len(m.weight)):
                print(i, m.weight[:,i].sum().item())


        if False:
            weight_inv = torch.inverse(m.weight * 30)

            # look at large values
            arr = []
            for i in range(len(weight_inv)):
                for j in range(len(weight_inv)):
                    arr.append((weight_inv[i][j].item(), i, j))

            arr.sort(key = lambda x: -abs(x[0]))
            for x in arr[:len(weight_inv)]:
                print(x)


        if True:
            torch.set_printoptions(threshold=10_000)
            #print(path)
            num = path.split(".")[-1]
            print("layer_{}_ln_s_weight = {}".format(num, model.transformer.ln_f.weight))
            print("layer_{}_ln_s_bias = {}".format(num, model.transformer.ln_f.bias))
