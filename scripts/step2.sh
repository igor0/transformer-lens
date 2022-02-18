#!/bin/bash

model=gpt2-medium.layers.$1
aug=$2
base_path=/mnt/ssd-1/igor
train_file=$base_path/data/pile/val.json

python run_clm_aug.py --model_name_or_path=$base_path/data/step1/$model --tokenizer_name=gpt2 --train_file=$train_file --output_dir=$base_path/data/step2/pile-$aug/$model --do_train --logging_steps=50 --num_train_epochs=1 --model_aug=$aug
