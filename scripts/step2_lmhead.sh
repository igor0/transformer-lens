#!/bin/bash

model=gpt2-medium.layers.$1
base_path=/home/mchorse/igor/data

python run_clm_lm_head.py --model_name_or_path=$base_path/step1/$model --tokenizer_name=gpt2 --dataset_name=the_pile --dataset_config=enron_emails --output_dir=$base_path/step2/enron-checkpoints/$model --do_train --do_eval --logging_steps=50 --num_train_epochs=1
