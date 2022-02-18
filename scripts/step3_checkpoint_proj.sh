#!/bin/bash

checkpoint_path=$1

python ~/igor/lm-evaluation-harness/main.py --model gpt2 --model_args tokenizer=gpt2,pretrained=$checkpoint_path,aug=proj --no_cache --tasks wikitext --limit 400 > $checkpoint_path/wikitext_400.txt
