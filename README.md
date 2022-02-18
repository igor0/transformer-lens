# Pythia - Shrinking Experiments

* Shrink a pretrained gpt2-based model by removing several highest layers (like logit lens)
* Train mlp.c_proj only in the final layer


## Commands

    python run_clm.py --model_name_or_path=/tmp/models/gpt2-medium.layers.19 --tokenizer_name=gpt2 --dataset_name=wikitext --dataset_config=wikitext-103-raw-v1 --output_dir=/tmp/models/out2 --do_train --do_eval --logging_steps=50 --num_train_epochs=1
