from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import copy
import os

#out_path = '/mnt/efs/fs1/pythia'
out_path = '/mnt/ssd-1/igor/data/step1'
config = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(config)

def shrink_model(model, layers):
    model = copy.deepcopy(model)
    model.base_model._modules['h'] = copy.copy(model.base_model._modules['h'][0:i+1])
    return model

def freeze_model(m):
    layer_count = len(m.base_model._modules['h'])
    for name, param in m.named_parameters():
        should_tune = name.startswith("transformer.h.{}.mlp.c_proj".format(layer_count-1))
        param.requires_grad = should_tune
        m.config.n_layer = layer_count
    return layer_count

def save_model(m, layer_count):
    output_dir = "{}/{}.layers.{}".format(out_path, config, layer_count)
    os.mkdir(output_dir)

    m.save_pretrained(output_dir)
    print("Wrote", output_dir)

def validate_model(m):
    input_ids = tokenizer.encode("How are", return_tensors="pt")
    print(tokenizer.decode(m.generate(input_ids, pad_token_id=50256).tolist()[0]))
    print("---------")

for i in range(0, len(model.base_model._modules['h'])):
    m = shrink_model(model, i+1)
    layer_count = freeze_model(m)
    save_model(m, layer_count)
    del m

