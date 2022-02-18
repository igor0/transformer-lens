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

import torch
import torch.nn
import transformers
from transformers import GPT2LMHeadModel
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

REF_LM_HEAD = "/mnt/ssd-1/igor/data/step2/pile-lm_head/gpt2-medium.layers.1"

class GPT2AugReplaceLMH(GPT2LMHeadModel):
    def from_pretrained(*args, **kwargs):
        ref = GPT2AugLMHead.from_pretrained(REF_LM_HEAD)
        model = GPT2LMHeadModel.from_pretrained(*args, **kwargs)
        model.lm_head = ref.lm_head
        return model

class GPT2AugCProj(GPT2LMHeadModel):
    def should_tune(self, param_name):
        layer_count = len(self.base_model._modules['h'])
        return param_name.startswith("transformer.h.{}.mlp.c_proj".format(layer_count-1))

class GPT2AugLMHead(GPT2LMHeadModel):
    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        return

    def should_tune(self, param_name):
        return "lm_head" in param_name

class GPT2AugLNF(GPT2LMHeadModel):
    def should_tune(self, param_name):
        return "ln_f" in param_name

class GPT2AugProjected(GPT2LMHeadModel):
    def init_weights(self):
        in_features = self.lm_head.in_features

        lm_head = self.lm_head
        del self.lm_head

        self.projector = torch.nn.Linear(in_features=in_features, out_features=in_features)
        self.lm_head = lm_head
        super().init_weights()

    def should_tune(self, param_name):
        return "projector" in param_name

    def parallelize(self, device_map=None):
        super().parallelize(device_map)
        self.projector = self.projector.to(self.transformer.first_device)

    def deparallelize(self):
        super().deparallelize()
        self.projector = self.projector.to("cpu")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        hidden_states = self.projector(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class GPT2AugScalar(GPT2LMHeadModel):
    def init_weights(self):
        in_features = self.lm_head.in_features

        lm_head = self.lm_head
        del self.lm_head

        self.mult = 10000.0
        self.projector = torch.nn.Parameter(torch.Tensor([1. / self.mult]))
        self.lm_head = lm_head
        super().init_weights()

    def should_tune(self, param_name):
        return "projector" in param_name

    def parallelize(self, device_map=None):
        super().parallelize(device_map)
        self.projector = self.projector.to(self.transformer.first_device)

    def deparallelize(self):
        super().deparallelize()
        self.projector = self.projector.to("cpu")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        hidden_states = self.projector * self.mult * hidden_states
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT2AugProjProj(GPT2LMHeadModel):
    def init_weights(self):
        in_features = self.lm_head.in_features

        lm_head = self.lm_head
        del self.lm_head

        N = self.N()
        self.proj_lora1 = torch.nn.Linear(in_features=in_features, out_features=N)
        self.proj_lora2 = torch.nn.Linear(in_features=N, out_features=in_features)
        self.lm_head = lm_head
        super().init_weights()

    def should_tune(self, param_name):
        return "proj_lora" in param_name

    def parallelize(self, device_map=None):
        super().parallelize(device_map)
        self.projector = self.projector.to(self.transformer.first_device)

    def deparallelize(self):
        super().deparallelize()
        self.projector = self.projector.to("cpu")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        hidden_states = hidden_states + self.proj_lora2(self.proj_lora1(hidden_states))
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT2AugProjProj1(GPT2AugProjProj):
    def N(self):
        return 1

class GPT2AugProjProj2(GPT2AugProjProj):
    def N(self):
        return 2

class GPT2AugProjProj4(GPT2AugProjProj):
    def N(self):
        return 4

class GPT2AugProjProj8(GPT2AugProjProj):
    def N(self):
        return 8

class GPT2AugProjProj16(GPT2AugProjProj):
    def N(self):
        return 16

class GPT2AugProjProj32(GPT2AugProjProj):
    def N(self):
        return 32

class GPT2AugProjProj64(GPT2AugProjProj):
    def N(self):
        return 64

class GPT2AugProjProj128(GPT2AugProjProj):
    def N(self):
        return 128

class GPT2AugProjProj256(GPT2AugProjProj):
    def N(self):
        return 256

def from_pretrained(model_aug, *args, **kwargs):
    if model_aug is None:
        model_type = transformers.GPT2LMHeadModel
    elif model_aug == "lm_head":
        model_type = GPT2AugLMHead
    elif model_aug == "proj":
        model_type = GPT2AugProjected
    elif model_aug == "scalar":
        model_type = GPT2AugScalar
    elif model_aug == "cproj":
        model_type = GPT2AugCProj
    elif model_aug == "replace_lmh":
        model_type = GPT2AugReplaceLMH
    elif model_aug == "ln_f":
        model_type = GPT2AugLNF
    elif model_aug == "projproj1":
        model_type = GPT2AugProjProj1
    elif model_aug == "projproj2":
        model_type = GPT2AugProjProj2
    elif model_aug == "projproj4":
        model_type = GPT2AugProjProj4
    elif model_aug == "projproj8":
        model_type = GPT2AugProjProj8
    elif model_aug == "projproj16":
        model_type = GPT2AugProjProj16
    elif model_aug == "projproj32":
        model_type = GPT2AugProjProj32
    elif model_aug == "projproj64":
        model_type = GPT2AugProjProj64
    elif model_aug == "projproj128":
        model_type = GPT2AugProjProj128
    elif model_aug == "projproj256":
        model_type = GPT2AugProjProj256
    else:
        model_type = None

    return model_type.from_pretrained(*args, **kwargs)
