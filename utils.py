import torch
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt


import re

import numpy as np

from munch import Munch
from IPython.display import display

from argument_logger import argument_logger


np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


COLORS = ['blue', 'black', 'red']
TOKENIZER_WORD_SEP = '▁'


class Utils:

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.vocab = tokenizer.get_vocab()

        assert len(self.vocab.keys()) == len(set(self.vocab.values()))
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    def token_rank_in_logits(self, logits, token_id):
        sorted_indices = torch.argsort(logits, descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
        return rank + 1  # Adding 1 because rank starts from 1


    def token_prob_in_logits(self, logits, token_id):
        probabilities = F.softmax(logits, dim=-1)
        return probabilities[token_id].item()


    def remove_all_forward_hooks(self) -> None:
        self._remove_all_forward_hooks_recursive(self.model)


    @staticmethod
    def _remove_all_forward_hooks_recursive(module):
        for _, child in module._modules.items():
            if child is not None:
                if hasattr(child, "_forward_hooks"):
                    child._forward_hooks = OrderedDict()
                Utils._remove_all_forward_hooks_recursive(child)

    def predict_next_token(self, logits):
        next_token_logits = logits[0, -1, :]
        predicted_token_id = torch.argmax(next_token_logits).item()
        predicted_token = self.tokenizer.decode([predicted_token_id], skip_special_tokens=True)
        
        return predicted_token

    def get_hidden_state_vectors(self, inp, hook_type='mlp'):
        model = self.model
        assert hook_type in ['mlp', 'attention', 'residual'], "Invalid hook type"

        hooks = []
        hidden_states_cache = []
        self.remove_all_forward_hooks()

        def capture_states_hook(_, __, out):
            if isinstance(out, tuple):
                hidden_states_cache.append(out[0])
            else:
                hidden_states_cache.append(out)

        if model.__class__.__name__ in ['GPT2LMHeadModel', 'GPTJForCausalLM']:
            hook_target = model.transformer.h
        else:
            hook_target = model.model.layers
        
        for layer in hook_target:
            target = layer
            if hook_type == 'mlp':
                target = layer.mlp
            elif hook_type == 'attention':
                target = layer.post_attention_layernorm
            
            hooks.append(target.register_forward_hook(capture_states_hook))
        

        with torch.no_grad():
            outputs = model(inp.to(model.device))

        hidden_states_cache = [t.to('cuda:0') for t in hidden_states_cache]  # Replace 'cuda:0' with your preferred device
        hidden_states_cache = torch.stack(hidden_states_cache)

        for hook in hooks:
            hook.remove()

        self.remove_all_forward_hooks()

        return Munch(hidden_states=hidden_states_cache.clone(), 
                     outputs=outputs,)

    @argument_logger(False)
    def add_hooks(self, hidden_states, layer_n_list, token_n, hook_type='mlp', remove_hooks_on_start=True,
                  zero_out_mask=None):
        model = self.model

        hook_target = None
        if model.__class__.__name__ in ['GPT2LMHeadModel', 'GPTJForCausalLM']:
            hook_target = model.transformer.h
        else:
            hook_target = model.model.layers

        assert len(hidden_states) == len(layer_n_list), "hidden_states and layer_n_list must have the same length"
        assert hook_type in ['mlp', 'attention', 'residual'], "Invalid hook type"
        if zero_out_mask is not None:
            assert zero_out_mask.shape[0] == len(hook_target), "zero_out_mask first dimension must match model layers"
            layer_n_set = set(layer_n_list)
            zero_out_layers = set(torch.where(zero_out_mask.any(dim=1))[0].numpy())
            if layer_n_set.intersection(zero_out_layers):
                raise ValueError("zero_out_mask should not overlap with layers specified in layer_n_list")

        if remove_hooks_on_start:
            self.remove_all_forward_hooks()

        hooks = []

        def replace_state_hook_wrapper(h_state, _, __, out, token_indices=token_n):

            if isinstance(out, tuple):
                out_modified = out[0].clone()

                out_modified[0, token_indices] = h_state.to(out_modified.device)
                
                return (out_modified,) + out[1:]
             
            else:
                out[0, token_indices] = h_state.to(out.device)
                return out

        # if hook_target:
        for hs, layer_n in zip(hidden_states, layer_n_list):
            target = hook_target[layer_n]

            if hook_type == 'mlp':
                target = target.mlp
            elif hook_type == 'attention':
                target = target.attn if model.__class__.__name__ in ['GPT2LMHeadModel', 'GPTJForCausalLM'] else target.self_attn
                
            hooks.append(target.register_forward_hook(
                lambda _, __, out, hs=hs: replace_state_hook_wrapper(hs, _, __, out, token_indices=token_n)
            ))

        if zero_out_mask is not None:
            # print('zeroing out...')
            zero_hs = torch.zeros(hs.shape[1:], dtype=hs.dtype, device=hs.device)

            for layer_index, layer in enumerate(hook_target):
                token_indices = torch.where(zero_out_mask[layer_index] == 1)[0]
                
                if len(token_indices) > 0:
                    # print(f'zeroing out layer {layer_index} with tokens {token_indices}')

                    # mlp
                    hooks.append(layer.mlp.register_forward_hook(
                        lambda _, __, out, hs=zero_hs, token_indices=token_indices: replace_state_hook_wrapper(hs, _, __, out, token_indices=token_indices)
                    ))
                    
                    # self-attention
                    attn_layer = layer.attn if model.__class__.__name__ in ['GPT2LMHeadModel', 'GPTJForCausalLM'] else layer.self_attn
                    hooks.append(attn_layer.register_forward_hook(
                        lambda _, __, out, hs=zero_hs, token_indices=token_indices: replace_state_hook_wrapper(hs, _, __, out, token_indices=token_indices)
                    ))

        return hooks


    @argument_logger
    def add_custom_hooks(self, source_hidden_states, layer_nums_original, layer_nums_target, token_positions):
        """
        Adds hooks to specified layers of the model to replace hidden states from target layers.

        Args:
            source_hidden_states (torch.Tensor): Tensors of hidden states to be inserted from target layers.
            layer_nums_original (List[int]): List of layer numbers in the original text to which hooks will be added.
            layer_nums_target (List[int]): List of layer numbers in the target text from which hidden states are taken.
            token_positions (List[int]): Token positions for which the hidden state will be replaced.

        Returns:
            List[torch.utils.hooks.RemovableHandle]: List of hook objects that were added.
        """
        hooks = []

        model = self.model

        def replace_state_hook_wrapper(layer_num_target, module, input, output):
            output_modified = output.clone()
            if output_modified.shape[1] == 1:  # Check for 'generate' mode
                return output

            for token_n in token_positions:
                output_modified[0, token_n] = source_hidden_states[layer_num_target, token_n].to(output_modified.device)

            return output_modified

        for layer_num_original, layer_num_target in zip(layer_nums_original, layer_nums_target):
            hook_target = None
            if  model.__class__.__name__ in ['GPT2LMHeadModel', 'GPTJForCausalLM']:
                hook_target = model.transformer.h
            else:
                hook_target = model.model.layers

            target = hook_target[layer_num_original]
            if hook_target == 'mlp':
                target = target.mlp

            hook_func = lambda module, input, output, layer_num_target=layer_num_target: replace_state_hook_wrapper(layer_num_target, module, input, output)
            hooks.append(hook_target.register_forward_hook(hook_func))

        return hooks

    @property
    def num_of_layers(self):
        model = self.model

        if 'gpt2' in model.config.name_or_path or 'gpt-j' in model.config.name_or_path:
            return len(model.transformer.h)
        else:
            try:
                return len(model.model.layers)
            except AttributeError:
                raise NotImplementedError()
        
    def token_rank_list(self, outputs, token_list):
        return [self.token_rank_in_logits(outputs.logits[0, -1], self.vocab[token]) for token in token_list]

    def token_prob_list(self, outputs, token_list):
        return [self.token_prob_in_logits(outputs.logits[0, -1], self.vocab[token]) for token in token_list]

    def predict_next_tokens(self, prompt, text_length=30):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')

        generated_ids = []
        for _ in range(text_length):
            with torch.no_grad():
                inp = torch.hstack([input_ids,
                                    torch.tensor([generated_ids], device='cuda')]).type(torch.LongTensor).to('cuda')
                outputs = self.model(inp)

            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits)

            generated_ids.append(next_token_id)

        return self.tokenizer.decode(torch.hstack([input_ids, torch.tensor([generated_ids], device='cuda')])[0], skip_special_tokens=True)

    @staticmethod
    def find_sublist_indices(a, sub_a):
        """
        Find the indices of the first occurence of sub_a in a
        """
        sub_length = len(sub_a)
        
        for i in range(len(a) - sub_length + 1):
            if a[i:i + sub_length] == sub_a:

                return (i, i + sub_length - 1)
        
        raise Exception(f'{sub_a} was not found in {a}')

    
    def get_token_positions(self, text_template, subject_original, subject_target):
        """
        Find the indices of the first occurence of subject_original in text_original
        """
        
        tokenizer = self.tokenizer
        
        text_original = text_template.format(subject_original)
        text_target = text_template.format(subject_target)
        
        text_original_tokenized = tokenizer.tokenize(text_original, add_special_tokens=True)
        text_target_tokenized = tokenizer.tokenize(text_target, add_special_tokens=True)

        subject_original_tokenized = tokenizer.tokenize(subject_original)
        subject_target_tokenized = tokenizer.tokenize(subject_target)

        token_positions_original = self.find_sublist_indices(text_original_tokenized, subject_original_tokenized)
        token_positions_target = self.find_sublist_indices(text_target_tokenized, subject_target_tokenized)

        return (torch.arange(token_positions_original[0], token_positions_original[1] + 1),
                torch.arange(token_positions_target[0], token_positions_target[1] + 1))

    @staticmethod
    def create_zero_out_mask(layer_start_indices, num_layers):
        num_tokens = len(layer_start_indices)
        zero_out_mask = torch.zeros(num_layers, num_tokens, dtype=torch.long)
        
        for token_idx, layer_start in enumerate(layer_start_indices):
            if layer_start < num_layers:
                zero_out_mask[layer_start:, token_idx] = 1
        
        return zero_out_mask

    def get_hidden_states(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.model.device), output_hidden_states=True)
            
        return torch.stack(outputs.hidden_states).cpu()

    @staticmethod
    def compare_hidden_states(h_states1, h_states2):
        dot_prods = []
        
        for hs1, hs2 in zip(h_states1, h_states2):
            dot_prod = torch.dot(hs1, hs2).item()
            dot_prods.append(dot_prod)
        
        return dot_prods
    
    @staticmethod
    def clear_text(text):
        return re.sub(f'[^\w\- {TOKENIZER_WORD_SEP}?#$%&*]+', '_', text)

    @staticmethod
    def interpolate_or_reduce_fixed_ends(hidden_states, target_length):
        num_layers, seq_len, hidden_size = hidden_states.shape
        new_states = torch.zeros((num_layers, target_length, hidden_size), dtype=hidden_states.dtype)

        if target_length > seq_len:
            interval = (seq_len - 3) / (target_length - 3)
            for i in range(num_layers):
                new_states[i, 0] = hidden_states[i, 0]
                new_states[i, -1] = hidden_states[i, -1]
                for j in range(1, target_length - 1):
                    idx = (j - 1) * interval + 1
                    lower = int(idx)
                    upper = min(lower + 1, seq_len - 2)
                    weight_upper = idx - lower
                    weight_lower = 1 - weight_upper
                    new_states[i, j] = hidden_states[i, lower] * weight_lower + hidden_states[i, upper] * weight_upper
        else:
            interval = (seq_len - 2) / (target_length - 2)
            for i in range(num_layers):
                new_states[i, 0] = hidden_states[i, 0]
                new_states[i, -1] = hidden_states[i, -1]
                for j in range(1, target_length - 1):
                    start = int((j - 1) * interval) + 1
                    end = int(j * interval) + 1
                    new_states[i, j] = hidden_states[i, start:end].mean(dim=0)

        return new_states


    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)#[self.vocab_reverse[tok_id] for tok_id in ids]


    def draw_to_agg_canvas(self, figure):
        canvas = figure.canvas
        canvas.draw()

        return canvas

    def display_plot(self, fgs, n):
        # global figures
        
        plt.figure(figsize=(7, 7))
        canvas = self.draw_to_agg_canvas(fgs[n-1])
        plt.imshow(np.array(canvas.renderer._renderer))
        plt.axis('off')
        
        display(plt.gcf())  # Explicitly display the current figure
        plt.close()         # Close the figure afterwards
