import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import hashlib
from typing import Dict, Optional

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.stats import entropy
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import gc


from argument_logger import argument_logger
from tokens_to_check import TokensToCheck as TokensToCheckOriginal
from graph_generator import generate_and_display_graphs

from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence


class Experiment:
    
    slots = ['text_original', 'subject_original', 'subject_target', 'tokens_to_check']  # legacy for hashing
    
    
    def __init__(self, model, tokenizer, text_template: str, subject_original: str, subject_target: str, tokens_to_check,
                 correct_token_prob_threshold=0.2, ignore_reliability_check=False, **kwargs):
        del kwargs
        
        self.model = model
        self.tokenizer = tokenizer
        self.correct_token_prob_threshold = correct_token_prob_threshold
    
        # len_tokens_original = len(tokenizer(subject_original).input_ids)
        # len_tokens_target = len(tokenizer(subject_target).input_ids)

        self.subject_original, self.subject_target = self.pad_sentences(subject_original, subject_target, tokenizer, 
                                                                        capitalize=text_template.startswith('{}'))
        
        self.text_template = text_template
        self.tokens_to_check = self._select_most_probable_tokens_to_check(tokens_to_check)
        
        if not ignore_reliability_check:
            self._check_reliability()

    def _select_most_probable_token(self, text, candidates, reverse=False):
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model(ids.to(self.model.device))
        candidates_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in candidates]
        # candidates_ids = torch.tensor(candidates_ids).squeeze()
        max_candidate_index = torch.argmax(outputs.logits[0, -1, candidates_ids] * (-1 if reverse else 1)).item()

        return max_candidate_index

    def _select_most_probable_tokens_to_check(self, tokens_to_check):
        if isinstance(tokens_to_check.original, list):
            max_candidate_index = self._select_most_probable_token(self.text_original, tokens_to_check.original)
            new_original_word = tokens_to_check.original_word[max_candidate_index]
        else:
            new_original_word = tokens_to_check.original_word

        if isinstance(tokens_to_check.target, list):
            max_candidate_index = self._select_most_probable_token(self.text_target, tokens_to_check.target)
            new_target_word = tokens_to_check.target_word[max_candidate_index]
        else:
            new_target_word = tokens_to_check.target_word

        if isinstance(tokens_to_check.control, list):
            max_candidate_index = self._select_most_probable_token(self.text_original, tokens_to_check.control)
            new_control_word = tokens_to_check.control_word[max_candidate_index]
        else:
            new_control_word = tokens_to_check.control_word


        new_tokens_to_check = TokensToCheckOriginal(new_original_word, new_target_word, new_control_word, tokens_to_check.add_space, self.tokenizer)

        return new_tokens_to_check

        
    def _check_reliability(self):
        self._check_prompt_reliability("original", self.text_original, self.tokens_to_check.original)
        self._check_prompt_reliability("target", self.text_target, self.tokens_to_check.target)
    
    def _check_prompt_reliability(self, prompt_type, text, token_label):
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model(ids.to(self.model.device))
        probabilities = F.softmax(outputs.logits, dim=-1)
        token_id = self.tokenizer.get_vocab()[token_label]
        tprob = probabilities[0, -1, token_id].item()
        top_prob, top_idx = torch.topk(probabilities[0, -1], 1, dim=-1)
        print(f'{prompt_type.capitalize()} prompt: "{text}"; {token_label} tprob: {tprob:.3f}, top token: {self.tokenizer.convert_ids_to_tokens(top_idx.item())} with prob: {top_prob.item():.3f}')
        
        if top_idx.item() != token_id:
            raise Exception(f"Top predicted token in the {prompt_type} prompt is not '{token_label}' (p={tprob:.3f}) but '{self.tokenizer.convert_ids_to_tokens(top_idx.item())}' (p={top_prob.item():.3f}).")

        if tprob < self.correct_token_prob_threshold:
            raise Exception(f"Token probability in the {prompt_type} prompt for '{token_label}' is {tprob:.3f}, which is below the threshold ({self.correct_token_prob_threshold}).")
        
    @staticmethod
    def pad_sentences(s1, s2, tokenizer, capitalize):
        padding = 'the '

        prefix1 = ''
        prefix2 = ''
        s1_new = s1[0].upper() + s1[1:] if capitalize else s1
        s2_new = s2[0].upper() + s2[1:] if capitalize else s2

        while len(tokenizer(s1_new).input_ids) != len(tokenizer(s2_new).input_ids):
            if len(tokenizer(s1_new).input_ids) < len(tokenizer(s2_new).input_ids):
                prefix1 = padding + prefix1
                prefix1 = prefix1.capitalize() if capitalize else prefix1
                s1_new = prefix1 + s1
            else:
                prefix2 = padding + prefix2
                prefix2 = prefix2.capitalize() if capitalize else prefix2
                s2_new = prefix2 + s2
                
        return s1_new, s2_new

    def __repr__(self) -> str:
        return (f"Experiment(\n"
                f"    text_template='{self.text_template}',\n"
                f"    subject_original='{self.subject_original}',\n"
                f"    subject_target='{self.subject_target}',\n"
                f"    tokens_to_check='{self.tokens_to_check}'\n"
                f")")
    
    @property
    def text_original(self):
        return self.text_template.format(self.subject_original)
    
    @property
    def text_target(self):
        return self.text_template.format(self.subject_target)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"No such attribute: {key}")


@dataclass
class ExperimentSetup:
    __experiments_to_run__ = ['residual', 'attention', 'mlp', 'compare_dot_prods', 'compare_lens', 'show_graphs']
    __global_experiment_settings__ = ['model_name', 'replacement_method', 'zero_out_following']
    __specific_experiment_settings__ = ['residual_kwargs', 'attention_kwargs', 'mlp_kwargs', 'compare_dot_prods_kwargs', 'compare_lens_kwargs', 'show_graphs_kwargs']

    model_name: str
    zero_out_following: bool

    residual: bool = True
    attention: bool = False
    mlp: bool = False
    compare_dot_prods: bool = False
    compare_lens: bool = False
    show_graphs: bool = True

    replacement_method: str = 'exact'

    residual_kwargs: Optional[Dict] = field(default=None)
    attention_kwargs: Optional[Dict] = field(default=None)
    mlp_kwargs: Optional[Dict] = field(default=None)
    compare_lens_kwargs: Optional[Dict] = field(default=None)
    compare_dot_prods_kwargs: Optional[Dict] = field(default=None)
    show_graphs_kwargs: Optional[Dict] = field(default=None)

    def __post_init__(self):
        if self.residual:
            if self.residual_kwargs is None:
                self.residual_kwargs = {}
        else:
            self.residual_kwargs = {}
        
        if self.attention:
            if self.attention_kwargs is None:
                self.attention_kwargs = {'window_size': 1}
        else:
            self.attention_kwargs = None
        
        if self.mlp:
            if self.mlp_kwargs is None:
                self.mlp_kwargs = {'window_size': 1}
        else:
            self.mlp_kwargs = None
        
        if self.compare_lens:
            if self.compare_lens_kwargs is None:
                self.compare_lens_kwargs = {'layer_nums_target': 7, 'layer_nums_original': 7}
        else:
            self.compare_lens_kwargs = None
        
        if self.compare_dot_prods:
            if self.compare_dot_prods_kwargs is None:
                self.compare_dot_prods_kwargs = {'layer_nums_target': 7, 'layer_nums_original': 7}
        else:
            self.compare_dot_prods_kwargs = None
        
        if self.show_graphs:
            if self.show_graphs_kwargs is None:
                self.show_graphs_kwargs = {}
        else:
            self.show_graphs_kwargs = None

    def __repr__(self) -> str:
        return (
            f"ExperimentSetup(\n"
            f"    model_name={self.model_name!r},\n"
            f"    residual={self.residual},\n"
            f"    attention={self.attention},\n"
            f"    mlp={self.mlp},\n"
            f"    compare_dot_prods={self.compare_dot_prods},\n"
            f"    compare_lens={self.compare_lens},\n"
            f"    show_graphs={self.show_graphs},\n"
            f"    replacement_method={self.replacement_method!r},\n"
            f"    zero_out_following={self.zero_out_following},\n"
            f"    residual_kwargs={self.residual_kwargs},\n"
            f"    attention_kwargs={self.attention_kwargs},\n"
            f"    mlp_kwargs={self.mlp_kwargs},\n"
            f"    compare_lens_kwargs={self.compare_lens_kwargs},\n"
            f"    compare_dot_prods_kwargs={self.compare_dot_prods_kwargs},\n"
            f"    show_graphs_kwargs={self.show_graphs_kwargs}\n"
            f")"
        )

@dataclass
class ExperimentResult:
    experiment: Experiment
    setup: ExperimentSetup
    graph: plt.Figure
    plot: plt.Figure
    axes: plt.Axes
    hash: str

    @property
    def raw(self):
        plot_data = {
            'model_name': self.setup.model_name,
            'prompts': {k: getattr(self.experiment, k) for k in Experiment.slots},
            # 'hash': self.hash,
            'experiments': [],
        }
        
        for ax, experiment_kind in zip(self.axes, ExperimentManager.__experiments_order__[1:]):  # [1:] because of 'show_graphs'
            setup = {s: getattr(self.setup, s) for s in ExperimentSetup.__global_experiment_settings__}
            setup[f'{experiment_kind}_kwargs'] = getattr(self.setup, f'{experiment_kind}_kwargs')

            axes = ax.children_axes if hasattr(ax, 'children_axes') else [ax]
            ax_data = {
                'kind': experiment_kind,
                'title': ax.get_title(),
                'experimental_setup': setup,

                'plots': [{'title': ax.get_title(),
                           'lines': self._get_ax_lines_data(ax),} for ax in axes],
            }

            plot_data['experiments'].append(ax_data)

        return plot_data
    

    def _get_ax_lines_data(self, ax):
        lines_data = []
        for line in ax.get_lines():
            line_data = {
                'x': list(line.get_xdata()),
                'y': list(line.get_ydata()),
                'label':  line.get_label(),
            }
            lines_data.append(line_data)

        return lines_data


class ExperimentManager:
    __experiments_order__ = ['show_graphs', 'residual', 'mlp', 'attention', 'compare_lens', 'compare_dot_prods']

    def __init__(self, experiment_setup, utils, tokenizer_word_sep, model_name, lensing=None, neptune_loader=None):
        # self.experiments = []

        self.experiment_setup = experiment_setup
        self._neptune_loader = neptune_loader
        self._model_name = model_name
        self._utils = utils
        self._lensing = lensing
        self._tokenizer_word_sep = tokenizer_word_sep

    
    @staticmethod
    def _md5_hash(*strings):
        # print(f"Calculating hash for {'_____'.join(map(str, strings))}")
        hash_object = hashlib.md5()
        
        for string in '_____'.join(map(str, strings)):
            hash_object.update(string.encode('utf-8'))

        return hash_object.hexdigest()

    def hash(self, experiment):
        return self._md5_hash(*([experiment[e] if not isinstance(e, TokensToCheckOriginal)
                                else experiment[e].original + experiment[e].target + experiment[e].control
                                for e in Experiment.slots] + [str(x) for x in self.experiment_setup.__dict__.values()]))

    def add_experiment(self, experiment, force=False) -> ExperimentResult:
        # print(f'Checking {self.hash(experiment)}')

        if not self._neptune_loader is None:
            is_exists = self._neptune_loader.check_result(self._model_name, self.hash(experiment))
        else:
            is_exists = False

        if is_exists and not force:
            print(f'Experiment {experiment} already exists')
            return
        else:
            print(f'Adding experiment {experiment}')

        result = self.run_experiment_and_visualize(experiment, )

        if self._neptune_loader is not None:
            self._neptune_loader.upload_result(self._model_name, result)
        
        # print(f'Done {self.hash(experiment)}, result.experiment.hash: {self.hash(result.experiment)}')
        return result

    def run_experiment_and_visualize(self, experiment):
        fig, graph = None, None

        residual = self.experiment_setup.residual
        attention = self.experiment_setup.attention
        mlp = self.experiment_setup.mlp
        compare_dot_prods = self.experiment_setup.compare_dot_prods
        compare_lens = self.experiment_setup.compare_lens
        show_graphs = self.experiment_setup.show_graphs

        residual_kwargs = self.experiment_setup.residual_kwargs
        mlp_kwargs = self.experiment_setup.mlp_kwargs
        attention_kwargs = self.experiment_setup.attention_kwargs
        compare_lens_kwargs = self.experiment_setup.compare_lens_kwargs
        show_graphs_kwargs = self.experiment_setup.show_graphs_kwargs
        compare_dot_prods_kwargs = self.experiment_setup.compare_dot_prods_kwargs

        utils = self._utils
        model = utils.model
        tokenizer = utils.tokenizer
        find_sublist_indices = utils.find_sublist_indices
        residual = self.experiment_setup.residual
        attention = self.experiment_setup.attention
        mlp = self.experiment_setup.mlp
        compare_dot_prods = self.experiment_setup.compare_dot_prods
        compare_lens = self.experiment_setup.compare_lens
        show_graphs = self.experiment_setup.show_graphs
        zero_out_following = self.experiment_setup.zero_out_following
        replacement_method = self.experiment_setup.replacement_method

        utils.remove_all_forward_hooks()

        text_original = experiment.text_original
        subject_original = experiment.subject_original
        subject_target = experiment.subject_target
        tokens_to_check = experiment.tokens_to_check
        
        subject_original_tokenized = tokenizer.tokenize(subject_original)
        subject_target_tokenized = tokenizer.tokenize(subject_target)
        
        assert len(subject_original_tokenized) == len(subject_target_tokenized)
        
        print(tokenizer.tokenize(subject_original))
        print(tokenizer.tokenize(subject_target))

        if show_graphs:
            graph = generate_and_display_graphs(model, tokenizer, text_original, experiment.text_target)

        if residual or mlp or attention or compare_lens or compare_dot_prods:
            num_of_plots =  max(1, sum(map(int, [mlp, residual, attention, compare_lens, compare_dot_prods])))
            
            fig = plt.figure(figsize=(7 * num_of_plots, 7))
            gs = fig.add_gridspec(4, 2 * num_of_plots, wspace=0.3, hspace=0.3)
                        
            text_target = experiment.text_target
            
            sub_index_original = find_sublist_indices(tokens_to_check.original_word, tokens_to_check.original.replace(self._tokenizer_word_sep, ""))[1] + 1
            sub_index_target = find_sublist_indices(tokens_to_check.target_word, tokens_to_check.target.replace(self._tokenizer_word_sep, ""))[1] + 1
            
            title = f'Model: {self._model_name}; Zero Out Following: {zero_out_following}; Replacement method: {replacement_method}\n' \
                    f'original: {text_original} [{tokens_to_check.original}]{tokens_to_check.original_word[sub_index_original:]}\n' \
                    f'target: {text_target} [{tokens_to_check.target}]{tokens_to_check.target_word[sub_index_target:]}'
            fig.suptitle(title, fontsize=10)

        axes = []
        if residual:
            ax_residual = fig.add_subplot(gs[:4, :2])
            ax_residual.set_title('Replace $x_{res2}$')

            self.perform_replace_residual_hidden_states(
                text_template=experiment.text_template,
                subject_original=subject_original,
                subject_target=subject_target,
                tokens_to_check=tokens_to_check,
                plot_line_style='-',
                plot_line_width=2,
                ax=ax_residual,
                **residual_kwargs,
            )
            axes.append(ax_residual)

        if mlp:
            axs_index = sum(map(int, [residual, mlp])) - 1
            ax_mlp = fig.add_subplot(gs[1:3, 2 * axs_index:2*(axs_index+1)])
            
            ax_mlp.set_title('Replace $x_{mlp}$' + f' (Window size: {mlp_kwargs["window_size"]})')

            self.perform_replace_partial_hidden_states(
                text_template=experiment.text_template,
                subject_original=subject_original,
                subject_target=subject_target,
                hook_type='mlp',
                tokens_to_check=tokens_to_check,
                plot_line_style='--',
                plot_line_width=2,
                ax=ax_mlp,
                window_size=mlp_kwargs['window_size'],
            )
            axes.append(ax_mlp)

        if attention:
            axs_index = sum(map(int, [residual, mlp, attention])) - 1
            ax_attn = fig.add_subplot(gs[1:3, 2 * axs_index:2*(axs_index+1)])
            
            ax_attn.set_title('Replace $x_{attn}$' + f' (Window size: {attention_kwargs["window_size"]})')

            self.perform_replace_partial_hidden_states(
                text_template=experiment.text_template,
                subject_original=subject_original,
                subject_target=subject_target,
                hook_type='attention',
                tokens_to_check=tokens_to_check,
                window_size=attention_kwargs['window_size'],
                plot_line_style='--',
                plot_line_width=2,
                ax=ax_attn,
            )
            axes.append(ax_attn)
            
        if compare_lens:
            print('compare_lens_kwargs', compare_lens_kwargs)
            
            axs_index = sum(map(int, [residual, mlp, attention, compare_lens])) - 1
            ax_lens = fig.add_subplot(gs[1:3, 2 * axs_index:2*(axs_index+1)])
            
            self.compare_lens_flows(experiment, ax=ax_lens, **compare_lens_kwargs);
            axes.append(ax_lens)
            
        if compare_dot_prods:
            print('compare_dot_prods_kwargs', compare_dot_prods_kwargs)
            
            axs_index = sum(map(int, [residual, mlp, attention, compare_lens, compare_dot_prods])) - 1
            ax_dot_prods = fig.add_subplot(gs[1:3, 2 * axs_index:2*(axs_index+1)])
            
            self.compare_dot_prods_on_last_token(experiment, ax=ax_dot_prods, **compare_dot_prods_kwargs);
            axes.append(ax_dot_prods)

    #     if residual or mlp:
    #         min_ylim = min(ax.get_ylim()[0] for ax in fig.axes)
    #         max_ylim = max(ax.get_ylim()[1] for ax in fig.axes)

    #         for ax in fig.axes:
    #             ax.set_ylim(min_ylim, max_ylim)

    #     fig.tight_layout()
        plt.show()

        return ExperimentResult(experiment=experiment, graph=graph, plot=fig,
                                 setup=self.experiment_setup, axes=axes, hash=self.hash(experiment))
    
    @argument_logger
    def compare_dot_prods_on_last_token(self, experiment: Experiment, layer_nums_target=5, layer_nums_original=5, ax=None):
        torch.cuda.empty_cache()

        utils = self._utils
        tokenizer = utils.tokenizer
        get_hidden_states = utils.get_hidden_states

        num_subject_tokens = len(tokenizer.tokenize(experiment['subject_original']))
        num_text_tokens = len(tokenizer.tokenize(experiment['text_original']))

        _zero_out_mask = utils.create_zero_out_mask(torch.concat([
            torch.tensor([utils.num_of_layers]),
            torch.ones(num_subject_tokens, dtype=int) * (layer_nums_original + 1),
        #     torch.ones(len(tokenizer.tokenize(experiment['text_original'])) - num_subject_tokens, dtype=int) * (layer_nums_original + 10 + 1),
        ]), utils.num_of_layers)# * 0

        print(experiment)
        # print(f'zero_out_mask:\n{_zero_out_mask}')
        print(tokenizer.tokenize(experiment.text_original))
        print(tokenizer.tokenize(experiment.text_target))
        utils.remove_all_forward_hooks()
        clean_original_h_states = get_hidden_states(experiment.text_original)[:, 0, -1, :].float()
        clean_target_h_states = get_hidden_states(experiment.text_target)[:, 0, -1, :].float()
    #     clean_original_traj, _ = make_lens_trajectory(experiment['text_original'])
    #     clean_target_traj, _ = make_lens_trajectory(experiment['text_original'].replace(experiment['subject_original'], experiment['subject_target']))
        utils.remove_all_forward_hooks()

        outputs, prob_object = self.replace_custom_partial_hidden_state(
            text_original=experiment['text_original'],
            subject_original=experiment['subject_original'],
            subject_target=experiment['subject_target'],
            tokens_to_check=experiment['tokens_to_check'],

            layer_nums_original=[layer_nums_original],
            layer_nums_target=[layer_nums_target],

            hook_type='residual',
            zero_out_mask=_zero_out_mask,

            argument_logger_do_print=False,
        )
        print(prob_object)
        
        modified_h_states = get_hidden_states(experiment['text_original'])[:, 0, -1, :]
        modified_vs_original = 1 - F.cosine_similarity(clean_original_h_states, modified_h_states, dim=1)
        modified_vs_target = 1 - F.cosine_similarity(clean_target_h_states, modified_h_states, dim=1)

        utils.remove_all_forward_hooks()

        gc.collect()
        torch.cuda.empty_cache()

        ax.plot(modified_vs_original, c='blue')
        ax.plot(modified_vs_target, c='red')
        
        ax.set_xlabel('Layer range')
        ax.set_ylabel('Cosine distance value')

        x_ticks = np.arange(1, utils.num_of_layers + 1, 1)
        x_labels = x_ticks
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        ax.set_title(f'Cosine distance ({layer_nums_target} -> {layer_nums_original})')
        ax.grid()

        return ax
    
    # @argument_logger
    def compare_lens_flows(self, experiment: Experiment, layer_nums_target=5, layer_nums_original=5,
                           token_index=-1, visualize_stat=False, regime='tuned_lens', zero_out_following=True, ax=None):
        torch.cuda.empty_cache()

        utils = self._utils
        tokenizer = utils.tokenizer
        create_zero_out_mask = utils.create_zero_out_mask
                
        make_lens_trajectory = self._lensing.make_lens_trajectory
        visualize_statistic = self._lensing.visualize_statistic

        num_subject_tokens = len(tokenizer.tokenize(experiment['subject_original']))
        num_text_tokens = len(tokenizer.tokenize(experiment['text_original']))

        _zero_out_mask = create_zero_out_mask(torch.concat([
            torch.tensor([utils.num_of_layers]),
            torch.ones(num_subject_tokens, dtype=int) * (layer_nums_original + 1),
        #     torch.ones(len(tokenizer.tokenize(experiment['text_original'])) - num_subject_tokens, dtype=int) * (layer_nums_original + 10 + 1),
        ]), utils.num_of_layers)# * 0

        if not zero_out_following:
            _zero_out_mask = _zero_out_mask * 0

        print(experiment)
        # print(f'zero_out_mask:\n{_zero_out_mask}')
        print(tokenizer.tokenize(experiment.text_original))
        print(tokenizer.tokenize(experiment.text_target))
        utils.remove_all_forward_hooks()
        clean_original_traj, _ = make_lens_trajectory(experiment.text_original)
        clean_target_traj, _ = make_lens_trajectory(experiment.text_target)
        utils.remove_all_forward_hooks()

        outputs, prob_object = self.replace_custom_partial_hidden_state(
            text_template=experiment['text_template'],
            subject_original=experiment['subject_original'],
            subject_target=experiment['subject_target'],
            tokens_to_check=experiment['tokens_to_check'],

            layer_nums_original=[layer_nums_original],
            layer_nums_target=[layer_nums_target],

            hook_type='residual',
            zero_out_mask=_zero_out_mask,

            argument_logger_do_print=False,
        )
        print(prob_object)
        
        title = f'target layers {layer_nums_target} -> original layers {layer_nums_original}\nmodified vs clean original'
        _, modified_vs_original_stat = make_lens_trajectory(
            experiment['text_original'], regime=regime, statistic="kl_divergence", other=clean_original_traj)
        
        if visualize_stat:
            visualize_statistic(modified_vs_original_stat, title=title, )
        
        title = f'target layers {layer_nums_target} -> original layers {layer_nums_original}\nmodified vs clean target'
        _, modified_vs_target_stat = make_lens_trajectory(
            experiment['text_original'], regime=regime, statistic="kl_divergence", other=clean_target_traj)

        if visualize_stat:
            visualize_statistic(modified_vs_target_stat, title=title, )

        utils.remove_all_forward_hooks()

        gc.collect()
        torch.cuda.empty_cache()

        ax.plot(modified_vs_original_stat.stats[:, token_index], c='blue', label='vs original')
        ax.plot(modified_vs_target_stat.stats[:, token_index], c='red', label='vs target')
        
        ax.set_xlabel('Layer range')
        ax.set_ylabel('KL divergence')

        x_ticks = np.arange(1, utils.num_of_layers + 1, 1)
        x_labels = x_ticks
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend()
        
        ax.set_title(f'Compare lens flows ({layer_nums_target} -> {layer_nums_original})\nregime: {regime}')
        ax.grid()

        return ax

    # @argument_logger
    def replace_custom_partial_hidden_state(self, text_template, subject_original, subject_target, tokens_to_check,
                                            layer_nums_original, layer_nums_target,
                                            hook_type='residual', zero_out_mask=None,
                                            argument_logger_do_print=True):
        """
        Replaces hidden states of specific layers in the original text with the hidden states 
        from corresponding layers in the target text.

        Args:
            text_original (str): The original text.
            subject_target (str): The target subject text.
            layer_nums_original (List[int]): List of layer numbers in the original text to be replaced.
            layer_nums_target (List[int]): List of layer numbers in the target text to copy hidden states from.
        """
        utils = self._utils
        model = utils.model
        tokenizer = utils.tokenizer
        get_token_positions = utils.get_token_positions

        utils.remove_all_forward_hooks()

        text_original = text_template.format(subject_original)
        # text_target = text_template.format(subject_target)
        
        token_positions_original, token_positions_target = get_token_positions(text_template, subject_original, subject_target)
        
    #     print(f'token_positions_original, token_positions_target: {token_positions_original, token_positions_target}')

        saved_hidden_states = utils.get_hidden_state_vectors(tokenizer(subject_target, return_tensors='pt').input_ids,
                                                            hook_type=hook_type)
        saved_hidden_states = saved_hidden_states.hidden_states

        subject_0_hidden_states = saved_hidden_states[layer_nums_target, 0][:, token_positions_target]

    #     print(f'subject_0_hidden_states.shape: {subject_0_hidden_states.shape}, len(layer_nums_original): {len(layer_nums_original)}')
        
    #     print(list(zip(subject_0_hidden_states, layer_nums_original)))

        hooks = utils.add_hooks(hidden_states=subject_0_hidden_states.to(model.dtype),
                                layer_n_list=layer_nums_original,
                                token_n=token_positions_original,
                                hook_type=hook_type,
                                zero_out_mask=zero_out_mask,
                                argument_logger_do_print=argument_logger_do_print)
        ids_original = tokenizer(text_original, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model(ids_original.to(model.device))

        token_kinds = ['original', 'target', 'control']
        
    #     print(tokens_to_check)
        prob = utils.token_prob_list(outputs, [getattr(tokens_to_check, k) for k in token_kinds])
        prob = dict(zip(token_kinds, prob))

    #     print(dict(zip(token_kinds, prob)))
    #     print(dict(zip([getattr(tokens_to_check, k) for k in probs.keys()], prob)))
        
        return outputs, prob

    # @argument_logger
    def perform_replace_residual_hidden_states(self, text_template, subject_original, subject_target, tokens_to_check,
                                               token_positions_original=None, token_positions_target=None,
                                               plot_line_style='-', plot_line_width=1, plot_prefix='', ax=None):

        utils = self._utils
        model = utils.model
        tokenizer = utils.tokenizer
        get_token_positions = utils.get_token_positions
        create_zero_out_mask = utils.create_zero_out_mask
        interpolate_or_reduce_fixed_ends = utils.interpolate_or_reduce_fixed_ends
        zero_out_following = self.experiment_setup.zero_out_following
        replacement_method = self.experiment_setup.replacement_method


        utils.remove_all_forward_hooks()

        if ax is None:
            ax = plt.gca()

        if not hasattr(ax, 'children_axes'):
            ax.children_axes = []

        ax1 = inset_axes(ax, width="100%", height="46%", loc="upper center", borderpad=0)
        ax2 = inset_axes(ax, width="100%", height="46%", loc="lower center", borderpad=0)

        ax.children_axes.append(ax1)
        ax.children_axes.append(ax2)

        text_original = text_template.format(subject_original)
        text_target = text_template.format(subject_target)

        if token_positions_original is None or token_positions_target is None:
            token_positions_original, token_positions_target = get_token_positions(text_template, subject_original, subject_target)
        
        num_subject_tokens = len(tokenizer.tokenize(subject_original))

        saved_hidden_states = utils.get_hidden_state_vectors(
            tokenizer(subject_target, return_tensors='pt').input_ids,
            hook_type='residual',
        ).hidden_states.detach().cpu()

        with torch.no_grad():
            original_logits = model(tokenizer(text_original, return_tensors='pt').input_ids.to(model.device)).logits.detach().cpu()
        with torch.no_grad():
            target_logits = model(tokenizer(text_target, return_tensors='pt').input_ids.to(model.device)).logits.detach().cpu()

        original_prob_distr = torch.nn.functional.softmax(original_logits, dim=-1)
        target_prob_distr = torch.nn.functional.softmax(target_logits, dim=-1)

        target_probs = []
        original_probs = []
        control_probs = []
        kl_divergences_original = []
        kl_divergences_target = []
        layers = np.arange(utils.num_of_layers)

        for start_layer in tqdm(layers):
            end_layer = start_layer + 1
            layer_range = np.arange(start_layer, end_layer)

            subject_0_hidden_states = saved_hidden_states[layer_range, 0][:, token_positions_target]

            # Choose the substitution method
            if replacement_method == 'exact':
                subject_0_hidden_states = saved_hidden_states[layer_range, 0][:, token_positions_target]
            elif replacement_method == 'mean':
                # saved_hidden_states.shape = (utils.num_of_layers, seq_len, hidden_size)
                # mean_state.shape = (layer_range, hidden_size)
                mean_state = saved_hidden_states[layer_range, 0][:, token_positions_target].mean(dim=1)

                # subject_0_hidden_states.shape = (layer_range, len(token_positions_original), hidden_size)
                subject_0_hidden_states = mean_state.unsqueeze(1).expand(-1, len(token_positions_original), -1)
            elif replacement_method == 'interpolate_or_reduce':
                subject_0_hidden_states = interpolate_or_reduce_fixed_ends(saved_hidden_states[layer_range, 0][:, token_positions_target],
                                                                        len(token_positions_original))#.squeeze(0)

            else:
                raise ValueError(f'Unknown replacement method: {replacement_method}')
            
            zero_out_mask = create_zero_out_mask(torch.concat([
                torch.tensor([utils.num_of_layers]),
                torch.ones(num_subject_tokens, dtype=int) * (layer_range[-1] + 1),
            ]), utils.num_of_layers)
            
            if not zero_out_following:
                zero_out_mask = zero_out_mask * 0
                
            utils.add_hooks(hidden_states=subject_0_hidden_states,
                            layer_n_list=layer_range,
                            token_n=token_positions_original,
                            remove_hooks_on_start=True,
                            hook_type='residual',
                            zero_out_mask=zero_out_mask,
                            argument_logger_do_print=False)

            with torch.no_grad():
                outputs = model(tokenizer(text_original, return_tensors='pt').input_ids.to(model.device))
            logits = outputs.logits

            original_prob = utils.token_prob_in_logits(logits[0, -1], tokenizer.vocab[tokens_to_check.original])
            target_prob = utils.token_prob_in_logits(logits[0, -1], tokenizer.vocab[tokens_to_check.target])
            control_prob = utils.token_prob_in_logits(logits[0, -1], tokenizer.vocab[tokens_to_check.control])

            original_probs.append(original_prob)
            target_probs.append(target_prob)
            
            control_probs.append(control_prob)
            
            modified_prob_distr = torch.nn.functional.softmax(logits, dim=-1)

            kl_div_original = entropy(original_prob_distr[0, -1].float().numpy(), modified_prob_distr[0, -1].detach().cpu().float().numpy())
            kl_div_target = entropy(target_prob_distr[0, -1].float().numpy(), modified_prob_distr[0, -1].detach().cpu().float().numpy())

            kl_divergences_original.append(kl_div_original)
            kl_divergences_target.append(kl_div_target)

            utils.remove_all_forward_hooks()

        
        ax1.plot(layers + 1, original_probs, label=f'{plot_prefix}{tokens_to_check.original} (original)',
                color='blue', linestyle=plot_line_style, linewidth=plot_line_width, 
                marker='o'
            )
        
        ax1.plot(layers + 1, target_probs, label=f'{plot_prefix}{tokens_to_check.target} (target)',
                color='red', linestyle=plot_line_style, linewidth=plot_line_width, 
                marker='o'
            )
        
        ax1.plot(layers + 1, control_probs, label=f'{plot_prefix}{tokens_to_check.control} (control)',
                color='black', linestyle=plot_line_style, linewidth=plot_line_width, 
                marker='o'
            )
        
        ax2.plot(layers + 1, kl_divergences_original, label='KL div w/ original',
                color='blue', linestyle=plot_line_style, linewidth=2, 
                marker='x')

        ax2.plot(layers + 1, kl_divergences_target, label='KL div w/ target',
                color='red', linestyle=plot_line_style, linewidth=2,
                marker='x')

        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Probability')
        ax1.legend()

        
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('KL divergence')
        ax2.legend()
        
        ax1.set_xticks(np.arange(1, utils.num_of_layers + 1, 1))
        ax1.tick_params(axis='both', which='major', labelsize=7, rotation=90)

        ax2.set_xticks(np.arange(1, utils.num_of_layers + 1, 1))
        ax2.tick_params(axis='both', which='major', labelsize=7, rotation=90)
        
        ax1.grid()
        ax2.grid()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # @argument_logger
    def perform_replace_partial_hidden_states(self, text_template, subject_original, subject_target, tokens_to_check,
                                              window_size, hook_type='mlp',
                                              silent=False, 
                                              output_dir=None,
                                              experiment_name=None,
                                              plot_line_style='-',
                                              plot_line_width=1,
                                              plot_prefix='',
                                              ax=None):
        
        replacement_method = self.experiment_setup.replacement_method
        zero_out_following = self.experiment_setup.zero_out_following
        
        assert replacement_method in ['exact', 'mean', 'interpolate_or_reduce']

        utils = self._utils
        model = utils.model
        tokenizer = utils.tokenizer
        get_token_positions = utils.get_token_positions
        create_zero_out_mask = utils.create_zero_out_mask
        interpolate_or_reduce_fixed_ends = utils.interpolate_or_reduce_fixed_ends

        if output_dir is None:
            output_dir = f'experiments/{self._model_name}/window_experiment'

        if ax is None:
            ax = plt.gca()
        
        text_original = text_template.format(subject_original)
        # text_target = text_template.format(subject_target)

        token_positions_original, token_positions_target = get_token_positions(text_template, subject_original, subject_target)

        if replacement_method == 'exact' and len(token_positions_original) != len(token_positions_target):
            raise ValueError(f'Size of the original and target token positions must be the same for exact replacement method. '
                            f'Use mean or interpolate_or_reduce instead.')

        utils.remove_all_forward_hooks()
        
        
        num_subject_tokens = len(tokenizer.tokenize(subject_original))


        ids_original = tokenizer(text_original, return_tensors="pt").input_ids

        if experiment_name is None:
            experiment_name = text_original

        output_dir = os.path.join(output_dir, experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f'Saving to "{os.path.abspath(output_dir)}"')

        # save the hidden states from the **target** subject run
        saved_hidden_states = utils.get_hidden_state_vectors(tokenizer(subject_target, return_tensors='pt').input_ids, hook_type=hook_type)    
        saved_hidden_states = saved_hidden_states.hidden_states

        print('saved_hidden_states.shape:', saved_hidden_states.shape)

        if not silent:
            print('window size =', window_size)

        probs = {token_kind: [] for token_kind in ['original', 'target', 'control']}

        for start_layer in tqdm(np.arange(0, utils.num_of_layers - window_size + 1)):
            end_layer = start_layer + window_size
            layer_range = np.arange(start_layer, end_layer)
            
            utils.remove_all_forward_hooks()

            # Choose the substitution method
            if replacement_method == 'exact':
                subject_0_hidden_states = saved_hidden_states[layer_range, 0][:, token_positions_target]
            elif replacement_method == 'mean':
                # saved_hidden_states.shape = (utils.num_of_layers, seq_len, hidden_size)
                # mean_state.shape = (layer_range, hidden_size)
                mean_state = saved_hidden_states[layer_range, 0][:, token_positions_target].mean(dim=1)

                # subject_0_hidden_states.shape = (layer_range, len(token_positions_original), hidden_size)
                subject_0_hidden_states = mean_state.unsqueeze(1).expand(-1, len(token_positions_original), -1)
            elif replacement_method == 'interpolate_or_reduce':
                subject_0_hidden_states = interpolate_or_reduce_fixed_ends(saved_hidden_states[layer_range, 0][:, token_positions_target],
                                                                        len(token_positions_original))#.squeeze(0)

            else:
                raise ValueError(f'Unknown replacement method: {replacement_method}')
            
            utils.remove_all_forward_hooks()

            zero_out_mask = create_zero_out_mask(torch.concat([
                torch.tensor([utils.num_of_layers]),
                torch.ones(num_subject_tokens, dtype=int) * (layer_range[-1] + 1),
            ]), utils.num_of_layers)
            
            if not zero_out_following:
                zero_out_mask = zero_out_mask * 0
            
            # Now, replace with the corresponding hidden state
            hooks = utils.add_hooks(subject_0_hidden_states.to(model.dtype), layer_range, token_positions_original, hook_type=hook_type,
                                    zero_out_mask=zero_out_mask, argument_logger_do_print=False)

            with torch.no_grad():
                outputs = model(ids_original.to(model.device))

            prob = utils.token_prob_list(outputs, [getattr(tokens_to_check, k) for k in probs.keys()])

            next_token_logits = outputs.logits[0, -1, :]
            top_5_tokens = [utils.vocab_reverse[int(index)] for index in torch.argsort(next_token_logits, descending=True)[:5]]

            for p, token_kind in zip(prob, probs.keys()):
                probs[token_kind].append(p)

            for hook in hooks:
                hook.remove()

        ax.plot(np.arange(len(probs['original'])) + 1, probs['original'],
                label=f'{plot_prefix}{tokens_to_check.original} (original)', 
                color='blue', linestyle=plot_line_style, linewidth=plot_line_width, 
            marker='o'
            )
        
        ax.plot(np.arange(len(probs['target'])) + 1, probs['target'],
                label=f'{plot_prefix}{tokens_to_check.target} (target)',
                color='red', linestyle=plot_line_style, linewidth=plot_line_width, 
            marker='o'
            )
        
        ax.plot(np.arange(len(probs['control'])) + 1, probs['control'],
                label=f'{plot_prefix}{tokens_to_check.control} (control)',
                color='black', linestyle=plot_line_style, linewidth=plot_line_width, 
                marker='o'
            )

        ax.set_xlabel('Layer range')
        ax.set_ylabel('Probability of prediction')

        x_ticks = np.arange(1, utils.num_of_layers + 1, 1)
        x_labels = [f"{i}-{i + window_size - 1}" for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        ax.grid()
        ax.legend()
    
    def compare_with_shift(self, experiment, shift, idx_in_token_positions, zero_out_following=False):
        print(f'{self._model_name}, zero_out_following={zero_out_following}')
        text_target = experiment.text_target
       
        utils = self._utils
        model = utils.model
        tokenizer = utils.tokenizer
        get_token_positions = utils.get_token_positions

        self._utils.remove_all_forward_hooks()

        subject_tokens_positions_original, subject_tokens_positions_target = get_token_positions(experiment.text_template, experiment.subject_original, experiment.subject_target)

        assert subject_tokens_positions_original.shape == subject_tokens_positions_target.shape and torch.all(subject_tokens_positions_original == subject_tokens_positions_target)

        original_input_ids = tokenizer(experiment['text_original'], return_tensors="pt")['input_ids']
        target_input_ids = tokenizer(text_target, return_tensors="pt")['input_ids']


        with torch.no_grad():
            original_outputs = model(original_input_ids.to(model.device), use_cache=False)
            target_outputs = model(target_input_ids.to(model.device), use_cache=False)

        # ====================================
        # Get top tokens and their prompts for OBJECTs

        # First, apply softmax to the last set of logits to obtain probabilities
        prob_top_token_object_original = torch.softmax(original_outputs.logits[:, -1, :], dim=-1)
        prob_top_token_object_target = torch.softmax(target_outputs.logits[:, -1, :], dim=-1)

        # Then, find the index and probability of the most likely token
        vocab_idx_top_token_object_original = prob_top_token_object_original.argmax(-1).item()
        top_prob_object_original = prob_top_token_object_original[0, vocab_idx_top_token_object_original].item()

        vocab_idx_top_token_object_target = prob_top_token_object_target.argmax(-1).item()
        top_prob_object_target = prob_top_token_object_target[0, vocab_idx_top_token_object_target].item()

        # Display the results with probabilities
        print(f'In CLEAN run with ORIGINAL prompt "{tokenizer.tokenize(experiment["text_original"])}",' +
            f' the top token is: "{utils.vocab_reverse[vocab_idx_top_token_object_original]}" ({top_prob_object_original:.2f})')
        print(f'In CLEAN run with TARGET prompt "{tokenizer.tokenize(text_target)}",' + 
            f' the top token is: "{utils.vocab_reverse[vocab_idx_top_token_object_target]}" ({top_prob_object_target:.2f})')

        # ====================================

        print()
        
        max_kl = -1

        min_width = 5
        max_width = 70

        base_width_per_index = 5.5
        fig_width = base_width_per_index * (len(idx_in_token_positions) + 1)
        fig_width = max(min_width, min(fig_width, max_width))

        fig_height = 3

        fig1, axs_prob = plt.subplots(1, len(idx_in_token_positions) + 1, figsize=(fig_width, fig_height))
        fig2, axs_kl = plt.subplots(1, len(idx_in_token_positions) + 1, figsize=(fig_width, fig_height))
        
        if len(idx_in_token_positions) == 0:
            axs_prob = [axs_prob]
            axs_kl = [axs_kl]
            
        _idx_in_token_positions = idx_in_token_positions if idx_in_token_positions else [None]
        for i, idx_token in enumerate(_idx_in_token_positions, start=1):
            # ====================================
            # Now the same for the SUBJECTs

            if idx_in_token_positions:
                prob_top_token_subject_original = torch.softmax(original_outputs.logits[:, subject_tokens_positions_original[idx_token], :], dim=-1)
                prob_top_token_subject_target = torch.softmax(target_outputs.logits[:, subject_tokens_positions_original[idx_token], :], dim=-1)

                # Get the indices and probabilities for the most likely subject token
                vocab_idx_top_token_subject_original = prob_top_token_subject_original.argmax(-1).item()
                top_prob_subject_original = prob_top_token_subject_original[0, vocab_idx_top_token_subject_original].item()

                vocab_idx_top_token_subject_target = prob_top_token_subject_target.argmax(-1).item()
                top_prob_subject_target = prob_top_token_subject_target[0, vocab_idx_top_token_subject_target].item()

                # Display the results with probabilities for the subject
                text_original_tokenized = tokenizer.tokenize(experiment["text_original"])
                text_target_tokenized = tokenizer.tokenize(text_target)

                print(f'In CLEAN run with ORIGINAL prompt "{text_original_tokenized[:idx_token + 1]}",' + 
                    f' the top subject token is: "{utils.vocab_reverse[vocab_idx_top_token_subject_original]}" ({top_prob_subject_original:.2f})')
                print(f'In CLEAN run with TARGET prompt "{text_target_tokenized[:idx_token + 1]}",' + 
                    f' the top subject token is: "{utils.vocab_reverse[vocab_idx_top_token_subject_target]}" ({top_prob_subject_target:.2f})')

            # ====================================

            prob_subject_original_list = []
            prob_subject_target_list = []
            kl_subject_with_original_list = []
            kl_subject_with_target_list = []
            
            
            prob_object_list = []
            kl_object_with_original_list = []
            kl_object_with_target_list = []

            for start in tqdm(range(utils.num_of_layers - shift), total=utils.num_of_layers - shift):
                utils.remove_all_forward_hooks()

                layer_nums_target = torch.arange(start, start + 1)
                layer_nums_original = torch.arange(start + shift, start + shift + 1)

                num_subject_tokens = len(tokenizer.tokenize(experiment['subject_original']))
                num_text_tokens = len(tokenizer.tokenize(experiment['text_original']))
                _zero_out_mask = utils.create_zero_out_mask(torch.concat([
                    torch.tensor([utils.num_of_layers]),
                    torch.ones(num_subject_tokens, dtype=int) * (layer_nums_original + 1),
                ]), utils.num_of_layers)# * 0
                if not zero_out_following:
                    _zero_out_mask = _zero_out_mask * 0
                
                new_outputs, prob_object = self.replace_custom_partial_hidden_state(
                    text_template=experiment.text_template,
                    subject_original=experiment.subject_original,
                    subject_target=experiment.subject_target,
                    tokens_to_check=experiment.tokens_to_check,
                    layer_nums_original=layer_nums_original,
                    layer_nums_target=layer_nums_target,
                    hook_type='residual',
                    zero_out_mask=_zero_out_mask,
                    argument_logger_do_print=False,
                )

                # =========================================================
                # Object
                prob_object_list.append(prob_object)

                kl_object_with_original = kl_divergence(Categorical(logits=original_outputs.logits[:, -1, :]),
                                                        Categorical(logits=new_outputs.logits[:, -1, :])).item()
                kl_object_with_target = kl_divergence(Categorical(logits=target_outputs.logits[:, -1, :]),
                                                    Categorical(logits=new_outputs.logits[:, -1, :])).item()
                kl_object_with_original_list.append(kl_object_with_original)
                kl_object_with_target_list.append(kl_object_with_target)

                # =========================================================
                # Subject
                
                if len(idx_in_token_positions) == 0:
                    continue

                new_subject_probs = new_outputs.logits[0, subject_tokens_positions_original[idx_token], :].softmax(dim=-1).squeeze()  # torch.Size([vocab_size])
                prob_subject_original = new_subject_probs[vocab_idx_top_token_subject_original].item()
                prob_subject_target = new_subject_probs[vocab_idx_top_token_subject_target].item()
                prob_subject_original_list.append(prob_subject_original)
                prob_subject_target_list.append(prob_subject_target)

                kl_subject_with_original = kl_divergence(Categorical(logits=original_outputs.logits[:, subject_tokens_positions_original[idx_token], :]),
                                                        Categorical(logits=new_outputs.logits[:, subject_tokens_positions_original[idx_token], :])).item()
                kl_subject_with_target = kl_divergence(Categorical(logits=target_outputs.logits[:, subject_tokens_positions_original[idx_token], :]),
                                                    Categorical(logits=new_outputs.logits[:, subject_tokens_positions_original[idx_token], :])).item()
                kl_subject_with_original_list.append(kl_subject_with_original)
                kl_subject_with_target_list.append(kl_subject_with_target)

                # =========================================================

            max_kl = max(max_kl, np.concatenate([kl_object_with_original_list, kl_object_with_target_list,
                                                kl_subject_with_original_list, kl_subject_with_target_list]).max())
            tick_labels = [f"{i+1}$\\to${i+1+shift}" for i in range(utils.num_of_layers - shift)]

            
            if len(idx_in_token_positions) == 0:
                continue
                
            # plot the prob/kl for the SUBJECT token
            axs_prob[i].plot(np.array(prob_subject_original_list), 'b', marker='x', label=f'$p({utils.vocab_reverse[vocab_idx_top_token_subject_original]})$')
            axs_prob[i].plot(np.array(prob_subject_target_list), 'r', marker='x', label=f'$p({utils.vocab_reverse[vocab_idx_top_token_subject_target]})$')
            axs_prob[i].set_title(f'{text_target_tokenized[idx_token]} $\\to$ {text_original_tokenized[idx_token]}')
            axs_prob[i].set_xticks(range(len(tick_labels)), tick_labels, rotation=90)
            axs_prob[i].set_ylim(0, 1)
            axs_prob[i].grid(True)
            axs_prob[i].legend()
            
            # plot the prob/kl for the SUBJECT token
            axs_kl[i].plot(kl_subject_with_original_list, 'b', marker='x', label='$KL(p(s_i)~||~p(s^*_i))$')
            axs_kl[i].plot(kl_subject_with_target_list, 'r', marker='x', label='$KL(p(\hat{s}_i)~||~p(s^*_i) )$')
            axs_kl[i].set_title(f'')
            axs_kl[i].set_xticks(range(len(tick_labels)), tick_labels, rotation=90)
            axs_kl[i].grid(True)
            axs_kl[i].legend()
            
        # plot the prob/kl for the object (last) token
        axs_prob[0].plot(np.array([p['original'] for p in prob_object_list]), 'b', marker='x', label=f'$p({utils.vocab_reverse[vocab_idx_top_token_object_original]})$')
        axs_prob[0].plot(np.array([p['target'] for p in prob_object_list]), 'r', marker='x', label=f'$p({utils.vocab_reverse[vocab_idx_top_token_object_target]})$')
        axs_prob[0].set_title(f'Prob of object token')
        axs_prob[0].set_xticks(range(len(tick_labels)), tick_labels, rotation=90)
        axs_prob[0].set_ylim(0, 1)
        axs_prob[0].grid(True)

        # plot the prob/kl for the object (last) token
        axs_kl[0].plot(kl_object_with_original_list, 'b', marker='x', label='$KL(p(o)~||~p(o^*))$')
        axs_kl[0].plot(kl_object_with_target_list, 'r', marker='x', label='$KL(p(\hat{o})~||~p(o^*))$')
        axs_kl[0].set_title(f'KL of object token with orig/target run')
        axs_kl[0].set_xticks(range(len(tick_labels)), tick_labels, rotation=90)
        
        
        axs_kl[0].grid(True)
        
        for i in range(len(axs_kl)):
            axs_kl[i].set_ylim(0, max_kl)
            
            axs_kl[i].set_ylabel('Probability')
            axs_kl[i].set_xlabel('Layers')
            axs_kl[i].legend(loc='upper right')
            
            axs_prob[i].set_ylabel('KL divergence')
            axs_prob[i].set_xlabel('Layers')
            axs_prob[i].legend(loc='upper right')
        
        
        plt.show()

        utils.remove_all_forward_hooks()
        torch.cuda.empty_cache()

        return (fig1, axs_prob), (fig2, axs_kl)
