# ========== Base Model ==========

from collections import defaultdict
import math
from typing import Tuple
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from tqdm import tqdm
from transformers import AutoConfig, AutoModel
from metrics.metrics import calc_metrics, get_true_prediction_with_certainty
from model.util import TensorDict, _repeat, configure_optimizers, margin_loss, repeat_context_emb
from utils.util import EasyDict

class FGNETBaseModel(pl.LightningModule):

    def __init__(
        self,
        optimizer_params,
        scheduler_params=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, logging=True)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        results = self._epoch_end(outputs, logging=False)
        for metric, value in results.items():
            print(f"{metric}: \t{value}")
        return results

class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.divisor = nn.Parameter(torch.tensor(math.sqrt(self.hidden_dim)), requires_grad=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, value_attention_mask=None, score=None, return_score=False) -> Tuple[torch.Tensor, torch.Tensor]:
        # query=(B,M,D), key=value=(B,N,D), attention_mask=(B,N)
        if score is None:
            score = query @ key.transpose(1,2) # (B,M,N)
        original_score = score.clone()
        if value_attention_mask is not None:
            _attention_mask = value_attention_mask.unsqueeze(dim=1)
            score += (_attention_mask-1) * 10000  
        score /= self.divisor
        attn = torch.softmax(score, dim=-1) # (B,M,N)
        context = attn @ value # (B,M,N) @ (B,N,D) = (B,M,D) -> query.shape
        if return_score:
            return context, attn, original_score
        return context, attn

class ClassifierHead(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_prob=0.1, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim),
        )
        self.classifier[-1].bias.data.zero_()
    
    def get_output_bias(self):
        return self.classifier[-1].bias
    
    def forward(self, input):
        return self.classifier(input)

class CrossAttentionClassifierHead(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_classes,
            dropout_prob=0.1,
            cross_attention=True,
            self_attention=True,
            use_label_mean=False,
        ):
        super().__init__()
        self.attention = DotProductAttention(hidden_size)
        self.classifier = ClassifierHead(
            hidden_size*4,
            num_classes,
            dropout_prob=dropout_prob,
        )
        self.cross_attention = cross_attention
        self.self_attention = self_attention
        self.trainable_reps = not self.cross_attention and not self.self_attention
        if self.trainable_reps:
            # If cross attention and self attention is deactivated, use trainable representations
            print("Using trainable cross attention embeddings")
            self.mention_c = nn.Parameter(torch.randn(hidden_size))
            self.label_rep_c = nn.Parameter(torch.randn(hidden_size))

        self.use_label_mean = use_label_mean

    def forward(self, emb1, emb2):
        context_emb = emb1["last_hidden_state"]
        label_emb = emb2["last_hidden_state"]

        emb1_token_pos = emb1.pop("token_pos", 0)
        mention_emb = context_emb[torch.arange(len(context_emb)),emb1_token_pos,:]
        a_c, a_l = emb1["attention_mask"], emb2["attention_mask"]

        if self.trainable_reps:
            emb2_token_pos = emb2.pop("token_pos", 0)
            label_rep_emb = label_emb[torch.arange(len(label_emb)),emb2_token_pos,:]
            mention_c = self.mention_c[None,:].repeat(len(label_emb),1)
            label_rep_c = self.label_rep_c[None,:].repeat(len(label_emb),1)
            h = torch.cat((mention_emb, mention_c, label_rep_emb, label_rep_c), dim=-1)
            output = self.classifier(h)
            return output

        # Cross Attention
        context_c, _, score = self.attention(context_emb, label_emb, label_emb, a_l, return_score=True)
        label_c, _ = self.attention(label_emb, context_emb, context_emb, a_c, score=score.transpose(1,2))
        
        # Self Attention
        mention_c = context_c[torch.arange(len(context_c)),emb1_token_pos,:]
        mention_c = mention_c.unsqueeze(1)
        mention_c, _ = self.attention(mention_c, context_c, context_c, a_c)
        mention_c = mention_c.squeeze(1)

        if self.use_label_mean:
            label_mask =  emb2["_attention_mask"] if "_attention_mask" in emb2 else a_l 

            label_rep_emb = label_emb * label_mask[:,:,None]
            label_rep_emb = label_rep_emb[:,1:]
            label_rep_emb = label_rep_emb.sum(dim=1) / (torch.norm(label_rep_emb, dim=1) + 1e-6)

            label_rep_c = label_c * label_mask[:,:,None]
            label_rep_c = label_rep_c[:,1:]
            label_rep_c = label_rep_c.sum(dim=1) / (torch.norm(label_rep_c, dim=1) + 1e-6)
        else:
            emb2_token_pos = emb2.pop("token_pos", 0)
            label_rep_emb = label_emb[torch.arange(len(label_emb)),emb2_token_pos,:]

            label_rep_c = label_c[torch.arange(len(label_c)),emb2_token_pos,:]
            label_rep_c = label_rep_c.unsqueeze(1)
            label_rep_c, _ = self.attention(label_rep_c, label_c, label_c, a_l)
            label_rep_c = label_rep_c.squeeze(1)

        # Concat with embedding of original CLS/MENTION token
        h = torch.cat((mention_emb, mention_c, label_rep_emb, label_rep_c), dim=-1)

        # Classifying
        output = self.classifier(h)
        return output


class SimHead(nn.Module):
    def forward(self, context_emb, label_emb):
        context_rep = context_emb["last_hidden_state"][:,0]
        label_rep = label_emb["last_hidden_state"][:,0]
        output = nn.functional.cosine_similarity(context_rep, label_rep)
        return output


class FGNETSeparatePredictor(FGNETBaseModel):
    def __init__(
            self,
            model_name,
            label_tokens_loader=None,
            eval_batch_size=None,
            num_tokens=None,
            head_cross_attention=True,
            head_self_attention=True,
            use_label_mean=False,
            head_dropout_prob=0.2,
            sim_loss_margin=None,
            threshold_start=None,
            threshold_step=None,
            freeze_encoder=False,
            label_fallback=None,
            **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["label_tokens_loader", "eval_batch_size"])

        self.encoder = AutoModel.from_pretrained(model_name)
        if num_tokens:
            self.encoder.resize_token_embeddings(num_tokens)
        self.config = AutoConfig.from_pretrained(model_name)

        self.sim_loss_margin = sim_loss_margin
        if self.sim_loss_margin:
            self.classifier = SimHead()
        else:
            cross_attention_kwargs = dict(
                hidden_size=self.config.hidden_size,
                num_classes=1,
                cross_attention=head_cross_attention,
                self_attention=head_self_attention,
                dropout_prob=head_dropout_prob,
                use_label_mean=use_label_mean,
            )
            self.classifier = CrossAttentionClassifierHead(**cross_attention_kwargs)
            self.loss_func = nn.BCEWithLogitsLoss()

        self.label_tokens_loader = label_tokens_loader
        self.eval_batch_size = eval_batch_size
        self.threshold_start = threshold_start
        self.threshold_step = threshold_step
        self.label_fallback = label_fallback

        if freeze_encoder:
            print("Freeze encoder")
            for p in self.encoder.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        return configure_optimizers(self.named_parameters(), self.optimizer_params, self.scheduler_params)
    
    def calc_loss(self, y_hat, y):
        if self.sim_loss_margin:
            return margin_loss(y_hat, y, self.sim_loss_margin)
        
        y_hat = y_hat.squeeze()
        y = y.float()
        return self.loss_func(y_hat, y)

    def forward(self, context, label, repeat_context_emb_if_needed=True):
        context_emb = self._encode_batch(context)
        label_emb = self._encode_batch(label)

        if repeat_context_emb_if_needed:
            context_emb = repeat_context_emb(context_emb, label_emb)

        output = self.classifier(context_emb, label_emb)
        return EasyDict(logits=output)

    def _encode_batch(self, batch):
        add_kwargs = {}
        for k in ["token_pos", "labels", "_attention_mask"]:
            if k in batch:
                add_kwargs[k] = batch.pop(k)
        
        output = self.encoder(**batch).last_hidden_state
        result = dict(last_hidden_state=output, attention_mask=batch["attention_mask"])
        return TensorDict(**result, **add_kwargs)
    
    def _encode_loader(self, loader):
        # Note: results are a automatically moved to cpu
        result = defaultdict(list)
        for input in loader:
            input = {k: v.to(self.device) for k, v in input.items()}
            with torch.no_grad():
                batch_result = self._encode_batch(input)
            for k, v in batch_result.items():
                result[k].append(v.cpu())
        result = {k: torch.cat(v, dim=0) for k, v in result.items()}
        return TensorDict(result)

    def training_step(self, batch, batch_idx):
        def do_pass(batch, prefix=""):
            if f"{prefix}labels" not in batch:
                return False
            labels = batch.pop(f"{prefix}labels")
            context_tokens = batch.pop(f"{prefix}context_tokens")
            label_tokens = batch.pop(f"{prefix}label_tokens")

            output = self(context_tokens, label_tokens).logits
            loss = self.calc_loss(output, labels)
            num_pos = labels.sum().item()
            num_neg = len(labels) - num_pos
            return loss, num_pos, num_neg

        loss, num_pos, num_neg = do_pass(batch)
        to_log = {"train/loss": loss.clone(), "train/pos": float(num_pos), "train/neg": float(num_neg)}
        if hasattr(self.classifier, "attention"):
            to_log["train/divisor"] = self.classifier.attention.divisor
        
        if self.trainer.is_global_zero:
            self.log_dict(to_log, rank_zero_only=True)
        return EasyDict(loss=loss)

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        input = dict(**batch, labels=labels)
        return self._encode_batch(input).to("cpu")

    def _apply_head(self, context_embeddings, label_embeddings):
        context_embeddings, label_embeddings = TensorDict(context_embeddings), TensorDict(label_embeddings)
        total_context_embeddings, total_label_embeddings = context_embeddings.tensor_size(), label_embeddings.tensor_size()
        
        label_batch_size = min(self.eval_batch_size, total_label_embeddings)
        context_batch_size = self.eval_batch_size // label_batch_size
        
        context_embeddings.to(self.device)
        label_embeddings.to(self.device)
        results = []
        for i_context in tqdm(range(0, total_context_embeddings, context_batch_size)):
            e_context = i_context + context_batch_size
            for i_label in range(0, total_label_embeddings, label_batch_size):
                e_label = i_label + label_batch_size

                contexts = context_embeddings.range(i_context, e_context)
                act_num_contexts = contexts.tensor_size()
                labels = label_embeddings.range(i_label, e_label)
                act_num_labels = labels.tensor_size()

                _contexts = TensorDict({k: _repeat(v, act_num_labels, 1) for k, v in contexts.items()})
                _labels = TensorDict({k: _repeat(v, act_num_contexts, 0) for k, v in labels.items()})
                # _contexts.to(self.device)
                # _labels.to(self.device)

                with torch.no_grad():
                    y = self.classifier(_contexts, _labels)
                results.append(y)

        results = torch.cat(results, dim=0).cpu()
        return results

    @staticmethod
    def _predict_from_probs(y_hat, y, threshold=0.5, fallback=None):
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y.float())
        result_metrics = dict(
            loss=loss.item(),
            **calc_metrics(y_hat, y, threshold, fallback=fallback),
        )
        true_prediction, certainty = get_true_prediction_with_certainty(y_hat, y, threshold, fallback=fallback)
        return EasyDict(
            predictions=true_prediction,
            certainty=certainty,
            metrics=result_metrics,
        )
        
    def _calc_probs(self, context_loader, label_loader):
        context_emb = self._encode_loader(context_loader)
        y = context_emb.pop("labels")
        label_emb = self._encode_loader(label_loader)
        results = self._apply_head(context_emb, label_emb)
        y_hat = torch.sigmoid(results)
        y_hat = y_hat.view(-1, label_emb.tensor_size())
        return y_hat, y

    def predict(self, context_loader, label_loader, threshold=0.5, fallback=None):
        y_hat, y = self._calc_probs(context_loader, label_loader)

        return EasyDict(
            **self._predict_from_probs(y_hat, y, threshold, fallback),
            y_hat=y_hat,
            y=y,
        )

    def best_threshold_result_metrics(self, y_hat, y):
        if not self.threshold_step:
            return None
         
        result_metrics, best_individual_scores, best_thresholds = {}, {}, {}
        threshold_start = self.threshold_start or self.threshold_step
        for _threshold in tqdm(np.arange(threshold_start, 1+self.threshold_step, self.threshold_step)):
            best_metrics = calc_metrics(y_hat, y, _threshold, metrics=["choi", "shot"], fallback=self.label_fallback)
            best_metrics = {k: v for k, v in best_metrics.items() if "f1" in k}
            if not best_individual_scores:
                best_individual_scores = best_metrics
                best_thresholds = {k: _threshold for k in best_metrics}
            else:
                for k, v in best_metrics.items():
                    if v > best_individual_scores[k]:
                        best_individual_scores[k] = v
                        best_thresholds[k] = _threshold
        
        max_threshold = best_thresholds["choi/f1-macro"]
        best_metrics = calc_metrics(y_hat, y, max_threshold, metrics=["choi", "shot"], fallback=self.label_fallback)
        best_metrics["threshold"] = max_threshold
        best_metrics = {f"best/{k}": v for k, v in best_metrics.items()}
        result_metrics.update(best_metrics)

        best_individual_scores = {f"best-ind/{k}": v for k, v in best_individual_scores.items()}
        result_metrics.update(best_individual_scores)
        best_thresholds = {f"best-ind/{k}-threshold": v for k, v in best_thresholds.items()}
        result_metrics.update(best_thresholds)
        return result_metrics

    def _epoch_end(self, outputs, logging=False):
        context_emb = {k: [v[k] for v in outputs] for k in outputs[0]}
        context_emb = {k: torch.cat(v, dim=0) for k, v in context_emb.items()}
        y = context_emb.pop("labels")

        label_emb = self._encode_loader(self.label_tokens_loader)
        results = self._apply_head(context_emb, label_emb)
        results = results.view(-1, label_emb.tensor_size())
        
        if self.sim_loss_margin:
            y_hat = results
            _y_hat = (y_hat + 1.)/2.
        else:
            y_hat = torch.sigmoid(results)
            _y_hat = y_hat
        
        loss = torch.nn.functional.binary_cross_entropy(_y_hat, y.float())
        result_metrics = dict(
            loss=loss.item(),
            **calc_metrics(_y_hat, y, metrics=["choi"], fallback=self.label_fallback),
        )

        best_threshold_metrics = self.best_threshold_result_metrics(y_hat, y)
        if best_threshold_metrics:
            result_metrics.update(best_threshold_metrics)

        if logging and self.trainer.is_global_zero:
            log_metrics = {f"val/{k}": v for k, v in result_metrics.items()}
            log_metrics["val_choi_f1"] = log_metrics["val/choi/f1-macro"]
            log_metrics["val_loss"] = log_metrics["val/loss"]
            log_metrics["val_best_choi_f1"] = log_metrics.get("val/best/choi/f1-macro", 0.0)

            self.log_dict(log_metrics, rank_zero_only=True)
        
        return result_metrics
