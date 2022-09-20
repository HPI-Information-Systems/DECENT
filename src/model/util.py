import torch
import transformers

class TensorDict(dict):
    def to(self, device):
        for k, v in self.items():
            self[k] = v.to(device)
        return self

    def range(self, fr, to):
        return TensorDict({k: v[fr:to] for k, v in self.items()})

    def tensor_size(self):
        k = list(self.keys())[0]
        return self[k].size(0)


def _repeat(t, n_repeats, dim):
    # dim=1: xxxyyy, dim=0: xyxyxy
    shape = t.shape
    t = torch.repeat_interleave(t.unsqueeze(dim), n_repeats, dim)
    return t.reshape(-1, *shape[1:])


def repeat_context_emb(context_emb, label_emb):
    context_batch_size = context_emb["last_hidden_state"].size(0)
    label_batch_size = label_emb["last_hidden_state"].size(0)
    assert label_batch_size % context_batch_size == 0, f"Label batch size ({label_batch_size}) has to be a multiple of context batch size ({context_batch_size}) to allow repeat"
    num_repeats = label_batch_size // context_batch_size
    if num_repeats == 1:
        return context_emb
    keys_to_repeat = ["last_hidden_state", "attention_mask", "token_pos", "query_key_value"]
    new_context_emb = {k: _repeat(context_emb[k], num_repeats, 0) for k in context_emb.keys() if k in keys_to_repeat}
    context_emb.update(new_context_emb)
    return context_emb


def margin_loss(y_hat, y, margin):
    pos = y_hat[y == 1]
    neg = y_hat[y == 0]
    n_repeats = len(neg) // len(pos)
    pos = pos.repeat(n_repeats)
    l = torch.max(neg - pos + margin, torch.zeros_like(pos))
    return torch.mean(l)


def setup_scheduler(optimizer, scheduler_type, **kwargs):
    lrs = [g["lr"] for g in optimizer.param_groups]
    if scheduler_type == "cyclic":
        base_multiplier = kwargs.pop("base_multiplier", 0.5)        
        base_lrs = [base_multiplier * lr for lr in lrs]
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            max_lr=lrs,
            base_lr=base_lrs,
            **kwargs,
        )
        scheduler.scale_mode = "cycle"
        return scheduler
    elif scheduler_type == "linear_warmup":
        return transformers.get_linear_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == "cosine_warmup":
        return transformers.get_cosine_schedule_with_warmup(optimizer, **kwargs)
    elif scheduler_type == "constant_warmup":
        kwargs.pop("num_training_steps")
        return transformers.get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        NotImplementedError(f"Scheduler type {scheduler_type} is not known.")


def configure_optimizers(named_parameters, optimizer_params, scheduler_params=None):
    def get_optimizer_grouped_parameters(_named_params, _optimizer_params, no_decay=["bias", "LayerNorm", "layer_norm"]):
        if len(_named_params) == 0:
            return []
        lr = float(_optimizer_params["lr"])
        _optimizer_grouped_parameters = [
            dict(params=[p for n, p in _named_params if not any(nd in n for nd in no_decay)], lr=lr, weight_decay=_optimizer_params["weight_decay"]),
            dict(params=[p for n, p in _named_params if any(nd in n for nd in no_decay)], lr=lr, weight_decay=0.0),
        ]
        return _optimizer_grouped_parameters
    
    print("Using optimizers for specific model parts:", optimizer_params)
    all_named_params = dict(named_parameters)
    is_param_specific = any(isinstance(v, dict)  for v in optimizer_params.values())
    if is_param_specific:
        if "__REST" in optimizer_params:
            rest_optimizer_params = optimizer_params["__REST"]
        
        optimizer_grouped_parameters = []
        for name, _optimizer_params in optimizer_params.items():
            if name == "__REST":
                continue
            assert isinstance(_optimizer_params, dict)

            named_params = [(n, p) for n, p in all_named_params.items() if n.startswith(name)]
            all_named_params = {n: p for n, p in all_named_params.items() if not n.startswith(name)}
            print(f"Params with '{name}' ({len(named_params)}) receive:", _optimizer_params)
            _optimizer_grouped_parameters = get_optimizer_grouped_parameters(named_params, _optimizer_params)
            optimizer_grouped_parameters.extend(_optimizer_grouped_parameters)
        
        if "__REST" in optimizer_params:
            print(f"Params with '__REST' ({len(all_named_params)}) receive:", rest_optimizer_params)
            _optimizer_grouped_parameters = get_optimizer_grouped_parameters(list(all_named_params.items()), rest_optimizer_params)
            optimizer_grouped_parameters.extend(_optimizer_grouped_parameters)
        else:
            assert len(all_named_params) == 0, f"There are named params remaining that didn't receive optimizer params: {len(all_named_params)}"
        
    else:
        _optimizer_grouped_parameters = get_optimizer_grouped_parameters(list(all_named_params.items()), optimizer_params)
        optimizer_grouped_parameters.extend(_optimizer_grouped_parameters)

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)
    if scheduler_params:
        scheduler = setup_scheduler(optimizer=optimizer, **scheduler_params)
        return [optimizer], [dict(scheduler=scheduler, interval="step")]
    return optimizer
