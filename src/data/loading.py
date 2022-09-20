import random
from typing import Dict, List
import datasets
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


MENTION_TOKEN_OPEN = "[MENTION/]"
MENTION_TOKEN = "[MENTION]"
MENTION_TOKEN_CLOSED = "[/MENTION]"

SPECIAL_TOKENS = [MENTION_TOKEN_OPEN, MENTION_TOKEN, MENTION_TOKEN_CLOSED]

def load_labels(file_path):
    with open(file_path) as f:
        return f.read().splitlines()

def build_enclosed_item(item):
    return f"{item['left_context_text']} {MENTION_TOKEN_OPEN} {item['word']} {MENTION_TOKEN_CLOSED} {item['right_context_text']}".strip()

def build_enclosed_item_batched(x):
    def build_enclosed_item(i):
        return f"{x['left_context_text'][i]} {MENTION_TOKEN_OPEN} {x['word'][i]} {MENTION_TOKEN_CLOSED} {x['right_context_text'][i]}"
    k = list(x.keys())[0]
    n = len(x[k])
    result = [build_enclosed_item(i) for i in range(n)]
    return result

def get_element(input):
    return input[0]

def find(l, value, default=-1):
    try:
        return l.index(value)
    except ValueError:
        return default


class FGNETBaseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_files: Dict,
            label_path,
            train_batch_size,
            eval_batch_size,
            num_workers,
            tokenizer,
            train_label_path=None,
            max_length=None,
            subset=None,
            label_max_length=32,
            use_enclosing_token_pos=True,
            is_distributed=False,
            **kwargs,
        ):
        super().__init__()
        
        assert all(k in ["train", "val", "test"] for k in data_files.keys())
        self.data_files = data_files

        self.labels = load_labels(label_path)
        print("#Labels:", len(self.labels))
        self.label_map = {l: i for i, l in enumerate(self.labels)}
        if train_label_path:
            self.train_labels = load_labels(train_label_path)
            # self.train_label_map = {l: i for i, l in enumerate(self.train_labels)}
        else:
            self.train_labels = self.labels
            # self.train_label_map = self.label_map
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_max_length = label_max_length

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        self.num_workers = num_workers
        self.subset = subset
        self.use_enclosing_token_pos = use_enclosing_token_pos
        self.is_distributed = is_distributed

    def setup(self, stage):
        self.dataset = {
            k: datasets.load_dataset("json", data_files={k: data_file}, split=k)
            for k, data_file in self.data_files.items()
        }
        print(self.dataset)

        if self.subset:
            print(f"Use subset: {self.subset}")
            for k, v in self.dataset.items():
                self.dataset[k] = v.select(range(self.subset))

    def _create_distributed(self, sampler):
        if not self.is_distributed:
            return sampler
        return DistributedSampler(sampler, shuffle=False)

    def train_dataloader(self):
        assert hasattr(self, "train_data")
        sampler = BatchSampler(
                RandomSampler(self.train_data),
                batch_size=self.train_batch_size,
                drop_last=True)
        sampler = self._create_distributed(sampler)
        return DataLoader(
            self.train_data,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=get_element,
            pin_memory=True,
        )
  
    def val_dataloader(self):
        assert hasattr(self, "val_data")
        return DataLoader(self.val_data, batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=True)
  
    def test_dataloader(self):
        assert hasattr(self, "test_data")
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=True)


class FGNETSeparateDataModule(FGNETBaseDataModule):

    def __init__(
            self,
            neg_batch_factor=1,
            **kwargs):
        super().__init__(**kwargs)
        self.neg_batch_factor = neg_batch_factor
    
    def _get_token_pos(self, x, id, default=0):
        if isinstance(x, torch.Tensor):
            token_pos = [find(t.tolist(), id, default) for t in x]
            return torch.tensor(token_pos)
        return [find(t, id, default) for t in x]

    def train_transform(self, batch):
        def _tokenize(x, max_length):
            return self.tokenizer(x, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

        def get_random_label_not(ys: List):
            labels = self.train_labels
            while True:
                x = np.random.choice(labels)
                if x not in ys:
                    return x

        pos_labels_text = [random.choice(y) for y in batch["y_category"]]
        neg_labels_text = [get_random_label_not(ys) for ys in batch["y_category"]*self.neg_batch_factor]
        pos_labels = torch.ones(len(pos_labels_text))
        neg_labels = torch.zeros(len(neg_labels_text))
        labels = torch.cat([pos_labels, neg_labels]).to(int)
        label_text = pos_labels_text + neg_labels_text
        label_tokens = _tokenize(label_text, self.label_max_length)

        context_tokens = _tokenize(batch["item"], self.max_length)
        if self.use_enclosing_token_pos:
            mention_open_id = self.tokenizer.convert_tokens_to_ids(MENTION_TOKEN_OPEN)
            context_tokens["token_pos"] = self._get_token_pos(context_tokens["input_ids"], mention_open_id, 0)

        return dict(
            context_tokens=context_tokens,
            label_tokens=label_tokens,
            labels=labels,
        )

    def prepare_eval_data(self, dataset, tokenizer):
        def _tokenize(x):
            return tokenizer(x["item"], truncation=True, padding=True, max_length=self.max_length)
        def one_hot(x):
            labels = [0] * len(self.labels)
            for c in x["y_category"]:
                i = self.label_map[c]
                labels[i] = 1
            return dict(labels=labels)
        
        dataset = dataset.map(_tokenize, batched=True, batch_size=len(dataset["item"]), desc="Tokenize...")
        dataset = dataset.map(one_hot, desc="One hot...")
        columns = [*tokenizer.model_input_names, "labels"]
        if self.use_enclosing_token_pos:
            mention_open_id = tokenizer.convert_tokens_to_ids(MENTION_TOKEN_OPEN)
            token_pos = self._get_token_pos(dataset["input_ids"], mention_open_id, 0)
            dataset = dataset.add_column("token_pos", token_pos)
            columns.append("token_pos")

        dataset.set_format(type="torch", columns=columns)
        return dataset

    def prepare_label_data(self, tokenizer):
        def _tokenize(x):
            return tokenizer(x, truncation=True, padding=True, max_length=self.label_max_length)
        labels = self.labels
        label_tokens = _tokenize(labels)
        
        print("Label token shape:", torch.tensor(label_tokens["input_ids"]).shape)
        dataset = datasets.Dataset.from_dict(label_tokens)
        dataset.set_format(type="torch")
        return dataset

    def setup(self, stage=None):
        if stage:
            print(f"Skip data module 'setup' during '{stage}'")
            return
        super().setup(stage)

        self.dataset = {k: d.map(lambda x: dict(item=build_enclosed_item_batched(x)), batched=True) for k, d in self.dataset.items()}
        
        if "train" in self.dataset:
            self.train_data = self.dataset["train"]
            self.train_data.set_transform(self.train_transform)
        
        tokenizer = self.tokenizer
        if "val" in self.dataset:
            print("Prepare validation data")
            self.val_data = self.dataset["val"]
            self.val_data = self.prepare_eval_data(self.val_data, tokenizer)

        if "test" in self.dataset:
            print("Prepare test data")
            self.test_data = self.dataset["test"]
            self.test_data = self.prepare_eval_data(self.test_data, tokenizer)

        print("Prepare label data")
        self.label_dataset = self.prepare_label_data(tokenizer)

    def label_dataloader(self):
        return DataLoader(self.label_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=True)
