# DECENT

**Decoupled Encoding and Cross-Attention for Efficient Named Entity Typing**

## Environment

### Requirements

The code has been tested with **Python 3.9.2** and the following [requirements](requirements.txt).


```bash
$ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

Given our GPU, we need CUDA 11.3 which is indexed under https://download.pytorch.org/whl/cu113. Depending on your specification you may not have to rely on this CUDA version and use a standard version of Pytorch. However, under these circumstances we cannot guarantee a successful environment setup.

### Dotenv

To override certain environment variables, copy `.env_template` to `.env` and adapt as needed, e.g. wandb key.

## Data

We use the data and the respective format provided by [Onoe et al.](https://github.com/yasumasaonoe/Box4Types)

It can be downloaded [here](https://drive.google.com/file/d/1T9kbqNS2UN84Z40y3j6zjFooNu-IuaSN/view?usp=sharing).


### Distantly Supervised Data
From [Choi et al.](https://aclanthology.org/P18-1009.pdf):
https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html but only needed when pretraining.

Use the following script for formatting:

```bash
$ python scripts/format.py --file el_train.json --output el_train_processed.json
```

## Checkpoints

### Models

You can [download](https://drive.google.com/file/d/1mieu2HgUJLZAtrMsgRdYNzOQJJdKrJoc/view?usp=sharing) the model checkpoints of DECENT trained on UFET, OntoNotes or FIGER.

Model Ids:
- UFET: `otkl66o1`
- FIGER: `1965bsiq`
- OntoNotes: `1oxzic0i`

### Output

- DECENT: [link](https://drive.google.com/file/d/1-CM59LYWxLw4puAVXjV4ok896G7y1XQI/view?usp=sharing) (contains model output + best prediction of UFET, OntoNotes and FIGER for dev and test)
- MLMET/Lite: [link](https://drive.google.com/file/d/1AryPlLB3ltk_bEklwp8hq_ybQ2jMc4QR/view?usp=sharing) (contains model output + best predictions of UFET for dev and test)

Overview of best [Loose Macro-F1](https://aiweb.cs.washington.edu/ai/pubs/ling-aaai12.pdf) scores for the different training configurations.
The respective threshold was identified using the validation dataset and is shown in parentheses.

| Model             | UFET (Dev)    | UFET (Test)   | FIGER (Dev)   | FIGER (Test)  | OntoNotes (Dev)   | OntoNotes (Test)  |
| ---               | ---           | ---           | ---           | ---           | ---               | ---               |
| DECENT (UFET)     | 50.84 (0.985) | 49.74 (0.985) | 60.37 (0.97)  | 69.12 (0.97)  | 82.41 (0.955)     | 83.10 (0.955)     |
| MLMET             | 49.06 (0.500) | 49.08 (0.5)   | -             | -             | -                 | -                 |         
| Lite              | 50.44 (0.93)  | 50.61 (0.93)  | -             | -             | -                 | -                 |  
| DECENT (FIGER)    | -             | -             | 90.87 (0.96)  | 83.81 (0.96)  | -                 | -                 |
| DECENT (OntoNotes)| -             | -             | -             | -             | 76.86 (0.99)      | 77.01 (0.99)      |

## Training

To train a model:
```bash
$ python src/train.py --config config/train.yaml
```

Overview of Training Configurations: 
| Config | Description |
| ---  | ---         |
| [`train.yaml`](config/train.yaml) | Training DECENT on UFET |
| [`figer.yaml`](config/figer.yaml) | Training DECENT on OntoNotes |
| [`onto.yaml`](config/onto.yaml) | Training DECENT on OntoNotes |
| [`pretrain.yaml`](config/pretrain.yaml) | Pretraining DECENT on distantly supervised UFET |
| [`fine_tune.yaml`](config/fine_tune.yaml) | Fine-tuning the pretrained model on UFET |
| [`unist.yaml`](config/unist.yaml) | Training configuration of [UniST](https://aclanthology.org/2022.naacl-main.190.pdf) on UFET |

Important Flags:

| Flag | Description |
| ---  | ---         |
| `--wandb.offline` | `True` to turn of wandb; default: `False`         |
| `--result-dir` | Result directory |

The full list of available parameters and there default values can be viewed in the respective [file](src/utils/config.py).

The parameters are given the following priority (highest to lowest):

1. Command line arguments, e.g. `--model.optimizer_params.classifier.lr 0.005`
2. [Configuration file](config)
3. [Default values](src/utils/config.py)


## Evaluation

### 1. Get Model Output

Use this script to get the prediction and output of a model for specific dataset.

Please refer to the documentation in the respective [file](src/predict.py#L171).

Example:
```bash
$ python src/predict.py predict --checkpoint BEST_MODEL.ckpt --dataset data/ufet/ufet_dev.json --labels data/ontology/ufet_types.txt --output OUTPUT_FOLDER --save-model-output CACHE_FOLDER --model-id 123 --batch-size 128
```

Note: `--save-model-output` needs to be defined to [find the best threshold](#2-best-threshold). 

### 2. Best Threshold

Use this script to find the best threshold for a model given its output.

Please refer to the documentation in the respective [file](src/predict.py#L120).

Example:
```bash
$ python src/predict.py reuse --cache MODEL_OUTPUT_CACHE.pkl --output OUTPUT_FOLDER --threshold-step 0.005
```

### 3. Additional Metrics 

Use this script to get additional metrics regarding granularity and type of mention.

Please refer to the documentation in the respective [file](src/eval.py#L29).

Example:
```bash
$ python eval.py --results PREDICTIONS.json --labels data/ontology/ufet_types.txt
```

## Miscellaneous

### UniST

We reimplemented parts of [UniST](https://aclanthology.org/2022.naacl-main.190.pdf) for our evaluation.
For more details view the paper by Huang et al.
You can train a model with our framework using the [`unist.yaml`](config/unist.yaml) training configuration.
The prediction and evaluation is the same as with DECENT.

### OntoNotes & FIGER

To validate our approach, we used the fine-grained datasets for OntoNotes and FIGER.
Training and prediction is the same as for UFET.
We provide results and predictions for a model that has been trained and the respective dataset and the model that has solely been trained on UFET.
