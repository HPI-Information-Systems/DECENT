import csv
import json
import os
import click
import torch
import numpy as np
from tqdm import tqdm

from data.loading import FGNETSeparateDataModule
from metrics.metrics import calc_metrics
from model.model import FGNETSeparatePredictor
from utils.util import EasyDict, num_available_cpus


# ======= Helper Methods =======

def get_wandb_id_from_path(path):
    def is_wandb_id(s: str):
        return len(s) == 8 and s.isalnum() and s.islower()
        
    path = os.path.normpath(path)
    comps = path.split(os.sep)
    pot_ids = [comp for comp in comps if is_wandb_id(comp)]
    assert len(pot_ids) == 1, len(pot_ids)
    return pot_ids[0]


def unique_name(args, include_threshold=True, include_model=True, extension=None):
    id = args.model_id or get_wandb_id_from_path(args.checkpoint)
    dataset_base = os.path.basename(args.dataset)
    dataset_base = os.path.splitext(dataset_base)[0]
    file_name = f"{id}-{dataset_base}"
    if args.subset:
        file_name += f"-s_{args.subset}"
    if include_model:
        model_file_name = os.path.basename(args.checkpoint)
        model = model_file_name.partition("-")[0] if "-" in model_file_name else os.path.splitext(model_file_name)[0]
        file_name += f"-{model}"
    if include_threshold:
        threshold = str(args.threshold)[2:]
        file_name += f"-t_{threshold}"
    if extension:
        file_name += extension
    return file_name

# ======= Prediction =======

def _predict(
        y_hat,
        y,
        items,
        labels,
        output_path,
        threshold=None,
        threshold_start=None,
        threshold_step=None,
        threshold_output_path=None,
        metric_to_watch="choi/f1-macro",
        verbose=False,
        fallback=None,
    ):
    assert threshold or threshold_step
    print("Watch metric:", metric_to_watch)

    if threshold_step:
        print("Threshold step:", threshold_step)
        threshold_scores = []
        pbar = tqdm(np.arange(threshold_start,1+threshold_step,threshold_step), disable=verbose)
        for _threshold in pbar:
            metrics = calc_metrics(y_hat, y, _threshold, metrics=["choi"], fallback=fallback)
            metrics["threshold"] = _threshold
            threshold_scores.append(metrics)
            if verbose:
                print(f"{round(_threshold,7)}:\t{metrics[metric_to_watch]}, {metrics['choi/f1-macro']}, {metrics['choi/precision-macro']}, {metrics['choi/recall-macro']}, {metrics['choi/f1-micro']}")
            else:
                pbar.set_postfix(dict(threshold=_threshold, score=round(metrics[metric_to_watch], 5)))

        if threshold_output_path:
            print("Write thresholds to:", threshold_output_path)
            with open(threshold_output_path, "w+") as f:
                fieldnames = threshold_scores[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(threshold_scores)

        max_score, max_threshold = -1, None
        for v in threshold_scores:
            if v[metric_to_watch] > max_score:
                max_score = v[metric_to_watch]
                max_threshold = v["threshold"]
        print("Max threshold:", max_threshold)
        threshold = max_threshold

    print("Threshold:", threshold)
    result = FGNETSeparatePredictor._predict_from_probs(y_hat, y, threshold, fallback=fallback)

    for k, v in result.metrics.items():
        print(f"{k}\t{v}")
    
    label_names = np.array(labels)
    outputs = []
    for x, true_pred, cert in zip(items, result.predictions, result.certainty):
        target, pred = true_pred
        o = dict(item=x, gold=label_names[target].tolist(), pred=label_names[pred].tolist(), cert=cert)
        outputs.append(o)
    print("#Output:", len(outputs))
    
    try:
        threshold_str = str(round(threshold,7))[2:]
        output_path = output_path.format(threshold=threshold_str)
    except:
        pass
    print("Save outputs to:", output_path)
    with open(output_path, "w+") as f:
        json.dump(outputs, f)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--cache", required=True, help="Path to the cached model output")
@click.option("--output", required=True, help="Location of output json (if output doesn't end in .json --output will be treated as a folder and file is automatically named)")
@click.option("--threshold", default=None, type=float, help="Fixed prediction threshold")
@click.option("--threshold-step", default=None, type=float, help="Step size for automatic threshold")
@click.option("--threshold-start", default=0, type=float, help="Start value for automatic threshold")
@click.option("--threshold-output-path", default=None, help="Location of csv file for threshold scores if desired")
@click.option("--add-cache", default=None, help="If the cache only contains y_hat, you can add a path for a pkl with y, items and labels")
@click.option("--metric", default="choi/f1-macro", help="Metric to optimize threshold for")
@click.option("--label-fallback", default=None, type=int, help="Label fallback instead of maximum probability")
@click.option("--verbose", is_flag=True, help="Verbose threshold and metric")
def reuse(**args):
    """
    Example (UFET):
    python src/predict.py reuse --cache MODEL_OUTPUT_CACHE.pkl --output OUTPUT_FOLDER --threshold-step 0.005

    Example (OntoNotes):
    python src/predict.py reuse --cache MODEL_OUTPUT_CACHE.pkl --output OUTPUT_FOLDER --threshold-step 0.005 --label-fallback 2
    
    Example (cache only contains prediction tensor)
    python src/predict.py reuse --cache PREDICTION.pkl --output OUTPUT_FOLDER --threshold-step 0.005 --add-cache MODEL_OUTPUT_CACHE.pkl
    """
    args = EasyDict(args)
    assert bool(args.threshold) ^ bool(args.threshold_step), "Only one of --threshold or --threshold-step should be defined"

    result = torch.load(args.cache)
    if isinstance(result, torch.Tensor):
        y_hat = result
        assert args.add_cache, "--cache only contains a single tensor, use --add-cache to add cached items for y, items and labels"
        add_data = torch.load(args.add_cache)
        result = dict(y_hat=y_hat, y=add_data["y"], items=add_data["items"], labels=add_data["labels"])

    output_path = args.output
    threshold_output_path = args.threshold_output_path
    if not output_path.endswith(".json"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.basename(args.cache)
        base = os.path.splitext(base)[0]
        if threshold_output_path:
            file_name = f"{base}_thresholds.csv"
            new_threshold_output_path = os.path.join(output_path, file_name)
            print(f"Thresholds scores will be written to {new_threshold_output_path} instead of {threshold_output_path}")
            threshold_output_path = new_threshold_output_path
        
        # Specific threshold will be filled in later
        file_name = f"{base}" + "-t_{threshold}.json"
        output_path = os.path.join(output_path, file_name)
    _predict(
        **result,
        threshold=args.threshold,
        threshold_start=args.threshold_start,
        threshold_step=args.threshold_step,
        output_path=output_path,
        threshold_output_path=threshold_output_path,
        metric_to_watch=args.metric,
        verbose=args.verbose,
        fallback=args.label_fallback,
    )


@cli.command()
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--dataset", required=True, help="Path to dataset to predict")
@click.option("--labels", required=True, help="Path to labels to predict dataset with")
@click.option("--output", required=True, help="Location of output json (if output doesn't end in .json --output will be treated as a folder and file is automatically named)")
@click.option("--save-model-output", default=None, help="Store the model output to reuse for faster prediction")
@click.option("--model-id", default=None, help="Model id for automatic file naming (if not specified and automatic naming is on, try to extract wandb id from path")
@click.option("--batch-size", default=128, help="Batch size to use for prediction")
@click.option("--subset", default=0, help="Use a subset of the dataset for faster debugging")
@click.option("--threshold", default=0.5, help="Prediction threshold")
@click.option("--label-fallback", default=None, type=int, help="Label fallback instead of maximum probability")
def predict(**args):
    """
    Example (UFET):
    python src/predict.py predict --checkpoint BEST_MODEL.ckpt --dataset data/ufet/ufet_dev.json --labels data/ontology/ufet_types.txt --output OUTPUT_FOLDER --save-model-output CACHE_FOLDER --model-id 123 --batch-size 128

    Example (OntoNotes):
    python src/predict.py predict --checkpoint BEST_MODEL.ckpt --dataset data/onto/onto_dev.json --labels data/ontology/onto_types.txt --output OUTPUT_FOLDER --save-model-output CACHE_FOLDER --model-id 123 --batch-size 128 --label-fallback 2
    """
    args = EasyDict(args)
    checkpoint_path = args.checkpoint
    print("Using checkpoint:", checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model = FGNETSeparatePredictor(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval_batch_size = args.batch_size
    model.eval()
    transformer_name = model.encoder.name_or_path
    print("Transformer name:", transformer_name)
    
    print("Dataset:", args.dataset)
    num_workers = num_available_cpus()
    data_module_kwargs = dict(
        data_files=dict(val=args.dataset),
        label_path=args.labels,
        tokenizer=transformer_name,
        train_batch_size=0,
        eval_batch_size=args.batch_size,
        num_workers=num_workers,
        max_length=128,
        subset=args.subset,
    )
    data_module = FGNETSeparateDataModule(**data_module_kwargs)
    data_module.setup()
    loader = data_module.val_dataloader()
    data = data_module.val_data
    
    model.to(device)
    y_hat, y = model._calc_probs(loader, data_module.label_dataloader())

    labels = data_module.labels
    data.reset_format()
    items = data["item"]
    output = dict(y_hat=y_hat, y=y, items=items, labels=labels)
    
    if args.save_model_output:
        if args.save_model_output.endswith(".pkl"):
            file_path = args.save_model_output
        else:
            unique_file_name = unique_name(args, include_threshold=False, extension=".pkl")
            os.makedirs(args.save_model_output, exist_ok=True)
            file_path = os.path.join(args.save_model_output, unique_file_name)
        print("Saving model output to:", file_path)
        torch.save(output, file_path)
    
    output_path = args.output
    if not output_path.endswith(".json"):
        file_name = unique_name(args, include_threshold=True, extension=".json")
        output_path = os.path.join(output_path, file_name)

    _predict(**output, threshold=args.threshold, output_path=output_path, fallback=args.label_fallback)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cli()
