from collections import defaultdict
import json
import click
from metrics.scorer import _compute_granul_prf1, macro

from utils.util import EasyDict

# From https://github.com/HKUST-KnowComp/MLMET/blob/main/utils/utils.py
pronouns = {'i', 'me', 'myself', 'we', 'us', 'ourselves', 'he', 'him', 'himself', 'she', 'her', 'herself',
            'it', 'they', 'them', 'themselves', 'you'}

def are_words_cap(mstr: str):
    words = mstr.split(' ')
    for w in words:
        if w.lower() in {'the', 'of', 'and', 'a', ','}:
            continue
        if len(w) > 0 and w[0].islower():
            return False
    return True

def get_mention_type(mstr):
    if mstr.lower() in pronouns:
        return "pronoun"
    if are_words_cap(mstr):
        return "named-entity"
    return "nominal"


@click.command()
@click.option("--results", required=True, help="Predicted results json")
@click.option("--labels", required=True, help="Path to labels")
def eval(**args):
    """
    Example:
    python eval.py --results PREDICTIONS.json --labels data/ontology/ufet_types.txt
    """
    args = EasyDict(args)
    with open(args.results) as f:
        results = json.load(f)
    print(results[0])
    _compute_granul_prf1(results, args.label)
    mention_types = defaultdict(list)
    for x in results:
        mention = x["item"]
        mention = mention.partition("[MENTION/]")[2]
        mention = mention.partition("[/MENTION]")[0]
        mention = mention.strip()
        mention_type = get_mention_type(mention)
        mention_types[mention_type].append(x)

    for mention_type, result in mention_types.items():
        print(mention_type, len(result), result[0]["item"])
        true_predictions = [(x["gold"], x["pred"]) for x in result]
        _, _, _, precision, recall, f1 = macro(true_predictions)
        print(precision, recall, f1)

if __name__ == "__main__":
    eval()

