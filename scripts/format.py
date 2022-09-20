import json
import click
from tqdm import tqdm

def remove_brackets(x: str):
    x = x.replace("-LRB-", "(")
    x = x.replace("-RRB-", ")")
    x = x.replace("-LSB-", "[")
    x = x.replace("-RSB-", "]")
    return x

@click.command()
@click.option("--file", required=True, help="Input file to format")
@click.option("--output", required=True, help="Output location")
def eval(**args):
    with open(args.file) as f:
        content = f.read().splitlines()

    with open(args.output, "w+") as f:
        for i, l in enumerate(tqdm(content)):
            line = json.loads(l)
            line["word"] = remove_brackets(line["mention_span"])
            del line["mention_span"]
            line["left_context_text"] = remove_brackets(" ".join(line["left_context_token"]))
            line["right_context_text"] = remove_brackets(" ".join(line["right_context_token"]))
            line["y_category"] = line["y_str"]

            o = json.dumps(line) + "\n"
            f.write(o)
