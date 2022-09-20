"""
  This code is largely borrowed from Eunsol Choi's implementation.

  GitHub: https://github.com/uwnlp/open_type
  Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf
"""

import json
import pickle
import numpy as np


def f1(p, r):
  if r == 0. or p == 0.:
    return 0.
  return 2 * p * r / float(p + r)


def strict(true_and_prediction):
  num_entities = len(true_and_prediction)
  correct_num = 0.
  for true_labels, predicted_labels in true_and_prediction:
    correct_num += set(true_labels) == set(predicted_labels)
  precision = recall = correct_num / num_entities
  return precision, recall, f1(precision, recall)


def macro(true_and_prediction):
  num_examples = len(true_and_prediction)
  p = 0.
  r = 0.
  pred_example_count = 0.
  pred_label_count = 0.
  gold_label_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
      p += per_p
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
      r += per_r
  if pred_example_count > 0:
    avg_elem_per_pred = pred_label_count / pred_example_count
    precision = p / pred_example_count
  else:
    avg_elem_per_pred = 0.
    precision = 0.
  if gold_label_count > 0:
    recall = r / gold_label_count
  else:
    recall = 0
    # print("WARNING: There are no gold labels when calculating macro")
  
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)


def micro(true_and_prediction):
  num_examples = len(true_and_prediction)
  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0:
    return num_examples, 0., 0., 0., 0., 0.
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)


def mrr(dist_list, gold):
  """
  dist_list: list of list of label probability for all labels.
  gold: list of gold indexes.

  Get mean reciprocal rank. (this is slow, as have to sort for 10K vocab)
  """
  mrr_per_example = []
  dist_arrays = np.array(dist_list)
  dist_sorted = np.argsort(-dist_arrays, axis=1)
  for ind, gold_i in enumerate(gold):
    gold_i_where = [i for i in range(len(gold_i)) if gold_i[i] == 1]
    rr_per_array = []
    sorted_index = dist_sorted[ind, :]
    for gold_i_where_i in gold_i_where:
      for k in range(len(sorted_index)):
        if sorted_index[k] == gold_i_where_i:
          rr_per_array.append(1.0 / (k + 1))
    mrr_per_example.append(np.mean(rr_per_array))
  return sum(mrr_per_example) * 1.0 / len(mrr_per_example)


def stratify(all_labels, types):
  """
  Divide label into three categories.
  """
  coarse = types[:9]
  fine = types[9:130]
  return ([l for l in all_labels if l in coarse],
          [l for l in all_labels if ((l in fine) and (not l in coarse))],
          [l for l in all_labels if (not l in coarse) and (not l in fine)])


def get_mrr(pred_fname):
  dicts = pickle.load(open(pred_fname, "rb"))
  mrr_value = mrr(dicts['pred_dist'], dicts['gold_id_array'])
  return mrr_value


def compute_granul_prf1(fname, type_fname):
  """UFET only"""
  with open(fname) as f:
    total = json.load(f)
  coarse_true_and_predictions = []
  fine_true_and_predictions = []
  finer_true_and_predictions = []
  with open(type_fname) as f:
    types = [x.strip() for x in f.readlines()]
  for k, v in total.items():
    coarse_gold, fine_gold, finer_gold = stratify(v['gold'], types)
    coarse_pred, fine_pred, finer_pred = stratify(v['pred'], types)
    coarse_true_and_predictions.append((coarse_gold, coarse_pred))
    fine_true_and_predictions.append((fine_gold, fine_pred))
    finer_true_and_predictions.append((finer_gold, finer_pred))

  for true_and_predictions in [coarse_true_and_predictions, fine_true_and_predictions, finer_true_and_predictions]:
    count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
    perf = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(
        count, avg_pred_count, p * 100, r * 100, f1 * 100)
    print(perf)


def _compute_granul_prf1(results, types):
  """UFET only"""
  coarse_true_and_predictions = []
  fine_true_and_predictions = []
  finer_true_and_predictions = []
  if not isinstance(types, list):
    with open(types) as f:
      types = [x.strip() for x in f.readlines()]
  for v in results:
    coarse_gold, fine_gold, finer_gold = stratify(v['gold'], types)
    coarse_pred, fine_pred, finer_pred = stratify(v['pred'], types)
    coarse_true_and_predictions.append((coarse_gold, coarse_pred))
    fine_true_and_predictions.append((fine_gold, fine_pred))
    finer_true_and_predictions.append((finer_gold, finer_pred))

  for true_and_predictions in [coarse_true_and_predictions, fine_true_and_predictions, finer_true_and_predictions]:
    count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
    perf = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(
        count, avg_pred_count, p * 100, r * 100, f1 * 100)
    print(perf)


def load_augmented_input(fname):
  output_dict = {}
  with open(fname) as f:
    for line in f:
      elem = json.loads(line.strip())
      mention_id = elem.pop("annot_id")
      output_dict[mention_id] = elem
  return output_dict


def visualize(gold_pred_fname, original_fname, type_fname):
  with open(gold_pred_fname) as f:
    total = json.load(f)
  original = load_augmented_input(original_fname)
  with open(type_fname) as f:
    types = [x.strip() for x in f.readlines()]
  for annot_id, v in total.items():
    elem = original[annot_id]
    mention = elem['mention_span']
    left = elem['left_context_token']
    right = elem['right_context_token']
    text_str = ' '.join(left)+" __"+mention+"__ "+' '.join(right)
    gold = v['gold']
    print('  |  '.join([text_str, ', '.join([("__"+v+"__" if v in gold else v )for v in v['pred']]), ','.join(gold)]))
