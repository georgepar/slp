# this script contains custom argument parsers
# - dataset: [mosei, mosi]
# - models: [rnn, transformer]
import os
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm


def get_mosei_parser():
    parser = ArgumentParser("Transformer-MOSEI example")

    # model hyperparameters
    parser.add_argument(
        "--max_length",
        dest="model.max_length",
        type=int,
        default=1024,
        help="Sequence max length for transformer layers",
    )
    parser.add_argument(
        "--kernel_size",
        dest="model.kernel_size",
        type=int,
        default=33,
        help="Kernel Size used for residual trick in transformer layers",
    )
    parser.add_argument(
        "--use_nystrom",
        dest="model.nystrom",
        default=False,
        action="store_true",
        help="Use nystrom self-attention approximation",
    )
    parser.add_argument(
        "--num_landmarks",
        dest="model.num_landmarks",
        type=int,
        default=32,
        help="Number of landmarks used for nystrom approximation",
    )
    parser.add_argument(
        "--num_layers",
        dest="model.num_layers",
        type=int,
        default=3,
        help="Number of Transformer Layers",
    )
    parser.add_argument(
        "--num_heads",
        dest="model.num_heads",
        type=int,
        default=4,
        help="Number of Transformer Heads per Layer",
    )
    parser.add_argument(
        "--dropout",
        dest="model.dropout",
        type=float,
        default=0.3,
        help="Drop probability for each Transformer Layer",
    )
    parser.add_argument(
        "--hidden_size",
        dest="model.hidden_size",
        type=int,
        default=100,
        help="Hidden Size for each Transformer Layer",
    )
    parser.add_argument(
        "--inner_size",
        dest="model.inner_size",
        type=int,
        default=200,
        help="Inner Size for each Transformer Layer",
    )
    parser.add_argument(
        "--prenorm",
        dest="model.prenorm",
        default=False,
        action="store_true",
        help="Use the normalization before residual."
        "Default value is False as in Vaswani.",
    )
    parser.add_argument(
        "--scalenorm",
        dest="model.scalenorm",
        default=False,
        action="store_true",
        help="Use scalenorm (L2) instead of LayerNorm (Vaswani).",
    )
    parser.add_argument(
        "--mmdrop",
        dest="model.multi_modal_drop",
        type=str,
        default="dropout",
        choices=["dropout", "mmdrop_hard", "mmdrop_soft", "none"],
        help="Which dropout is applied to the late fusion stage",
    )
    parser.add_argument(
        "--p-mmdrop",
        dest="model.p_mmdrop",
        type=float,
        default=0.33,
        help="probability of droping 1/3 modlities",
    )
    parser.add_argument(
        "--p-drop-modalities",
        dest="model.p_drop_modalities",
        default=[0.33, 0.33, 0.33],
        help="Per modal drop rate",
    )

    # dataset specifications
    parser.add_argument(
        "--pad-front",
        dest="preprocessing.pad_front",
        action="store_true",
        help="Handles front padding. Default is True",
    )
    parser.add_argument(
        "--pad-back",
        dest="preprocessing.pad_back",
        action="store_true",
        help="Handles back padding. Default is False",
    )
    parser.add_argument(
        "--remove-pauses",
        dest="preprocessing.remove_pauses",
        action="store_true",
        help="When used removes pauses from dataset",
    )
    parser.add_argument(
        "--crop-length",
        dest="preprocessing.max_length",
        default=-1,
        type=int,
        help="Crop feature sequences during data ingestion from cmusdk",
    )
    parser.add_argument(
        "--already-unaligned",
        dest="preprocessing.already_aligned",
        action="store_true",
        help="When used indicates unaligned data otherwise"
        " aligned scenario is used.",
    )
    parser.add_argument(
        "--do-align",
        dest="preprocessing.align_features",
        action="store_true",
        help="When used automatically aligns the features.",
    )

    return parser


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    f_score = f1_score((test_preds >= 0), (test_truth >= 0), average="weighted")
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    f_score_neg = f1_score((test_preds > 0), (test_truth > 0), average="weighted")
    binary_truth_neg = test_truth > 0
    binary_preds_neg = test_preds > 0

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    f_score_non_zero = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_non_zero = test_truth[non_zeros] >= 0
    binary_preds_non_zero = test_preds[non_zeros] >= 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1_pos": f_score,  # zeros are positive
        "bin_acc_pos": accuracy_score(binary_truth, binary_preds),  # zeros are positive
        "f1_neg": f_score_neg,  # zeros are negative
        "bin_acc_neg": accuracy_score(
            binary_truth_neg, binary_preds_neg
        ),  # zeros are negative
        "f1": f_score_non_zero,  # zeros are excluded
        "bin_acc": accuracy_score(
            binary_truth_non_zero, binary_preds_non_zero
        ),  # zeros are excluded
    }


def eval_mosei_senti_old(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] >= 0), (test_truth[non_zeros] >= 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] >= 0
    binary_preds = test_preds[non_zeros] >= 0
    f_score_neg = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_neg = test_truth[non_zeros] > 0
    binary_preds_neg = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1": f_score,
        "f1_neg": f_score_neg,
        "bin_acc_neg": accuracy_score(binary_truth_neg, binary_preds_neg),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print("{}:\t{}".format(k, v))


def save_metrics(metrics, results_file):
    with open(results_file, "w") as fd:
        for k, v in metrics.items():
            print("{}:\t{}".format(k, v), file=fd)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["neutral", "happy", "sad", "angry"]
    test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()

    results = {}
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average="weighted")
        acc = accuracy_score(test_truth_i, test_preds_i)
        results["{}_acc".format(emos[emo_ind])] = acc
        results["{}_f1".format(emos[emo_ind])] = f1

    return results


def run_evaluation(model, test_loader, results_file):
    ordered_metrics = [
        "mae",
        "corr",
        "acc_7",
        "acc_5",
        "f1_pos",
        "bin_acc_pos",
        "f1_neg",
        "bin_acc_neg",
        "f1",
        "bin_acc",
    ]
    model.eval()
    preds, ground_truths = [], []
    for batch in tqdm(test_loader, desc="Running on test set"):
        pred, gt = model.predictor.get_predictions_and_targets(model, batch)
        preds.append(pred)
        ground_truths.append(gt)

    preds = torch.cat(preds)
    ground_truths = torch.cat(ground_truths)
    metrics = eval_mosei_senti(preds, ground_truths, exclude_zero=True)
    if not os.path.exists(results_file):
        with open(results_file, "w") as fd:
            header = "\t".join(ordered_metrics) + "\n"
            fd.write(header)
            values = [f"{metrics[m].item():.4}" for m in ordered_metrics]
            line = "\t".join(values) + "\n"
            fd.write(line)
    else:
        with open(results_file, "a") as fd:
            values = [f"{metrics[m].item():.4f}" for m in ordered_metrics]
            line = "\t".join(values) + "\n"
            fd.write(line)
    return metrics
