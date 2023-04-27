from typing import List
import pandas as pd
import numpy as np
import ast
import copy

SPLIT_INTO_WORDS = True


def apply_lit(input):
    if input == "set()":
        return set()
    else:
        try:
            return set(ast.literal_eval(input))
        except:
            # TJBatch extractor results need to be split using the ; delimiter
            return set(input.split(";"))


def apply_lower(input):
    return set([x.lower() for x in list(input)])


def Containment_IoU(input1, input2):  # pred, true
    # input1, input2 are names in one record
    intersect_count = 0
    union_count = 0

    input1 = list(input1)
    input2 = list(input2)

    for i in input1:
        remain1 = copy.copy(input1)
        remain1.remove(i)
        for j in remain1:
            if i in j:
                try:
                    input1.remove(i)
                except:
                    continue
    for i in input2:
        remain2 = copy.copy(input2)
        remain2.remove(i)
        for j in remain2:
            if i in j:
                try:
                    input2.remove(i)
                except:
                    continue

    for i in input1:
        union_count += 1
        if ~(" " in i):
            for j in input2:
                if (i in j) or (j in i):
                    intersect_count += 1
                    input2.remove(j)
        else:
            for s in i.split(" "):
                for j in input2:
                    if (i in j) or (j in i):
                        intersect_count += 1
                        input2.remove(j)

    union_count += len(input2)

    mod_IoU = intersect_count / union_count if union_count > 0 else 1
    return mod_IoU


def Exact_Set(input1, input2):  # pred, true
    input1 = set(input1)
    input2 = set(input2)

    if input1 == input2:
        return 1
    else:
        return 0


def Exact_F1(pred, true):
    if SPLIT_INTO_WORDS:
        pred = [p.split() for p in pred]
        pred = set([y.lower() for x in pred for y in x])
        true = set([t.lower() for t in true])
    else:
        pred = set([x.lower().strip() for x in pred])
        true = set([t.lower().strip() for t in true])

    tp = 0
    fp = 0
    fn = 0

    for i in pred:
        if i in true:
            tp += 1
        else:
            #   print('FP:', repr(i), end=' ')
            fp += 1
    for i in true:
        if i not in pred:
            #   print('FN:', repr(i), end=' ')
            fn += 1
    #   if fp!=0 or fn!=0: print()
    return tp, fp, fn


def Partial_F1(pred, true):
    # pred = set(pred)
    pred = [p.split() for p in pred]
    pred = set([y.lower() for x in pred for y in x])
    true = set(true)

    tp = 0
    fp = 0
    fn = 0

    for i in pred:
        tp_flag = 0
        for j in true:
            if i in j or j in i:
                tp_flag = 1
                break
        if tp_flag == 1:
            tp += 1
        else:
            fp += 1
    for i in true:
        fn_flag = 1
        for j in pred:
            if i in j or j in i:
                fn_flag = 0
                break
        if fn_flag == 1:
            fn += 1

    return tp, fp, fn


def check_empty(input1, input2):
    input1 = set(input1)
    input2 = set(input2)

    if input1 == input2 and input1 == set():
        return True
    else:
        return False


def ad_level(pred, true):
    if SPLIT_INTO_WORDS:
        pred = [p.split() for p in pred]
        pred = set([y.lower() for x in pred for y in x])
        true = set([t.lower() for t in true])
    else:
        pred = set([x.lower().strip() for x in pred])
        true = set([t.lower().strip() for t in true])

    # fn if true is not empty but pred is empty
    if true and not pred:
        # print('FN', pred, true)
        return 0, 0, 1

    # TN
    if not true and not pred:
        return 0, 0, 0

    # compute IOU
    iou = len(pred & true) / len(pred | true)
    if iou >= 0.5:
        # print('TP', pred, true)
        return 1, 0, 0
    else:
        # print('FP', pred, true)
        return 0, 1, 0

    # return tp, fp, fn


Containment_IoU = np.vectorize(Containment_IoU)
Exact_Set = np.vectorize(Exact_Set)
Exact_F1 = np.vectorize(Exact_F1)
Partial_F1 = np.vectorize(Partial_F1)
check_empty = np.vectorize(check_empty)
ad_level = np.vectorize(ad_level)


def f1(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    ground_truth_column: List[str],
    prediction_column: List[str],
    epsilon: float = 1e-7,
    ignore_duplicates: bool = True,
):
    ### modified
    def apply_lit(x):
        return set(filter(None, (y.strip() for y in x)))

    results = []
    for pc, tc in zip(prediction_column, ground_truth_column):
        print(pc, end="\t")

        true_col = (
            ground_truth[tc]
            .fillna("")
            .replace(to_replace=r"^N$", value="", regex=True)
            .apply(lambda x: x.lower())
            .str.split("|")
            .apply(lambda x: set(filter(None, (y.strip() for y in x))))
        )
        pred_col = (
            prediction[pc]
            .fillna("")
            .replace(to_replace=r"^N$", value="", regex=True)
            .apply(lambda x: x.lower())
            .str.split("|")
            .apply(lambda x: set(filter(None, (y.strip() for y in x))))
        )

        comparison = Exact_F1(pred_col, true_col)
        tp = np.sum(comparison[0])
        fp = np.sum(comparison[1])
        fn = np.sum(comparison[2])

        avg_precision = tp / (tp + fp)
        avg_recall = tp / (tp + fn)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        # print('Individual strict match F1:', np.mean(avg_f1))
        # print('Individual strict match precision:', np.mean(avg_precision))
        # print('Individual strict match recall:', np.mean(avg_recall))
        print(
            f"{np.mean(avg_f1) :.6f}\t{np.mean(avg_precision) :.6f}\t{np.mean(avg_recall) :.6f}"
        )
        results.append(
            [pc, np.mean(avg_f1), np.mean(avg_precision), np.mean(avg_recall)]
        )
    return results


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate F1 score for each column in entity and token levels."
    )
    parser.add_argument(
        "--ground_truth", type=str, help="Name and location of ground truth CSV file."
    )
    parser.add_argument(
        "--ground_truth_column",
        nargs="+",
        help="Names of columns in ground truth CSV file that contain the entities.",
    )
    parser.add_argument(
        "--prediction", type=str, help="Location of prediction CSV file."
    )
    parser.add_argument(
        "--prediction_column",
        nargs="+",
        help="Names of columns in prediction CSV file that contain the entities.",
    )
    parser.add_argument(
        "--ignore_duplicates",
        type=int,
        default=1,
        help="Whether to ignore duplicate entities in the prediction file.",
    )
    args = parser.parse_args()

    # Load ground truth CSV file
    ground_truth = pd.read_csv(args.ground_truth)

    # Load prediction CSV file
    prediction = pd.read_csv(args.prediction)

    f1(
        ground_truth,
        prediction,
        args.ground_truth_column,
        args.prediction_column,
        ignore_duplicates=[False, True][args.ignore_duplicates],
    )
