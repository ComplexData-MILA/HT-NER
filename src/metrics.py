from typing import List
import pandas as pd
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")


def f1(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    ground_truth_column: List[str],
    prediction_column: List[str],
    epsilon: float = 1e-7,
):
    # Initialize dictionaries to keep track of true positives, false positives, and false negatives
    entity_tp = {}
    entity_fp = {}
    entity_fn = {}
    token_tp = {}
    token_fp = {}
    token_fn = {}

    # Iterate through each specified column in the ground truth and prediction CSV files
    for gt_col, pred_col in zip(ground_truth_column, prediction_column):
        col = gt_col + " with " + pred_col
        # Extract entities and predicted entities from the current column
        ground_truth_entities = (
            ground_truth[gt_col]
            .fillna("")
            .apply(lambda x: x.lower())
            .str.split("|")
            .apply(lambda x: list(y.strip() for y in x))
            .tolist()
        )
        predicted_entities = (
            prediction[pred_col]
            .fillna("")
            .apply(lambda x: x.lower())
            .str.split("|")
            .apply(lambda x: list(y.strip() for y in x))
            .tolist()
        )

        # Initialize the true positives, false positives, and false negatives to 0
        entity_tp[col] = epsilon  # entity_tp.get(col, 0)
        entity_fp[col] = epsilon  # entity_fp.get(col, 0)
        entity_fn[col] = epsilon  # entity_fn.get(col, 0)
        # Initialize the true positives, false positives, and false negatives to 0
        token_tp[col] = epsilon  # token_tp.get(col, 0)
        token_fp[col] = epsilon  # token_fp.get(col, 0)
        token_fn[col] = epsilon  # token_fn.get(col, 0)

        # Iterate through each row in the ground truth and prediction CSV files
        for j in range(len(ground_truth_entities)):

            pred_list = predicted_entities[j]
            pred_set = set(pred_list)
            ground_truth_list = ground_truth_entities[j]
            ground_truth_set = set(ground_truth_list)

            # Iterate through each ground truth entity in the current row
            for entity in ground_truth_list:
                if not entity:
                    continue

                # If the entity is also present in the predicted entities for the current row, count it as a true positive
                if entity in pred_set:
                    entity_tp[col] += 1
                # Otherwise, count it as a false negative
                else:
                    entity_fn[col] += 1

            # Iterate through each predicted entity in the current row
            for entity in pred_list:
                if not entity:
                    continue

                # If the predicted entity is not present in the ground truth entities for the current row, count it as a false positive
                if entity not in ground_truth_set:
                    entity_fp[col] += 1

            # Flatten the list of entities and predicted entities to evaluate the token-level F1 score

            ground_truth_tokens = []
            for row in ground_truth_list:
                for entity in row:
                    ground_truth_tokens.extend(tokenizer.tokenize(entity))

            predicted_tokens = []
            for row in pred_list:
                for entity in row:
                    predicted_tokens.extend(tokenizer.tokenize(entity))

            pred_set = set(predicted_tokens)
            ground_truth_set = set(ground_truth_tokens)

            # Iterate through each ground truth token
            for token in ground_truth_tokens:

                # If the token is also present in the predicted tokens, count it as a true positive
                if token in pred_set:
                    token_tp[col] += 1
                # Otherwise, count it as a false negative
                else:
                    token_fn[col] += 1

            # Iterate through each predicted token
            for token in predicted_tokens:

                # If the predicted token is not present in the ground truth tokens, count it as a false positive
                if token not in ground_truth_set:
                    token_fp[col] += 1

    # Compute precision, recall, and F1 score for each entity
    entity_precision = {}
    entity_recall = {}
    entity_f1 = {}
    for entity in entity_tp.keys():
        entity_precision[entity] = entity_tp[entity] / (
            entity_tp[entity] + entity_fp.get(entity, 0)
        )
        entity_recall[entity] = entity_tp[entity] / (
            entity_tp[entity] + entity_fn.get(entity, 0)
        )
        entity_f1[entity] = (
            2
            * entity_precision[entity]
            * entity_recall[entity]
            / (entity_precision[entity] + entity_recall[entity])
        )

    # Compute precision, recall, and F1 score for each token
    token_precision = {}
    token_recall = {}
    token_f1 = {}
    for token in token_tp.keys():
        token_precision[token] = token_tp[token] / (
            token_tp[token] + token_fp.get(token, 0)
        )
        token_recall[token] = token_tp[token] / (
            token_tp[token] + token_fn.get(token, 0)
        )
        token_f1[token] = (
            2
            * token_precision[token]
            * token_recall[token]
            / (token_precision[token] + token_recall[token])
        )

    # Create matrices to display the precision, recall, and F1 score for each entity and token
    entity_matrix = pd.DataFrame(
        {"Precision": entity_precision, "Recall": entity_recall, "F1 Score": entity_f1}
    )
    token_matrix = pd.DataFrame(
        {"Precision": token_precision, "Recall": token_recall, "F1 Score": token_f1}
    )

    # Print the matrices
    print("\nEntity-Level Evaluation:")
    print(entity_matrix)
    print("\nToken-Level Evaluation:")
    print(token_matrix)


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
    args = parser.parse_args()

    # Load ground truth CSV file
    ground_truth = pd.read_csv(args.ground_truth)

    # Load prediction CSV file
    prediction = pd.read_csv(args.prediction)

    f1(ground_truth, prediction, args.ground_truth_column, args.prediction_column)
