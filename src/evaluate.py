"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/ner_test_data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/test', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, terms,
            data_loader):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    cand_terms = {}
    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch = next(data_iterator)
        labels_batch = data_batch['labels']

        # compute model forward pass
        output_batch = model(data_batch)

        # remove padding & bert sub-words
        labels_batch = labels_batch.contiguous().view(-1)
        labels_mask = labels_batch >= 0
        if 'bert_mask' in data_batch.keys():
            output_batch = output_batch[data_batch['bert_mask'].contiguous().view(-1) == 1, :]
        else:
            output_batch = output_batch[labels_mask, :]
        labels_batch = labels_batch[labels_mask]

        # compute loss
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # extract terms classified by the model
        cand_terms = get_candidate_terms(output_batch, data_batch['sentences'],
                                         cand_terms, data_loader)

        # compute all metrics on this batch
        outputs, labels = compute_ner_labels(output_batch, labels_batch, data_loader)
        summary_batch = {'NER ' + metric: compute_metric(outputs, labels, metric)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all NER metrics in summary
    metrics_eval = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}

    # compute term metrics
    outputs, labels, term_info = compute_term_labels(terms, cand_terms)
    for metric in metrics:
        metrics_eval['Term ' + metric] = compute_metric(outputs, labels, metric)

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_eval.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_eval, term_info


def get_candidate_terms(outputs, data, terms, data_loader):
    """ Extracts all candidate terms for the given outputs.
    """

    predicted_labels = np.argmax(outputs, axis=1)
    probs = np.exp(np.max(outputs, axis=1))
    data = [word.lower() for sent in data for word in sent]

    i = 0
    while i < len(data):
        label = predicted_labels[i]
        if data_loader.id2tag[label] == 'S':
            cand_term = data[i]
            prob = probs[i]
            if cand_term in terms.keys():
                terms[cand_term].append(prob)
            else:
                terms[cand_term] = [prob]
        elif data_loader.id2tag[label] == 'B':
            cand_term = [data[i]]
            prob = [probs[i]]
            i += 1
            label = predicted_labels[i]
            while data_loader.id2tag[label] == 'I':
                cand_term.append(data[i])
                prob.append(probs[i])
                i += 1
                label = predicted_labels[i]
            if data_loader.id2tag[label] == 'E':
                cand_term.append(data[i])
                prob.append(probs[i])
                cand_term = ' '.join(cand_term)
                if cand_term in terms.keys():
                    terms[cand_term].append(sum(prob) / len(prob))
                else:
                    terms[cand_term] = [sum(prob) / len(prob)]
        i += 1

    return terms


def compute_ner_labels(outputs, labels, data_loader=None):
    """Converts tag level outputs and labels to token level outputs and labels. A 1 in the newly
    returned output corresponds to a predicted key term and a 0 otherwise. This ignores partial
    key phrase matching and thus key phrases are condensed into a single value in both the
    predicted outputs and the actual labels.


    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], padding already excluded

    Returns:
        outputs: (np.ndarray) dimension batch_size*seq_len 1 or 0 corresponding to predicted key
        term or phrase or not
        labels: (np.ndarray) dimension batch_size*seq_len 1 or 0 corresponding to key
        term/phrase or not
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    token_labels = []
    token_outputs = []

    i = 0
    while i < len(outputs):
        if labels[i] == -1:
            pass
        elif data_loader.id2tag[labels[i]] == 'S':
            token_labels.append(1)
            if data_loader.id2tag[outputs[i]] == 'S':
                token_outputs.append(1)
            else:
                token_outputs.append(0)
        elif data_loader.id2tag[labels[i]] == 'O':
            token_labels.append(0)
            if data_loader.id2tag[outputs[i]] == 'O':
                token_outputs.append(0)
            else:
                token_outputs.append(1)
        elif data_loader.id2tag[labels[i]] == 'B':
            token_labels.append(1)
            label = []
            predicted_label = []
            while data_loader.id2tag[labels[i]] != 'E':
                label.append(data_loader.id2tag[labels[i]])
                predicted_label.append(data_loader.id2tag[outputs[i]])
                i += 1
            if ''.join(label) == ''.join(predicted_label):
                token_outputs.append(1)
            else:
                token_outputs.append(0)
        i += 1

    return token_outputs, token_labels


def compute_term_labels(terms, candidate_terms):
    """ Converts the actual key terms and a list of candidate terms provided by the model into
        a list of predicted outputs and labels to facilitate computation of metrics.
        Currently candidates are any words tagged at least once

        Args:
            terms: list list of all key terms in the corpus
            candidate_terms: (dictionary) dictionary mapping all terms tagged at least once by the
                model to a list of confidence scores for each of those taggings

        Returns:
            outputs: (list) list of predicted key term labels for union of candidate terms and terms
            labels: (list) list of actual key term labels for union of candidate terms and terms
            term_info: (dict) dictionary containing list of term average confidence scores split
                into false positives, true positives, and false negatives

    """


    terms = [term.lower() for term in terms]
    # handle multiple representations
    for term in terms:
        split = term.split(';')
        if len(split) == 2:
            word, acronym = split
            word = word.strip()
            acronym = acronym.strip()
            if word in candidate_terms.keys() and acronym in candidate_terms.keys():
                candidate_terms[term] = candidate_terms[word] + candidate_terms[acronym]
                candidate_terms.pop(word)
                candidate_terms.pop(acronym)
            elif word in candidate_terms.keys():
                candidate_terms[term] = candidate_terms[word]
                candidate_terms.pop(word)
            elif acronym in candidate_terms.keys():
                candidate_terms[term] = candidate_terms[acronym]
                candidate_terms.pop(acronym)

    # extract out false positive, true positives, and false negatives with confidence score
    terms = set(terms)
    cand_terms = set(candidate_terms.keys())
    term_info = {'false_pos': [], 'true_pos': [], 'false_neg': []}
    for term in terms.intersection(cand_terms):
        term_info['true_pos'].append((term, sum(candidate_terms[term]) / len(candidate_terms[term])))
    for term in cand_terms.difference(terms):
        term_info['false_pos'].append((term, sum(candidate_terms[term]) / len(candidate_terms[term])))
    for term in terms.difference(cand_terms):
        term_info['false_neg'].append((term, 0))

    # label for the union of candidate and actual terms to compute metrics
    all_words = terms.union(cand_terms)
    labels = np.array([1 if word in terms else 0 for word in all_words])
    outputs = np.array([1 if word in candidate_terms.keys() else 0 for word in all_words])

    return outputs, labels, term_info

def compute_metric(outputs, labels, metric):

    if metric == 'accuracy':
        return np.sum(np.array(outputs) == np.array(labels))/float(len(outputs))
    elif metric == 'precision':
        return precision_score(labels, outputs, average='binary')
    elif metric == 'recall':
        return recall_score(labels, outputs, average='binary')
    elif metric == 'f1':
        return f1_score(labels, outputs, average='binary')
    else:
        raise ValueError('Unsupported metric: %s' % metric)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = ['accuracy', 'precision', 'recall', 'f1']

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    model = net.Baseline(params).cuda() if params.cuda else net.Baseline(params)

    loss_fn = net.loss_fn

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics, term_info  = evaluate(model, loss_fn, test_data_iterator, metrics,
                                        params, num_steps, test_data['terms'], data_loader)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

    # save best false positive, true positive, and false negative list for error analysis
    for list_type in ['false_pos', 'true_pos', 'false_neg']:
        fp_path = os.path.join(model_dir, '%s_test.txt' % list_type)
        with open(fp_path, 'w') as f:
            for fp in sorted(term_info[list_type]):
                f.write('%s, %.3f\n' % (fp[0], fp[1]))

