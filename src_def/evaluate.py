"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
import model.def_net as def_net
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument("--is_def", default=True, action="store_true")

def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
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
    tagged_sentences = []
    labels = []
    loss_avg = utils.RunningAverage()    

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch = next(data_iterator)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)
        if __name__ == '__main__':
            data_batch = data_batch.data.cpu().numpy().tolist()
            labels_batch = labels_batch.tolist()
            output_batch = output_batch > 0.5
            for x, y, z in zip(data_batch, output_batch, labels_batch):
                z = int(z[0])
                y = int(y[0])
                tagged_sent = " ".join([data_loader.vocabi2c[_] for _ in x]) + '<' + str(data_loader.inv_tag_map[y]) + '/>' + '<' + str(data_loader.inv_tag_map[z]) + '/>'
                tagged_sentences.append(tagged_sent)

        # update the average loss
        loss_avg.update(loss.item())
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    if __name__ == '__main__':    
        # write out tagged sentences
        ofname = os.path.join(args.model_dir, 'output_tagged_sentences.txt')
        with open(ofname, 'w') as fout:
            fout.write("\n".join(tagged_sentences))
    
    return metrics_mean, loss_avg()


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
    data_loader = DataLoader(args.data_dir, params, args.is_def)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']
    logging.info("Loading {}".format(args.data_dir))

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.info("- done.")

    # Define the model
    if args.is_def:
        model = def_net.Net(params).cuda() if params.cuda else def_net.Net(params)
        loss_fn = def_net.loss_fn
        metrics = def_net.metrics
    else:
        model = net.Net(params).cuda() if params.cuda else net.Net(params)
        loss_fn = net.loss_fn
        metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics, _ = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
