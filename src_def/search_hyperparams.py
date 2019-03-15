"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument("--is_def", default=False, action="store_true")

def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    if args.is_def:
        cmd_ = "train.py --is_def"
    else:
        cmd_ = "train.py"
    cmd = "{python} {} --model_dir={model_dir} --data_dir {data_dir}".format(cmd_, python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    hyper_params = {}
    #hyper_params['learning_rate'] = [1e-4, 1e-3, 1e-2]
    #hyper_params['batch_size'] = [32, 64, 128, 256, 512, 1024, 2048]
    #hyper_params['defm_dropout_rate'] = [0.6,0.7,0.8,0.9]
    hyper_params['defm_cnn_kernels'] = [3,4,5,6,7,8,9]

    for k,v in hyper_params.items():
        for x in v:
            # Modify the relevant parameter in params
            params.dict[k] = x
            # Launch job (name has to be unique)
            job_name = "{}_{}".format(k,x)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)

