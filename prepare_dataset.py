import argparse
import os
import h5py
import yaml
import numpy as np
from utils import get_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    
    dataset_kwargs = config['dataset_kwargs']
    dataset_name = config['dataset_name']
    landmark = config['landmark']

    seqs, ts, cs, target, h_ws, mask = get_data(dataset_name,
                                                landmark,
                                                dataset_kwargs)
    # From jax.numpy to numpy
    seqs = np.array(seqs)
    ts = np.array(ts)
    cs = np.array(cs)
    target = np.array(target)
    h_ws = np.array(h_ws)
    mask = np.array(mask)
    
    root_path = '/Users/mariana/Documents/projects/Huawei/SurvanData'
    try:
        assert os.path.exists(root_path)
    except FileNotFoundError:
        raise Exception('Wrong root path!')
    
    horizon = dataset_kwargs['horizon']
    if dataset_name != 'single_task':
        task_id = 'mixed'

    file_path = os.path.join(root_path, dataset_name, task_id, f'H_{horizon}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, 'dataset.h5')

    with h5py.File(file_path, 'w') as file:
        # Store each array with a unique key
        file.create_dataset('seqs', data=seqs)
        file.create_dataset('ts', data=ts)
        file.create_dataset('cs', data=cs)
        file.create_dataset('h_tgt', data=target)
        file.create_dataset('h_ws', data=h_ws)
        file.create_dataset('mask', data=mask)
