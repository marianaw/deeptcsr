import os
from shutil import copyfile


import yaml
import argparse
import gc

from lambda_cox import LambdaSA


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

if __name__ == '__main__':  

    #Read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    parser.add_argument('--exp_name', type=str, default='exp_1', 
                        help='exp name')
    parser.add_argument('--seed', help='Experiment seed', type=int, default=42)
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    exp_name = args.exp_name
    seed = args.seed

    #Model parameters
    output_file = config['output_file'].format(exp_name)
    config['output_file'] = output_file
    
    agent = LambdaSA(config, seed)

    try:
        agent.train()
        # copyfile(config_path, os.path.join(output_file, 'config.yaml'))
        config_path = os.path.join(output_file, 'config.yaml')
        with open(config_path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    except KeyboardInterrupt:
        gc.collect()
        pass
