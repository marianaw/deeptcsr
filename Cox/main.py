import os
from shutil import copyfile


import yaml
import argparse
import gc

from lambda_cox import LambdaSA
from baseline_cox import SA
from deep_lambda_cox import DeepLambdaSA


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

if __name__ == '__main__':  

    #Read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='Path to configuration file.')
    parser.add_argument('--exp_name', type=str, default='exp_1', 
                        help='exp name')
    parser.add_argument('--agent', type=str, default='SA',
                        help='SA or LambdaSA or DeepLambdaSA')
    parser.add_argument('--seed', help='Experiment seed', type=int, default=42)
    parser.add_argument('--size', help='Test set ratio', type=float, default=.2)
    parser.add_argument('--val_size', help='Validation set ratio (pass "none" to skip)', type=float, default=None)

    # Overwrites some entries in config
    parser.add_argument('--taskid', help='Task id', type=int, default=None)
    parser.add_argument('--lambda_', help='Lambda', type=float, default=None)
    parser.add_argument('--target_lr', help='Target learning rate', type=float, default=None)
    parser.add_argument('--landmark', help='Landmarking', type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    exp_name = args.exp_name
    seed = args.seed
    type_agent = args.agent
    size = args.size
    val_size = args.val_size

    #Model parameters
    if args.taskid is not None:
        config['dataset_kwargs']['task_id'] = args.taskid

    if args.lambda_ is not None:
        config['lambda_'] = args.lambda_
    if args.target_lr is not None:
        config['target_lr'] = args.target_lr
    if args.landmark is not None:
        config['landmark'] = bool(args.landmark)

    output_file = config['output_file'].format(args.agent)
    output_file = os.path.join(output_file, 
                               config['dataset_name'],
                               config['arch']['type'],
                               'lambda_{}'.format(config['lambda_']),
                               'landmark_{}'.format(config['landmark']),
                               'target_lr_{}'.format(config['target_lr']),
                               exp_name)
    config['output_file'] = output_file

    if type_agent == "SA":
        agent = SA(config, seed)
    elif type_agent == 'LambdaSA':
        agent = LambdaSA(config, seed)
    elif type_agent == 'DeepLambdaSA':
        agent = DeepLambdaSA(config, seed)
    else:
        raise Exception('Agent type not found')

    try:
        # import ipdb; ipdb.set_trace()
        gens = agent.get_train_test(test_size=size, val_size=val_size)
        if len(gens) == 2:
            train_gen, test_gen = gens
            val_gen = None
        else:
            train_gen, test_gen, val_gen = gens

        agent.train(train_gen, val_gen=val_gen)
        agent.save()
        agent.eval(test_gen, suffix='test')
        if val_gen is not None:
            agent.eval(val_gen, suffix='val')

        config_path = os.path.join(output_file, 'config.yaml')
        with open(config_path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        
    except KeyboardInterrupt:
        gc.collect()
        pass
