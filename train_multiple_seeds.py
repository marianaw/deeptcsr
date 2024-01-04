
from baseline_cox import SA
from utils import concordance_index, score
import yaml
import pandas as pd


if __name__ == '__main__':

    config_path = 'config_server.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    output_path = 'Results/task_{}.csv'.format(config['dataset_kwargs']['task_id'])

    num_exps = 1
    size = .2

    res2_is = []
    res2_lm = []
    
    res3_is = []
    res3_lm = []

    for fold in range(num_exps):
            
        print(".", end="")
        
        # Initial state.
        config['landmark'] = False
        model = SA(config, seed=fold)
        
        train_gen, test_gen = model.get_train_test(test_size=size)
        seqs = test_gen.X
        ts = test_gen.ts
        cs = test_gen.cs
        
        _ = model.train(train_gen, test_gen)
        res2_is.append(model.integrated_brier_score(seqs[:, 0], ts, cs))
        beta = model.state.params['cox_linear_model']['beta']
        scores = score(beta, seqs[:, 0])
        res3_is.append(concordance_index(scores, ts, cs))
        
        # Landmarking.
        config['landmark'] = True
        model = SA(config, seed=fold)
        
        train_gen, test_gen = model.get_train_test(test_size=size)
        seqs = test_gen.X
        ts = test_gen.ts
        cs = test_gen.cs
        
        _ = model.train(train_gen, test_gen)
        res2_lm.append(model.integrated_brier_score(seqs[:, 0], ts, cs))
        beta = model.state.params['cox_linear_model']['beta']
        scores = score(beta, seqs[:, 0])
        res3_lm.append(concordance_index(scores, ts, cs))
            
        print()
    
    df = pd.DataFrame({'init_state_bs': res2_is,
                       'init_state_ci': res3_is,
                       'landm_bs': res2_lm,
                       'landm_ci': res3_lm})
    df.to_csv(output_path)
