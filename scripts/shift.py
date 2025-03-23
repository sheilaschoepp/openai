import os
import pandas as pd



normal = "../data/PPO_FetchReachEnv-v7:1000000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v7_4 = "../data/PPO_FetchReachEnv-v15:500000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:2500_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v11_4 = "../data/PPO_FetchReachEnv-v17:500000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:2500_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v8_o = "../data/PPO_FetchReachEnv-v16:500000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:2500_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v12_o = "../data/PPO_FetchReachEnv-v18:500000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:2500_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"


def shift_to_data(exp_path, new_exp_path):
    if not os.path.exists(exp_path):
        print(f"Folder does not exist: {exp_path}")
        return

    for seed_folder in os.listdir(exp_path):
        seed_path = os.path.join(exp_path, seed_folder, 'csv')
        csv_file = os.path.join(seed_path, 'eval_data.csv')
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            output_csv_file = f'eval_data_{seed_folder.strip()[-1] if len(seed_folder)==5 else seed_folder.strip()[-2:]}.csv'
            os.makedirs(f'data/{new_exp_path}', exist_ok=True)
            data.to_csv(f'data/{new_exp_path}/{output_csv_file}', index=False)


shift_to_data(normal, 'Expert')
# shift_to_data(v7_4, 'FetchReach-LLM-4-v7')
# shift_to_data(v8_o, 'FetchReach-LLM-o-v8')
# shift_to_data(v11_4, 'FetchReach-LLM-4-v9')
# shift_to_data(v12_o, 'FetchReach-LLM-o-v10')