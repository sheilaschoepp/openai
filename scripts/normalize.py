import os
import pandas as pd
import time

normal = "../data/PPO_FetchReachEnv-E-v20:1000000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v7_3 = "../data/PPO_FetchReachEnv-LLM-3-v7:1000000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v11_3 = "../data/PPO_FetchReachEnv-E-v11:1000000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v8_4 = "../data/PPO_FetchReachEnv-LLM-4-v8:1000000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
v12_4 = "../data/PPO_FetchReachEnv-E-v12:1000000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"



folders = [v7_3, v11_3, v8_4, v12_4, normal]

data_folder_names = ["v7_3", "v11_3", "v8_4", "v12_4", "normal"]

global_min_return = float('inf')
global_max_return = float('-inf')


def extract_min_max_from_folder(folder):
    global global_min_return, global_max_return 
    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return

    for seed_folder in os.listdir(folder):
        seed_path = os.path.join(folder, seed_folder, 'csv')
        csv_file = os.path.join(seed_path, 'eval_data.csv')
        
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            min_return = data['average_return'].min()
            max_return = data['average_return'].max()
            global_min_return = min(global_min_return, min_return)
            global_max_return = max(global_max_return, max_return)
        else:
            print(f"CSV file not found: {csv_file}")

def normalize(value):
    value = (value - global_min_return)/(global_max_return-global_min_return)
    return value.round(6)

def normalize_experiment(exp_path, new_exp_path):
    if not os.path.exists(exp_path):
        print(f"Folder does not exist: {exp_path}")
        return

    for seed_folder in os.listdir(exp_path):
        seed_path = os.path.join(exp_path, seed_folder, 'csv')
        csv_file = os.path.join(seed_path, 'eval_data.csv')
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            data['average_return'] = data['average_return'].apply(normalize)
            output_csv_file = f'eval_data_normalalized_{seed_folder.strip()[-1] if len(seed_folder)==5 else seed_folder.strip()[-2:]}.csv'
            os.makedirs(f'data/{new_exp_path}', exist_ok=True)
            data.to_csv(f'data/{new_exp_path}/{output_csv_file}', index=False)








for folder in folders:
    extract_min_max_from_folder(folder)



print(f"Global Min Return: {global_min_return}")
print(f"Global Max Return: {global_max_return}")

normalize_experiment(normal, 'FetchReach-v2')
normalize_experiment(v7_3, 'FetchReach-LLM-3-v7')
normalize_experiment(v8_4, 'FetchReach-LLM-4-v8')
normalize_experiment(v11_3, 'FetchReach-LLM-3-v9')
normalize_experiment(v12_4, 'FetchReach-LLM-4-v10')


