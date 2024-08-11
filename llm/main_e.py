import os
from dotenv import load_dotenv
from function_retriever import FunctionRetriever
import time

load_dotenv()

if __name__ == "__main__":
    f = open('new_prompt.txt', 'r')
    main_prompt = f.read()
    f.close()

    api_key = os.getenv("OPENAI_API_KEY")

    retriever = FunctionRetriever(api_key)
    function_code_3 = retriever.retrieve_and_verify_function_expert(main_prompt)
    function_code_4 = retriever.retrieve_and_verify_function(main_prompt)
    # function_code_v2 = retriever.retrieve_and_verify_function(prompt_without_reward_func.format(robot_info))

    main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    relative_path_to_copy = 'custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint'
    new_directory_name_v1 = 'FetchReachEnv_v1_BrokenElbowFlexJoint_LLM_3e_v1'
    new_directory_name_v2 = 'FetchReachEnv_v1_BrokenElbowFlexJoint_LLM_4e_v2'


    new_file_path_v1 = f'/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/{new_directory_name_v1}/fetch_env.py'
    new_file_path_v2 = f'/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/{new_directory_name_v2}/fetch_env.py'
    method_name = 'compute_reward'

    src_directory = os.path.join(main_project_dir, relative_path_to_copy)


    retriever.copy_env(src_directory, new_directory_name_v1)
    retriever.replace_method_in_class(new_file_path_v1, 'FetchEnv', method_name, function_code_3)

    retriever.copy_env(src_directory, new_directory_name_v2)
    retriever.replace_method_in_class(new_directory_name_v2, 'FetchEnv', method_name, function_code_4)