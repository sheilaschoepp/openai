import os
from dotenv import load_dotenv
from function_retriever import FunctionRetriever

# Load environment variables from the .env file
load_dotenv()

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    prompt_with_reward_func = """
        You are a reinforcement learning expert who has full knowledge on the working of fetchreach and similar robots.
        
        A fetch reach robot is learning to do a particular task using the Proximal Policy Optimization algorithm which is for the reacher's hand to reach a particular point.
        We already have a reward function defined that makes the robot learn to do that task in a normal environment:
        achieved_goal: NDArray[float64]
        distance_threshold (float): the threshold after which a goal is considered achieved
            def compute_reward(self, achieved_goal, goal, info):
                # Compute distance between goal and the achieved goal.
                d = goal_distance(achieved_goal, goal)
                if self.reward_type == 'sparse':
                    return -(d > self.distance_threshold).astype(np.float32)
                else:
                    return -d
        Our aim is to make the robot learn to do the same task in case there is a fault in it. You can assume that the reward function we are currently using is a standard dense
        reward function to make the fetchreach's tip reach a certain point in space.

        In the faulty case, the robot is experiencing a Position Slippage fault at the Shoulder Lift Joint. This problem is caused when gear teeth break, leading to operational issues as the gear slips
        to the next non-broken tooth, causing inaccurate gear movements. Specifically, if the joint position is expected to change by x radians, this fault would cause the joint 
        position to move by x+c radians, where c is a constant noise value of, in this case, 0.05 radians. Give a reward function that considers this fault and would make fetchreach'
        reach learning convergence in the least amount of time. Do not give any explaination, just return the reward function without markdown. You are only allowed to use the given sample 
        parameters for the new reward function. The name of the function should be compute_reward. You are expected to use all the knowledge in expert reward shaping and it's related techniques.
        This is the format for the info parameter:
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } 
        where 
        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        
        Only return one function and incorporate your whole idea into it.
        """
    
    prompt_without_reward_func = """
                You are a reinforcement learning expert who has full knowledge on the working of fetchreach and similar robots.
        
        A fetch reach robot is learning to do a particular task using the Proximal Policy Optimization algorithm which is for the reacher's hand to reach a particular point.
        The arguments that are provided to make the function are as follows:
        achieved_goal: NDArray[float64]
        distance_threshold (float): the threshold after which a goal is considered achieved
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } 
    
        The goal is to make the robot learn to do the same task in case there is a fault in it.

        In the faulty case, the robot is experiencing a Position Slippage fault at the Shoulder Lift Joint. This problem is caused when gear teeth break, leading to operational issues as the gear slips
        to the next non-broken tooth, causing inaccurate gear movements. Specifically, if the joint position is expected to change by x radians, this fault would cause the joint 
        position to move by x+c radians, where c is a constant noise value of, in this case, 0.05 radians. Give a reward function that considers this fault and would make fetchreach
        reach learning convergence in the least amount of time. Do not give any explaination, just return the reward function in python without markdown. You are only allowed to use the given sample 
        parameters for the new reward function. The name of the function should be compute_reward. You are expected to use all the knowledge in expert reward shaping and it's related techniques.
        This is the format for the info parameter:
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } 
        where 
        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        
        Only return one function and incorporate your whole idea into it.
    """

    retriever = FunctionRetriever(api_key)
    function_code = retriever.retrieve_and_verify_function(prompt_with_reward_func)
    main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    relative_path_to_copy = 'custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint'
    new_directory_name = 'FetchReachEnv_v1_BrokenShoulderLiftJoint_LLM_v1'  # Name of the new directory

    src_directory = os.path.join(main_project_dir, relative_path_to_copy)
    retriever.copy_env(src_directory, new_directory_name)

    if function_code:
        print("Function code:")
        print(function_code)