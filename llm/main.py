import os
from dotenv import load_dotenv
from function_retriever import FunctionRetriever
import time

# Load environment variables from the .env file
load_dotenv()

if __name__ == "__main__":
    f = open('fault.txt', 'r')
    robot_info = f.read()
    f.close()

    api_key = os.getenv("OPENAI_API_KEY")
    
    prompt_for_expert = """
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

        In the faulty case, the robot is experiencing a Position Slippage fault at the Elbow Flex Joint. This problem is caused when gear teeth break, leading to operational issues as the gear slips
        to the next non-broken tooth, causing inaccurate gear movements. Specifically, if the joint position is expected to change by x radians, this fault would cause the joint 
        position to move by x+c radians, where c is a constant noise value of 0.05 radians. Give a reward function that considers this fault and would make fetchreach'
        reach learning convergence in the least amount of time. Do not give any explaination, just return the reward function without markdown. You are only allowed to use the given sample 
        parameters for the new reward function. The name of the function should be compute_reward. You are expected to use all the knowledge in expert reward shaping and it's related techniques.
        This is the format for the info parameter:
        info = {{
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }}
        where 
        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        
        def _get_obs(self):
                # positions
                grip_pos = self.sim.data.get_site_xpos('robot0:grip')
                dt = self.sim.nsubsteps * self.sim.model.opt.timestep
                grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
                robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
                if self.has_object:
                    object_pos = self.sim.data.get_site_xpos('object0')
                    # rotations
                    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
                    # velocities
                    object_velp = self.sim.data.get_site_xvelp('object0') * dt
                    object_velr = self.sim.data.get_site_xvelr('object0') * dt
                    # gripper state
                    object_rel_pos = object_pos - grip_pos
                    object_velp -= grip_velp
                else:
                    object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
                gripper_state = robot_qpos[-2:]
                gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

                if not self.has_object:
                    achieved_goal = grip_pos.copy()
                else:
                    achieved_goal = np.squeeze(object_pos.copy())
                obs = np.concatenate([
                    grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                    object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel
                ])

                return {{
                    'observation': obs.copy(),
                    'achieved_goal': achieved_goal.copy(),
                    'desired_goal': self.goal.copy(),
                }}
        
        def goal_distance(goal_a, goal_b):
            assert goal_a.shape == goal_b.shape
            return np.linalg.norm(goal_a - goal_b, axis=-1)
        
        Only return one function and incorporate your whole idea into it. You are allowed to use numpy as np. No need to make any imports or use anything else not mentioned.
        The following is the faulty robot representation in xml being used in Mujoco. The robot fault modification is mentioned at the top:
        {}
    """
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

        In the faulty case, the robot is experiencing a Position Slippage fault at the Elbow Flex Joint. This problem is caused when gear teeth break, leading to operational issues as the gear slips
        to the next non-broken tooth, causing inaccurate gear movements. Specifically, if the joint position is expected to change by x radians, this fault would cause the joint 
        position to move by x+c radians, where c is a constant noise value of 0.05 radians. Give a reward function that considers this fault and would make fetchreach'
        reach learning convergence in the least amount of time. Do not give any explaination, just return the reward function without markdown. You are only allowed to use the given sample 
        parameters for the new reward function. The name of the function should be compute_reward. You are expected to use all the knowledge in expert reward shaping and it's related techniques.
        This is the format for the info parameter:
        info = {{
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }}
        where 
        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)

        def _get_obs(self):
                # positions
                grip_pos = self.sim.data.get_site_xpos('robot0:grip')
                dt = self.sim.nsubsteps * self.sim.model.opt.timestep
                grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
                robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
                if self.has_object:
                    object_pos = self.sim.data.get_site_xpos('object0')
                    # rotations
                    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
                    # velocities
                    object_velp = self.sim.data.get_site_xvelp('object0') * dt
                    object_velr = self.sim.data.get_site_xvelr('object0') * dt
                    # gripper state
                    object_rel_pos = object_pos - grip_pos
                    object_velp -= grip_velp
                else:
                    object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
                gripper_state = robot_qpos[-2:]
                gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

                if not self.has_object:
                    achieved_goal = grip_pos.copy()
                else:
                    achieved_goal = np.squeeze(object_pos.copy())
                obs = np.concatenate([
                    grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                    object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel
                ])

                return {{
                    'observation': obs.copy(),
                    'achieved_goal': achieved_goal.copy(),
                    'desired_goal': self.goal.copy(),
                }}
        
        def goal_distance(goal_a, goal_b):
            assert goal_a.shape == goal_b.shape
            return np.linalg.norm(goal_a - goal_b, axis=-1)

        The following is an expert reward funciton for the faulty case:
        {}
        Your task if to make a better adapting reward function for this faulty case.
        Only return one function and incorporate your whole idea into it. You are allowed to use numpy as np. No need to make any imports or use anything else not mentioned.
        The following is the faulty robot representation in xml being used in Mujoco. The robot fault modification is mentioned at the top:
        {}
        """
    
    prompt_without_reward_func = """
                You are a reinforcement learning expert who has full knowledge on the working of fetchreach and similar robots.
        
        A fetch reach robot is learning to do a particular task using the Proximal Policy Optimization algorithm which is for the reacher's hand to reach a particular point.
        The arguments that are provided to make the function are as follows:
        achieved_goal: NDArray[float64]
        distance_threshold (float): the threshold after which a goal is considered achieved
        info
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
    
        The goal is to make the robot learn to do the same task in case there is a fault in it.

        In the faulty case, the robot is experiencing a Position Slippage fault at the Elbow Flex Joint. This problem is caused when gear teeth break, leading to operational issues as the gear slips
        to the next non-broken tooth, causing inaccurate gear movements. Specifically, if the joint position is expected to change by x radians, this fault would cause the joint 
        position to move by x+c radians, where c is a constant noise value of 0.05 radians. Give a reward function that considers this fault and would make fetchreach
        reach learning convergence in the least amount of time. Do not give any explaination, just return the reward function in python without markdown. You are only allowed to use the given sample 
        parameters for the new reward function. The name of the function should be compute_reward. You are expected to use all the knowledge in expert reward shaping and it's related techniques.
        This is the format for the info parameter:
        info = {{
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }} 
        where 
        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)
        
        def _get_obs(self):
                # positions
                grip_pos = self.sim.data.get_site_xpos('robot0:grip')
                dt = self.sim.nsubsteps * self.sim.model.opt.timestep
                grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
                robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
                if self.has_object:
                    object_pos = self.sim.data.get_site_xpos('object0')
                    # rotations
                    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
                    # velocities
                    object_velp = self.sim.data.get_site_xvelp('object0') * dt
                    object_velr = self.sim.data.get_site_xvelr('object0') * dt
                    # gripper state
                    object_rel_pos = object_pos - grip_pos
                    object_velp -= grip_velp
                else:
                    object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
                gripper_state = robot_qpos[-2:]
                gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

                if not self.has_object:
                    achieved_goal = grip_pos.copy()
                else:
                    achieved_goal = np.squeeze(object_pos.copy())
                obs = np.concatenate([
                    grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                    object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel
                ])

                return {{
                    'observation': obs.copy(),
                    'achieved_goal': achieved_goal.copy(),
                    'desired_goal': self.goal.copy(),
                }}

        Here are some variables or functions you can access:
        Compute distance between goal and the achieved goal:
        def goal_distance(goal_a, goal_b):
            assert goal_a.shape == goal_b.shape
            return np.linalg.norm(goal_a - goal_b, axis=-1)

        self.reward_type : has values "sparse" or "dense"
        
        Only return one function and incorporate your whole idea into it. You are allowed to use numpy as np. No need to make any imports or use anything else not mentioned.
        The following is the faulty robot representation in xml being used in Mujoco. The robot fault modification is mentioned at the top:
        {}
    """


    retriever = FunctionRetriever(api_key)
    expert_function = retriever.retrieve_and_verify_function_expert(prompt_for_expert.format(robot_info))
    function_code_v1 = retriever.retrieve_and_verify_function(prompt_with_reward_func.format(expert_function, robot_info))
    function_code_v2 = retriever.retrieve_and_verify_function(prompt_without_reward_func.format(robot_info))

    main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    relative_path_to_copy = 'custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint'
    new_directory_name_v1 = 'FetchReachEnv_v1_BrokenElbowFlexJoint_LLM_v1'
    new_directory_name_v2 = 'FetchReachEnv_v1_BrokenElbowFlexJoint_LLM_v2'
    expert_directory_name = 'FetchReachEnv_v1_BrokenElbowFlexJoint_Expert'


    new_file_path_v1 = f'/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/{new_directory_name_v1}/fetch_env.py'
    new_file_path_v2 = f'/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/{new_directory_name_v2}/fetch_env.py'
    expert_path = f'/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/{expert_directory_name}/fetch_env.py'
    method_name = 'compute_reward'

    src_directory = os.path.join(main_project_dir, relative_path_to_copy)


    retriever.copy_env(src_directory, new_directory_name_v1)
    retriever.replace_method_in_class(new_file_path_v1, 'FetchEnv', method_name, function_code_v1)


    retriever.copy_env(src_directory, new_directory_name_v2)
    retriever.replace_method_in_class(new_file_path_v2, 'FetchEnv', method_name, function_code_v2)

    retriever.copy_env(src_directory, expert_directory_name)
    retriever.replace_method_in_class(expert_path, 'FetchEnv', method_name, expert_function)