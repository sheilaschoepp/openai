import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import os
import pickle

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from utils.rl_glue import BaseEnvironment


class Environment(BaseEnvironment):
    """
    OpenAI environment.

    The environment class implements the dynamics of the task and
    generates the observations and rewards. -Brian Tanner & Adam White
    """

    def __init__(self, env_name, seed, render=False):
        """
        Initialize environment variables.

        @param env_name: str
            name of OpenAI environment
        @param seed: int
            seed for random number generator
        @param render: bool
            if true, render experiment
        """

        super(Environment, self).__init__()

        self.env_name = env_name
        self.seed = seed

        render_mode = None
        if render:
            render_mode = "human"

        if "Ant" in env_name:
            self.env = gym.make(env_name, render_mode=render_mode)
        elif "Fetch" in env_name:
            gym.register_envs(gymnasium_robotics)
            self.env = gym.make(env_name, render_mode=render_mode, reward_type="dense")
            self.env = CustomFetchFlattenObservation(self.env)
        else:
            exit("Environment not supported.")


    def env_init(self):
        """
        Initialize the environment variables that you want to reset
        before starting a new run.
        """

        observation, info = self.env.reset(seed=self.seed)
        self.env.action_space.seed(seed=self.seed)

    def env_start(self):
        """
        Initialize the variables that you want to reset before starting
        a new episode.

        @return observation: np.ndarray
            a float64 numpy array with shape (observation_dim,)
            representing the observation of the environment
        """

        observation, info = self.env.reset()

        return observation

    def env_step(self, action):
        """
        A step taken by the environment.

        @param action: np.ndarray
            a float64 numpy array with shape (action_dim,) representing
            the action selected by the agent

        @return (reward, observation, terminated):
        reward: np.ndarray
            a float64 representing the reward received for taking action
        observation: np.darray
            a float64 numpy array with shape (observation_dim,)
            representing the observation of the environment after
            taking an action
        terminated: boolean
            true if the goal has been reached; otherwise false
        """

        observation, reward, terminated, truncated, info = self.env.step(action)

        terminal = terminated or truncated

        return reward, observation, terminal

    def env_message(self, message):
        """
        Receive a message from RLGlue.

        @param message: str
            the message passed

        @return response: str
            the agent's response to the message (optional)
        """

        split_message = message.split(", ")
        response = ""

        if split_message[0] == "close":
            self.env_close()
        if split_message[0] == "load":
            data_dir = split_message[1]
            self.env_load(data_dir)
        if split_message[0] == "save":
            data_dir = split_message[1]
            self.env_save(data_dir)

        return response

    def env_action_dim(self):
        """
        Retrieve the action dimension.

        @return action_dim: int
            the action dimension
        """

        action_dim = self.env.action_space.shape[0]

        return action_dim

    def env_observation_dim(self):
        """
        Retrieve the observation dimension.

        @return observation_dim: int
            the observation dimension
        """

        observation_dim = self.env.observation_space.shape[0]

        return observation_dim

    def env_close(self):
        """
        Close the environment.
        """

        self.env.close()

    def env_load(self, dir_):
        """
        Restore the environment's state and random number generator
        (RNG) settings to reproduce a previously saved environment
        configuration.

        @param dir_: str
            load directory
        """

        self.env_load_state(dir_)
        self.env_load_rng(dir_)

    def env_load_state(self, dir_):
        """
        Restore the state of the MuJoCo simulator, including the saved
        observation and the simulator's position and velocity data
        (qpos and qvel), to reproduce a previously saved environment
        state.

        File format: .pickle

        @param dir_: str
            load directory
        """

        pickle_foldername = dir_ + "/pickle"

        with open(pickle_foldername + "/mujoco_qpos.pickle", "rb") as handle:
            mujoco_qpos = pickle.load(handle)
        with open(pickle_foldername + "/mujoco_qvel.pickle", "rb") as handle:
            mujoco_qvel = pickle.load(handle)

        # set the simulator's internal state to the saved state
        self.env.unwrapped.data.qpos[:] = mujoco_qpos
        self.env.unwrapped.data.qvel[:] = mujoco_qvel

        # advance the simulation to apply the changes
        mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)

    def env_load_rng(self, dir_):
        """
        Restore the state of the Gymnasium environment's random number
        generators (RNGs) to ensure reproducible results.

        Loads the saved RNG states for both the main environment and its
        action space, allowing for consistent sampling of actions and
        environment dynamics.

        File format: .pickle

        @param dir_: str
            load directory
        """

        pickle_foldername = dir_ + "/pickle"

        with open(pickle_foldername + "/env_np_random_state.pickle", "rb") as handle:
            env_np_random_state = pickle.load(handle)
        with open(pickle_foldername + "/env_action_space_np_random_state.pickle", "rb") as handle:
            env_action_space_np_random_state = pickle.load(handle)

        self.env.np_random.bit_generator.state = env_np_random_state
        self.env.action_space.np_random.bit_generator.state = env_action_space_np_random_state

    def env_save(self, dir_):
        """
        Save the current state of the environment to ensure
        reproducibility.

        This method saves both the simulator's physical state (such as
        positions and velocities) and the random number generator (RNG)
        states used by the environment.

        @param dir_: str
            save directory
        """

        self.env_save_state(dir_)
        self.env_save_rng(dir_)

    def env_save_state(self, dir_):
        """
        Save the current state of the MuJoCo simulator, including the
        observation, and the simulator's position and velocity data
        (qpos and qvel).

        This method captures:
        - The current observation as perceived by the environment.
        - qpos: The simulator's internal state of positions.
        - qvel: The simulator's internal state of velocities.

        File format: .pickle

        @param dir_: str
            save directory
        """

        mujoco_qpos = self.env.unwrapped.data.qpos.copy()
        mujoco_qvel = self.env.unwrapped.data.qvel.copy()

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(pickle_foldername + "/mujoco_qpos.pickle", "wb") as f:
            pickle.dump(mujoco_qpos, f)
        with open(pickle_foldername + "/mujoco_qvel.pickle", "wb") as f:
            pickle.dump(mujoco_qvel, f)

    def env_save_rng(self, dir_):
        """
        Save the state of the random number generators (RNGs) used by
        the Gymnasium environment, including the environment's main RNG
        and the action space's RNG.

        File format: .pickle

        @param dir_: str
            save directory
        """

        env_np_random_state = self.env.np_random.bit_generator.state
        env_action_space_np_random_state = self.env.action_space.np_random.bit_generator.state

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(pickle_foldername + "/env_np_random_state.pickle", "wb") as f:
            pickle.dump(env_np_random_state, f)
        with open(pickle_foldername + "/env_action_space_np_random_state.pickle", "wb") as f:
            pickle.dump(env_action_space_np_random_state, f)


class CustomFetchFlattenObservation(ObservationWrapper):
    def __init__(self, env):

        super().__init__(env)

        self.keys_order = ["observation", "desired_goal", "achieved_goal"]

        # Build a new single Box space from the chosen key order.
        low_list, high_list = [], []

        for key in self.keys_order:
            space = self.env.observation_space.spaces[key]
            low_list.append(space.low.flatten())
            high_list.append(space.high.flatten())

        self.observation_space = Box(
            low=np.concatenate(low_list),
            high=np.concatenate(high_list),
            dtype=self.env.observation_space.spaces[self.keys_order[0]].dtype
        )

    def observation(self, observation):
        # Concatenate in the custom order
        return np.concatenate(
            [observation[key].flatten() for key in self.keys_order],
            axis=0
        )

# Example usage:
# env = gym.make("YourEnv-v0")
# wrapped_env = CustomFlattenObservation(env, ["observation", "desired_goal", "achieved_goal"])
# obs = wrapped_env.reset()
# print(obs)