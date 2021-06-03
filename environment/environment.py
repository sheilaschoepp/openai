import os
import pickle

import gym
from utils.rl_glue import BaseEnvironment


class Environment(BaseEnvironment):
    """
    OpenAI environment.

    The environment class implements the dynamics of the task and generates the observations and rewards.  -Brian Tanner & Adam White
    """

    def __init__(self, env_name, seed, render=False):
        """
        Initialize environment variables.

        @param env_name: string
            name of OpenAI environment
        @param seed: int
            seed for random number generator
        @param render: bool
            if true, render experiment
        """

        super(Environment, self).__init__()
        
        self.env_name = env_name
        self.render = render

        self.env = gym.wrappers.FlattenObservation(gym.make(self.env_name))
        self.env_seed(seed)

    def env_init(self):
        """
        Initialize the environment variables that you want to reset before starting a new run.
        """

        pass

    def env_start(self):
        """
        Initialize the variables that you want to reset before starting a new episode.
        Determine the initial state.

        @return state: float64 numpy array with shape (state_dim,)
            state of the environment
        """

        state = self.env.reset()

        if self.render:
            self.env.render()

        return state

    def env_step(self, action):
        """
        A step taken by the environment.

        At the start of a run, some number of random actions may be executed as defined by the 'random_steps' argument.
        After all random steps are taken, the agent's selected action is executed.  When an action is executed, the
        resultant reward, next_state, and terminal status are returned.

        @param action: float64 numpy array with shape (action_dim,)
            action selected by the agent

        @return (reward, next_state, terminal):
        reward: float64
            reward received for taking action
        next_state: float64 numpy array with shape (state_dim,)
            the state of the environment after taking action
        terminal: boolean
            true if the goal state has been reached; otherwise false
        """

        if self.render:
            self.env.render()

        next_state, reward, terminal, _ = self.env.step(action)

        return reward, next_state, terminal

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

    def env_close(self):
        """
        Close the environment.
        """

        self.env.close()

    def env_load(self, dir_):
        """
        Load environment's data.

        @param dir_: string
            load directory
        """

        self.env_load_rng(dir_)

    def env_load_rng(self, dir_):
        """
        Load the state of the random number generators used by OpenAI Gym:
        self.env.np_random and self.env.action_space.np_random

        File format: .pickle

        @param dir_: string
            load directory
        """

        pickle_foldername = dir_ + "/pickle"

        with open(pickle_foldername + "/env_np_random_state.pickle", "rb") as handle:
            env_np_random_state = pickle.load(handle)

        with open(pickle_foldername + "/env_action_space_np_random_state.pickle", "rb") as handle:
            env_action_space_np_random_state = pickle.load(handle)

        self.env.np_random.set_state(env_np_random_state)
        self.env.action_space.np_random.set_state(env_action_space_np_random_state)

    def env_save(self, dir_):
        """
        Save environment's data.

        @param dir_: string
            save directory
        """

        self.env_save_rng(dir_)

    def env_save_rng(self, dir_):
        """
        Save the state of the random number generators used by OpenAI Gym:
        self.env.np_random and self.env.action_space.np_random

        File format: .pickle

        @param dir_: string
            save directory
        """

        env_np_random_state = self.env.np_random.get_state()
        env_action_space_np_random_state = self.env.action_space.np_random.get_state()

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(pickle_foldername + "/env_np_random_state.pickle", "wb") as f:
            pickle.dump(env_np_random_state, f)

        with open(pickle_foldername + "/env_action_space_np_random_state.pickle", "wb") as f:
            pickle.dump(env_action_space_np_random_state, f)

    def env_seed(self, seed):
        """
        Seed the environment.

        @param seed: int
            seed for random number generator
        """

        self.env.seed(seed)
        self.env.action_space.np_random.seed(seed)  # to ensure deterministic sampling from action space

    def env_state_dim(self):
        """
        Retrieve the state dimension.

        @return state_dim: int
            the state dimension
        """

        state_dim = self.env.observation_space.shape[0]

        return state_dim
