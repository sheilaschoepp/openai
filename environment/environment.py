import gymnasium as gym

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

        @param env_name: string
            name of OpenAI environment
        @param seed: int
            seed for random number generator
        @param render: bool
            if true, render experiment
        """

        super(Environment, self).__init__()

        if render:
            render_mode = "human"
        else:
            render_mode = None

        self.env = gym.make(env_name, render_mode=render_mode)

        observation, info = self.env.reset(seed=seed)

    def env_init(self):
        """
        Initialize the environment variables that you want to reset
        before starting a new run.
        """

        pass

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

        # combine terminated and truncated flags into a single terminal
        # flag
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