"""
Glues together an experiment, agent, and environment.
"""

from abc import ABCMeta, abstractmethod


class RLGlue:
    """
    Facilitates interaction between an agent and environment for reinforcement learning experiments.

    The RL-Glue program mediates teh communication between the agent and environment programs in response to commands
    from the experiment program.  -Brian Tanner & Adam White

    args:
        env_obj: an object that implements BaseEnvironment
        agent_obj: an object that implements BaseAgent
    """

    def __init__(self, env_obj, agent_obj):
        self._environment = env_obj  # environment
        self._agent = agent_obj  # agent

        # useful statistics
        self._total_reward = None
        self._num_steps = None
        self._num_episodes = None
        self._episode_reward = None
        self._num_ep_steps = None

        self._action = None

    def total_reward(self):
        """
        @return self._total_reward: float64
            amount of reward accumulated in a single run
        """
        return self._total_reward

    def episode_reward(self):
        """
        @return self._episode_reward: float64
            amount of reward accumulated in an episode
        """
        return self._episode_reward

    def num_steps(self):
        """
        @return self._num_steps: int
            number of steps in a single run
        """
        return self._num_steps

    def num_episodes(self):
        """
        @return self._num_episodes: int
            number of episodes in a single run
        """
        return self._num_episodes

    def num_ep_steps(self):
        """
        @return self._num_ep_steps: int
            number of steps in the current episode
        """
        return self._num_ep_steps

    # repeat for each run
    def rl_init(self, total_reward=0, num_steps=0, num_episodes=0):
        """
        Reset experiment data.
        Reset action.
        Reset agent and environment.
        """

        self._total_reward = total_reward  # amount of reward accumulated in a single run
        self._num_steps = num_steps  # number of steps in a single run
        self._num_episodes = num_episodes  # number of episodes in a single run
        self._episode_reward = 0  # amount of reward accumulated in an episode
        self._num_ep_steps = 0  # number of steps in the current episode

        self._action = None

        self._agent.agent_init()
        self._environment.env_init()

    # repeat for each episode
    def rl_start(self):
        """
        Starts RLGlue experiment.

        @return (state, self._action):
        state: float64 numpy array with shape (state_dim,)
            state of the environment
        self._action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        self._episode_reward = 0  # reward accumulated in an episode
        self._num_ep_steps = 0  # number of steps in the current episode
        # self._num_steps = max(self._num_steps, 0)  # number of steps in a run

        self._num_episodes += 1  # moved here from rl_step

        state = self._environment.env_start()
        self._action = self._agent.agent_start(state)

        return state, self._action

    def rl_step(self):
        """
        Takes a step in a RLGlue experiment.

        @return (reward, next_state, terminal, self._action):
        reward: float64
            reward received for taking action
        next_state: float64 numpy array with shape (state_dim,)
            state of the environment
        terminal: boolean
            true if the goal state has been reached after taking action; otherwise false
        self._action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """
        reward, next_state, terminal = self._environment.env_step(self._action)  # returns reward, next_state, done

        self._num_ep_steps += 1
        self._num_steps += 1

        self._total_reward += reward
        self._episode_reward += reward

        if terminal:
            self._action = self._agent.agent_end(reward, next_state, terminal)
            # self._num_episodes += 1  # moved to rl_start to handle the case when we terminate an episode early by reaching max steps
        else:
            self._action = self._agent.agent_step(reward, next_state, terminal)

        return reward, next_state, terminal, self._action

    # repeat for each episode
    def rl_episode(self, max_steps_this_episode=0):
        """
        Runs an episode in a RLGlue experiment.

        @param max_steps_this_episode: int
            the maximum number of steps that can be taken in the current episode (<=0 if no limit on number of steps)
        
        @return terminal: boolean
            true if the goal state has been reached after taking action; otherwise false
        """""
        
        terminal = False

        self.rl_start()

        while not terminal and ((max_steps_this_episode <= 0) or (self._num_ep_steps < max_steps_this_episode)):
            _, _, _, terminal = self.rl_step()

        self._num_episodes += 1

        return terminal

    # CONVENIENCE FUNCTIONS BELOW
    def rl_env_start(self):
        """
        Useful when manually specifying agent actions (for debugging). Starts
        RL-Glue environment.

        Returns:
            state observation
        """
        self._num_ep_steps = 0

        return self._environment.env_start()

    def rl_env_step(self, action):
        """
        Useful when manually specifying agent actions (for debugging).Takes a
        step in the environment based on an action.

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        reward, state, terminal = self._environment.env_step(action)

        self._total_reward += reward

        if terminal:
            self._num_episodes += 1
        else:
            self._num_ep_steps += 1
            self._num_steps += 1

        return reward, state, terminal

    def rl_agent_message(self, message):
        """
        pass a message to the agent

        Args:
            message (str): the message to pass

        returns:
            str: the agent's response
        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_agent_response = self._agent.agent_message(message_to_send)
        if the_agent_response is None:
            the_agent_response = ""

        return the_agent_response

    def rl_env_message(self, message):
        """
        pass a message to the environment

        Args:
            message (str): the message to pass

        Returns:
            the_env_response (str) : the environment's response
        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_env_response = self._environment.env_message(message_to_send)
        if the_env_response is None:
            return ""

        return the_env_response


class BaseAgent:
    """
    Defines the interface of an RLGlue Agent

    ie. These methods must be defined in your own Agent classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Declare agent variables."""
        pass

    @abstractmethod
    def agent_init(self):
        """Initialize agent variables."""

    @abstractmethod
    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, next_state, terminal):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            next_state (state observation): the agent's current state
            terminal (boolean): boolean indicating if the goal state has ben reached
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward, next_state, terminal):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward received for entering the terminal state
            next_state (state observation): the agent's current state
            terminal (boolean): boolean indicating if the goal state has ben reached
        """

    @abstractmethod
    def agent_message(self, message):
        """
        receive a message from rlglue
        args:
            message (str): the message passed
        returns:
            str : the agent's response to the message (optional)
        """


class BaseEnvironment:
    """
    Defines the interface of an RLGlue environment

    ie. These methods must be defined in your own environment classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Declare environment variables."""

    @abstractmethod
    def env_init(self):
        """
        Initialize environment variables.
        """

    @abstractmethod
    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    @abstractmethod
    def env_message(self, message):
        """
        receive a message from RLGlue
        Args:
           message (str): the message passed
        Returns:
           str: the environment's response to the message (optional)
        """
