import random

import numpy as np


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, capacity):
        """
        Initialize or set replay buffer attributes.

        @param capacity: int
            the capacity of the replay buffer
        """

        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, terminal):
        """
        Store a tuple in the buffer.  If the buffer is at maximum capacity, replace the oldest tuple in the
        buffer with the new tuple.

        @param state: float64 numpy array with shape (state_dim,)
            state of the environment
        @param action: torch.float32 tensor with shape torch.Size([action_dim])
            action selected by the agent
        @param reward: float64
            reward received for taking action
        @param next_state: float64 numpy array with shape (state_dim,)
            the state of the environment after taking action
        @param terminal: boolean
            true if the goal state has been reached after taking action; otherwise false
        """

        data = (state.astype("float32"), action.astype("float32"), np.array([reward]).astype("float32")[0], next_state.astype("float32"), np.array([terminal]).astype("float32")[0])

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample from the buffer.

        @param batch_size: int
            the number of samples to randomly draw from the buffer

        @return (state, action, reward, next_state, terminal):
        states: float32 numpy array with shape (batch_size, state_dim)
            state of the environment
        actions: float32 numpy array with shape (batch_size, action_dim)
            action selected by the agent
        rewards: float32 numpy array with shape (batch_size,)
            reward received for taking action
        next_states: float32 numpy array with shape (batch_size, state_dim)
            the state of the environment after taking action
        terminals: float32 numpy array with shape (batch_size,)
            1.0 (true) if the goal state has been reached after taking action; otherwise 0.0 (false)
        """

        indexes = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]

        states, actions, rewards, next_states, terminals = [], [], [], [], []

        for i in indexes:
            data = self.buffer[i]
            state, action, reward, next_state, terminal = data
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            terminals.append(terminal)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminals)

    def __len__(self):
        """
        Retrieve the length of the replay buffer.
        """

        return len(self.buffer)
