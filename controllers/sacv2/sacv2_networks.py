import torch
import torch.nn as nn
from torch.distributions import Normal


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class QNetwork(nn.Module):
    """
    Action value function.
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Initialize soft q network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param hidden_dim: int
            hidden layer dimension
        """

        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, x):
        """
        Calculate the action value for state-action pairs.

        @param x: torch.float32 tensor with shape torch.Size([batch_size, state_dim + action_dim])
            state and action of the environment

        @return x: torch.float32 tensor with shape torch.Size([batch_size, 1])
            action value of the given state-action pair
        """

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class TwinnedQNetwork(nn.Module):
    """
    Action value function.
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Initialize twinned soft q network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param hidden_dim: int
            hidden layer dimension
        """

        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.Q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        """
        Calculate the action values for state-action pairs.

        @param state: torch.float32 tensor with shape torch.Size([batch_size, state_dim])
            state of the environment
        @param action: torch.float32 tensor with shape torch.Size([batch_size, action_dim])
            action selected by the agent

        @return q1, q2: torch.float32 tensors with shape torch.Size([batch_size, 1])
            action values of the given state-action pair
        """

        x = torch.cat([state, action], 1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)

        return q1, q2


class GaussianPolicyNetwork(nn.Module):
    """
    Agent policy.
    """

    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    epsilon = 1e-6

    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Initialize policy network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param hidden_dim: int
            hidden layer dimension
        """

        super(GaussianPolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(init_weights)

    def forward(self, state):
        """
        Calculate the mean and log standard deviation of the policy distribution.

        @param state: torch.float32 tensor with shape torch.Size([1, state_dim]) or torch.Size([batch_size, state_dim])
            state of the environment

        @return (mean, log_std):
        mean: torch.float32 tensor with shape torch.Size([1, action_dim]) or torch.Size([batch_size, action_dim])
            mean of the policy distribution
        log_std: torch.float32 tensor with shape torch.Size([1, action_dim]) or torch.Size([batch_size, action_dim])
            log standard deviation of the policy distribution
        """

        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        """
        Sample an action using the reparameterization trick:
        - sample noise from a normal distribution,
        - multiply it with the standard deviation of the policy distribution,
        - add it to the mean of the policy distribution, and
        - apply the tanh function to the result.

        @param state: torch.float32 tensor with shape torch.Size([1, state_dim]) or torch.Size([batch_size, state_dim])
            state of the environment

        @return action, log_prob, mean, log_std:
        action: torch.float32 tensor with shape torch.Size([1, action_dim]) or torch.Size([batch_size, action_dim])
            (normalized) action selected by the agent
        log_prob: torch.float32 tensor with shape torch.Size([1, 1]) or torch.Size([batch_size, 1])
            log probability of the action
        mean: torch.float32 tensor with shape torch.Size([1, action_dim]) or torch.Size([batch_size, action_dim])
            mean of the policy distribution
        """

        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean
