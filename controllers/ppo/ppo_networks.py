import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal


def init_weights(m):
    if type(m) == nn.Linear:
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class ValueNetwork(nn.Module):
    """
    Value network that predicts the value of a given state.
    """

    def __init__(self, state_dim, hidden_dim):
        """
        Initialize value network.

        @param state_dim: int
            environment state dimension
        @param hidden_dim: int
            hidden layer dimension
        """

        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(init_weights)

    def forward(self, state):
        """
        Calculate the value of a state.

        @param state: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([num_samples + 1, state_dim])
            or torch.Size([mini_batch_size, 1]) representing the state
            of the environment

        @return value: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([num_samples + 1, 1]) or
            torch.Size([mini_batch_size, 1]) representing the value of
            the given state
        """

        value = torch.tanh(self.linear1(state))
        value = torch.tanh(self.linear2(value))
        value = self.linear3(value)

        return value


class PolicyNetwork(nn.Module):
    """
    Policy network that defines the agent's action distribution
    for a given state.
    """

    def __init__(self, state_dim, action_dim, hidden_dim, log_std):
        """
        Initialize policy network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param hidden_dim: int
            hidden layer dimension
        @param log_std: float
            log standard deviation of the policy distribution
        """

        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(init_weights)

        self.log_std = nn.Parameter(torch.full((action_dim,), log_std))

    def forward(self, state):
        """
        Calculate the mean of policy.

        @param state: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([1, state_dim]) or
            torch.Size([mini_batch_size, state_dim]) representing the
            state of the environment

        @return mean: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([1, action_dim]) or
            torch.Size([mini_batch_size, action_dim]) representing the
            mean of the policy distribution
        """

        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))

        mean = self.mean_linear(x)

        return mean


class ActorCriticNetwork(nn.Module):
    """
    Actor Critic (Policy and Value Function).
    """

    def __init__(self, state_dim, action_dim, hidden_dim, log_std):
        """
        Initialize actor-critic network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param hidden_dim: int
            hidden layer dimension
        @param log_std: float
            log standard deviation of the policy distribution
        """

        super(ActorCriticNetwork, self).__init__()

        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim, log_std)
        self.value_network = ValueNetwork(state_dim, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def evaluate(self, state, action):
        """
        Evaluate:
        - calculate the log probability of the action in the given state,
        - calculate the entropy of the policy distribution, and
        - calculate the value of the given state.

        Note: torch.diag_embed works on one or more samples.

        @param state: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([mini_batch_size, state_dim])
            representing the state of the environment
        @param action: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([mini_batch_size, action_dim])
            representing the action selected by the agent

        @return (log_prob, entropy):
        log_prob: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([mini_batch_size, 1])
            representing the log probability of an action
        entropy: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([mini_batch_size, 1])
            representing the entropy of the policy
        """

        # mean = self.policy_network(state)
        #
        # std = torch.exp(self.policy_network.log_std)
        # variance = std.pow(2).expand_as(mean)
        # covariance = torch.diag_embed(variance)
        #
        # mv_normal = MultivariateNormal(mean, covariance)
        #
        # log_prob = mv_normal.log_prob(action).unsqueeze(1)
        # entropy = mv_normal.entropy().unsqueeze(1)

        mean = self.policy_network(state)

        std = torch.exp(self.policy_network.log_std)

        normal = Normal(mean, std)

        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        entropy = normal.entropy()
        entropy = entropy.sum(1, keepdim=True)

        return log_prob, entropy

    def value(self, state):
        """
        Calculate the value of a state.

        @param state: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([num_samples + 1, state_dim])
            or torch.Size([mini_batch_size, 1]) representing the state
            of the environment

        @return value: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([num_samples + 1, 1]) or
            torch.Size([mini_batch_size, 1]) representing the value of
            the given state
        """

        value = self.value_network(state)

        return value

    def sample(self, state, deterministic=False):
        """
        Sample action from the policy distribution.

        Note: torch.diag only works on a single sample.

        @param state: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([1, state_dim])
            representing the state of the environment
        @param deterministic: bool
            if True, use the mean of the policy distribution for action
            selection (deterministic); if False, sample stochastically
            from the policy distribution

        @return (action, log_prob):
        action: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([1, action_dim])
            representing the action selected by the agent
        log_prob: torch.Tensor (torch.float32)
            a tensor with shape torch.Size([1, 1]) representing the
            log probability of the action selected by the agent
        """

        # mean = self.policy_network(state)
        #
        # std = torch.exp(self.policy_network.log_std)
        # variance = std.pow(2)
        # covariance = torch.diag(variance)
        #
        # mv_normal = MultivariateNormal(mean, covariance)
        #
        # if not deterministic:
        #     action = mv_normal.sample()
        # else:
        #     action = mean
        #
        # log_prob = mv_normal.log_prob(action).unsqueeze(0)

        mean = self.policy_network(state)

        std = torch.exp(self.policy_network.log_std)

        normal = Normal(mean, std)

        if not deterministic:
            action = normal.sample()
        else:
            action = mean

        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
