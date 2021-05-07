import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Memory:
    """
    Memory.
    """

    def __init__(self, num_samples, state_dim, action_dim, gamma, use_gae, gae_lambda, mini_batch_size, device):
        """
        Initialize memory.

        @param num_samples: int
            number of samples used to update the network(s)
        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param gamma: float
            discount factor
        @param use_gae: bool
            use generalized advantage estimation
        @param gae_lambda: float
            generalized advantage estimation smoothing parameter
        @param mini_batch_size: int
            number of samples per mini-batch
        """

        self.gamma = gamma

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.mini_batch_size = mini_batch_size

        self.states = torch.zeros(num_samples + 1, state_dim).to(device=device)
        self.actions = torch.zeros(num_samples, action_dim).to(device=device)
        self.log_probs = torch.zeros(num_samples, 1).to(device=device)
        self.rewards = torch.zeros(num_samples, 1).to(device=device)
        self.terminals = torch.zeros(num_samples + 1, 1).to(device=device)

        self.values = torch.zeros(num_samples + 1, 1).to(device=device)

        self.returns = torch.zeros(num_samples + 1, 1).to(device=device)
        self.advantages = torch.zeros(num_samples, 1).to(device=device)

        self.num_steps = num_samples
        self.step = 0

    def add(self, state, action, log_prob, reward, terminal):
        """
        Store a tuple in the memory.

        @param state: torch.float32 tensor with shape torch.Size([1, state_dim])
            state of the environment
        @param action: torch.float32 tensor with shape torch.Size([1, action_dim])
            action selected by the agent
        @param log_prob: torch.float32 tensor with shape torch.Size([1, 1])
            log probability of action
        @param reward: float64
            reward received for taking action
        @param terminal: boolean
            true if the goal state has been reached after taking action; otherwise false
        """

        self.states[self.step].copy_(state.squeeze(0))
        self.actions[self.step].copy_(action.squeeze(0))
        self.log_probs[self.step].copy_(log_prob.squeeze(0))
        self.rewards[self.step].copy_(torch.FloatTensor([reward]))
        self.terminals[self.step + 1].copy_(torch.FloatTensor([terminal]))

        self.step = (self.step + 1) % self.num_steps

    def clear(self):
        """
        Clear the memory.

        Note: To clear the memory, we only need to set the step to 0.  All old samples will be overwritten by new samples.
        Note: Filling all tensors with 0.0 is not a required step.
        """

        self.states.fill_(0.0)
        self.actions.fill_(0.0)
        self.log_probs.fill_(0.0)
        self.rewards.fill_(0.0)
        self.terminals.fill_(0.0)

        self.values.fill_(0.0)

        self.returns.fill_(0.0)
        self.advantages.fill_(0.0)

        self.step = 0

    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages.
        Save to memory.
        """

        # compute returns

        if self.use_gae:

            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = self.rewards[step] + (1 - self.terminals[step + 1]) * self.gamma * self.values[step + 1] - self.values[step]
                gae = delta + (1 - self.terminals[step + 1]) * self.gamma * self.gae_lambda * gae
                self.returns[step] = gae + self.values[step]

        else:

            self.returns[-1].copy_(self.values[-1])
            for step in reversed(range(self.num_steps)):
                self.returns[step] = self.rewards[step] + (1 - self.terminals[step + 1]) * self.gamma * self.returns[step + 1]

        # compute (and normalize) advantages

        self.advantages = self.returns[:-1] - self.values[:-1]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

    def generate_mini_batches(self):
        """
        Generate mini-batches from the samples in memory.

        @return (states_batch, actions_batch, log_probs_batch, returns_batch, advantages_batch):
        states_batch: torch.float32 tensor with shape torch.Size([self.mini_batch_size, state_dim])
            state of the environment
        actions_batch: torch.float32 tensor with shape torch.Size([self.mini_batch_size, action_dim])
            action selected by the agent
        log_probs_batch: torch.float32 tensor with shape torch.Size([self.mini_batch_size, 1])
            log probability of the action selected by the agent
        returns_batch: torch.float32 tensor with shape torch.Size([self.mini_batch_size, 1])
            returns from each state
        advantages_batch: torch.float32 tensor with shape torch.Size([self.mini_batch_size, 1])
            advantages of each state
        """

        subset_random_sampler = SubsetRandomSampler(range(self.num_steps))
        batch_sampler = BatchSampler(subset_random_sampler, self.mini_batch_size, drop_last=True)

        for indices in batch_sampler:
            
            states_batch = self.states[indices]
            values_batch = self.values[indices]
            actions_batch = self.actions[indices]
            log_probs_batch = self.log_probs[indices]
            
            returns_batch = self.returns[indices]
            advantages_batch = self.advantages[indices]

            yield states_batch, values_batch, actions_batch, log_probs_batch, returns_batch, advantages_batch

    def __len__(self):
        """
        Retrieve the length of the memory.
        """

        return self.step
