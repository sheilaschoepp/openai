import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam

from controllers.sacv2.mod.sacv2_networks import TwinnedQNetwork, GaussianPolicyNetwork
from utils.replay_buffer import ReplayBuffer
from utils.rl_glue import BaseAgent


class SACv2(BaseAgent):
    """
    Soft Actor Critic (SAC) Agent (based on the paper at https://arxiv.org/abs/1812.05905).

    The agent program implements the learning algorithm and action-selection mechanism.  -Brian Tanner & Adam White
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma,
                 tau,
                 alpha,
                 lr,
                 hidden_dim,
                 replay_buffer_size,
                 batch_size,
                 model_updates_per_step,
                 target_update_interval,
                 automatic_entropy_tuning,
                 device,
                 loss_data):
        """
        Initialize agent variables.
        Initialize agent neural network(s):
        -value network,
        -target value network,
        -soft q network 1,
        -soft q network 2, and
        -policy network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param gamma: float
            discount factor
        @param tau: float
            target smoothing coefficient
        @param alpha: float
            temperature parameter that determines the relative importance of the entropy term against the reward term
        @param lr: float
            learning rate
        @param hidden_dim: int
            hidden dimension
        @param replay_buffer_size: int
            size of the replay buffer
        @param batch_size: int
            number of samples per minibatch
        @param model_updates_per_step: int
            number of NN model updates per time step
        @param target_update_interval: int
            number of target value network updates per number gradient steps (network updates)
        @param automatic_entropy_tuning: bool
            if True, automatically tune the temperature
        @param device: string
            indicates whether using 'cuda' or 'cpu'
        @param loss_data: float64 numpy zeros array with shape (n_time_steps - batch_size, 5)
            numpy array to store agent loss data
        """

        super(SACv2, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.model_updates_per_step = model_updates_per_step
        self.replay_buffer_size = replay_buffer_size
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device
        self.alpha = torch.Tensor([alpha]).to(device=self.device)  # must come after self.device
        self.loss_data = loss_data

        self.loss_index = 0

        self.num_updates = 0

        # critic network
        self.q_network = TwinnedQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)
        self.target_q_network = TwinnedQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)

        # actor network
        self.policy_network = GaussianPolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)

        # (hard update) target q value network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.q_optimizer_1 = Adam(self.q_network.Q1.parameters(), lr=self.lr)
        self.q_optimizer_2 = Adam(self.q_network.Q2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy_network.parameters(), lr=self.lr)

        # mse loss criteria
        self.q_criterion_1 = nn.MSELoss()
        self.q_criterion_2 = nn.MSELoss()

        if self.automatic_entropy_tuning:
            # target entropy = −dim(A)
            self.target_entropy = -torch.prod(torch.tensor(action_dim)).to(device=self.device).item()
            # optimize log(alpha) instead of alpha
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # RL problem
        self.state = None
        self.action = None

        # mode - learn or evaluation
        # if train, select an action using a policy distribution; add experience to the replay buffer; policy model mode is 'train'
        # if eval, select a deterministic action (distribution mean); do not add experience to the replay buffer; policy mode is 'eval'
        self.mode = "train"

    def agent_init(self):
        """
        Initialize the variables that you want to reset before starting a new run.
        """

        pass

    def agent_start(self, state):
        """
        Initialize the variables that you want to reset before starting a new episode.
        Agent selects an action.

        @param state: float64 numpy array with shape (state_dim,)
            state of the environment

        @return self.action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        self.state = state

        self.action = self.agent_select_action(state)

        return self.action

    def agent_step(self, reward, next_state, terminal):
        """
        A step taken by the agent.
        Store the (s, a, r, s', t) tuple into the replay buffer:
        s: state,
        a: action,
        r: reward,
        s': next_state, and
        t: terminal.
        If it is time to update, update the parameters for the NN(s).
        Agent selects an action.

        @param reward: float64
            reward received for taking action
        @param next_state: float64 numpy array with shape (state_dim,)
            the state of the environment after taking action
        @param terminal: boolean
            true if the goal state has been reached after taking action; otherwise false

        @return self.action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        if self.mode == "train":

            self.replay_buffer.add(self.state, self.action, reward, next_state, terminal)

            if len(self.replay_buffer) > self.batch_size:
                for i in range(self.model_updates_per_step):
                    self.agent_update_network_parameters()

        self.state = next_state

        self.action = self.agent_select_action(self.state)

        return self.action

    def agent_end(self, reward, next_state, terminal):
        """
        The goal state has been reached.  Terminate the agent.
        Store the (s, a, r, s', t) tuple into the replay buffer:
        s: state,
        a: action,
        r: reward,
        s': next_state, and
        t: terminal.
        If it is time to update, update the parameters for the NN(s).

        @param reward: float64
            reward received for taking action
        @param next_state: float64 numpy array with shape (state_dim,)
            the state of the environment after taking action
        @param terminal: boolean
            true if the goal state has been reached after taking action; otherwise false

        @return self.action: None
        """

        if self.mode == "train":

            self.replay_buffer.add(self.state, self.action, reward, next_state, terminal)

            if len(self.replay_buffer) > self.batch_size:
                for i in range(self.model_updates_per_step):
                    self.agent_update_network_parameters()

        self.state = None
        self.action = None

        return self.action

    def agent_message(self, message):
        """
        Receive a message from RLGlue.
        @param message: str
            the message passed
        @return response: str
            the agent's response to the message (default: "")
        """

        split_message = message.split(", ")
        response = ""

        if split_message[0] == "clear_replay_buffer":
            self.agent_clear_replay_buffer()
        if split_message[0] == "load":
            data_dir = split_message[1]
            time_steps = int(split_message[2])
            self.agent_load(data_dir, time_steps)
        if split_message[0] == "load_model":
            data_dir = split_message[1]
            time_steps = int(split_message[2])
            self.agent_load_models(data_dir, time_steps)
        if split_message[0] == "mode":
            mode = split_message[1]
            self.agent_set_policy_mode(mode)
        if split_message[0] == "reinitialize_networks":
            self.agent_reinitialize_networks()
        if split_message[0] == "save":
            data_dir = split_message[1]
            time_steps = int(split_message[2])
            self.agent_save(data_dir, time_steps)
        if split_message[0] == "save_model":
            data_dir = split_message[1]
            time_steps = int(split_message[2])
            self.agent_save_models(data_dir, time_steps)

        return response

    def agent_clear_replay_buffer(self):
        """
        Clear the replay buffer.
        """

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def agent_load(self, dir_, t):
        """
        Load agent's data.

        @param dir_: string
            load directory
        @param t: int
            number of time steps into training
        """

        self.agent_load_alpha(dir_)  # this must come before agent_load_models
        self.agent_load_models(dir_, t)
        self.agent_load_num_updates(dir_)
        self.agent_load_replay_buffer(dir_)

    def agent_load_alpha(self, dir_):
        """
        Load alpha attributes: log_alpha, alpha and alpha_optimizer.

        File format: .npy

        @param dir_: string
            save directory
        """

        if self.automatic_entropy_tuning:

            pt_foldername = dir_ + "/pt"

            self.target_entropy = torch.load(pt_foldername + "/target_entropy.pt", map_location=torch.device(self.device))
            self.log_alpha = torch.load(pt_foldername + "/log_alpha.pt", map_location=torch.device(self.device))
            self.alpha = torch.load(pt_foldername + "/alpha.pt", map_location=torch.device(self.device))
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)

    def agent_load_models(self, dir_, t):
        """
        Load NNs and optimizers.  Set NNs to training mode.

        Important: you must initialize them before loading!

        File format: .tar

        @param dir_: string
            load directory
        @param t: int
            number of time steps into training
        """

        tar_foldername = dir_ + "/tar"

        if self.device == "cuda":

            # send to gpu
            checkpoint = torch.load(tar_foldername + "/{}.tar".format(t))

            # load neural network(s)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
            self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])

            # send to GPU
            self.q_network.to(self.device)
            self.target_q_network.to(self.device)
            self.policy_network.to(self.device)

        else:

            # send to CPU
            checkpoint = torch.load(tar_foldername + "/{}.tar".format(t), map_location=torch.device(self.device))

            # load neural network(s)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
            self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])

        # set network(s) to training mode
        self.q_network.train()
        self.target_q_network.train()
        self.policy_network.train()

        # load Adam optimizers
        self.q_optimizer_1.load_state_dict(checkpoint["q_optimizer_1_state_dict"])
        self.q_optimizer_2.load_state_dict(checkpoint["q_optimizer_2_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        if self.automatic_entropy_tuning:

            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

    def agent_load_num_updates(self, dir_):
        """
        Load number of updates.

        File format: .npy

        @param dir_: string
            load directory
        """

        numpy_foldername = dir_ + "/npy"

        self.loss_index = int(np.load(numpy_foldername + "/loss_index.npy"))
        self.num_updates = int(np.load(numpy_foldername + "/num_updates.npy"))

    def agent_load_replay_buffer(self, dir_):
        """
        Load replay buffer.

        File format: .pickle

        @param dir_: string
            load directory
        """

        pickle_foldername = dir_ + "/pickle"

        with open(pickle_foldername + "/replay_buffer.pickle", "rb") as f:
            self.replay_buffer = pickle.load(f)

    def agent_reinitialize_networks(self):
        """
        Randomly re-initialize the network(s) and optimizer(s).
        """

        self.num_updates = 0

        self.q_network = TwinnedQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)
        self.target_q_network = TwinnedQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)

        self.policy_network = GaussianPolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)

        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(param.data)

        self.q_optimizer_1 = Adam(self.q_network.Q1.parameters(), lr=self.lr)
        self.q_optimizer_2 = Adam(self.q_network.Q2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy_network.parameters(), lr=self.lr)

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor(self.action_dim)).to(device=self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)

    def agent_save(self, dir_, t):
        """
        Save agent's data.

        @param dir_: string
            save directory
        @param t: int
            number of time steps into training
        """

        self.agent_save_log_alpha(dir_)
        self.agent_save_models(dir_, t)
        self.agent_save_num_updates(dir_)
        self.agent_save_replay_buffer(dir_)

    def agent_save_log_alpha(self, dir_):
        """
        Save log_alpha and related instance attributes.

        File format: .pt

        @param dir_: string
            save directory
        """

        if self.automatic_entropy_tuning:

            pt_foldername = dir_ + "/pt"
            os.makedirs(pt_foldername, exist_ok=True)

            torch.save(self.target_entropy, pt_foldername + "/target_entropy.pt")
            torch.save(self.log_alpha, pt_foldername + "/log_alpha.pt")
            torch.save(self.alpha, pt_foldername + "/alpha.pt")

    def agent_save_models(self, dir_, t):
        """
        Save NNs and optimizers.

        File format: .tar

        @param dir_: string
            save directory
        @param t: int
            number of time steps into training
        """

        foldername = dir_ + "/tar"
        os.makedirs(foldername, exist_ok=True)

        if self.automatic_entropy_tuning:

            torch.save({
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "policy_network_state_dict": self.policy_network.state_dict(),
                "q_optimizer_1_state_dict": self.q_optimizer_1.state_dict(),
                "q_optimizer_2_state_dict": self.q_optimizer_2.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict()
            }, foldername + "/{}.tar".format(t))

        else:

            torch.save({
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "policy_network_state_dict": self.policy_network.state_dict(),
                "q_optimizer_1_state_dict": self.q_optimizer_1.state_dict(),
                "q_optimizer_2_state_dict": self.q_optimizer_2.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict()
            }, foldername + "/{}.tar".format(t))

    def agent_save_num_updates(self, dir_):
        """
        Save number of updates.

        File format: .npy

        @param dir_: string
            save directory
        """

        npy_foldername = dir_ + "/npy"
        os.makedirs(npy_foldername, exist_ok=True)

        np.save(npy_foldername + "/loss_index.npy", self.loss_index)
        np.save(npy_foldername + "/num_updates.npy", self.num_updates)

    def agent_save_replay_buffer(self, dir_):
        """
        Save replay buffer.

        File format: .pickle

        @param dir_: string
            save directory
        """

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(pickle_foldername + "/replay_buffer.pickle", "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def agent_select_action(self, state):
        """
        Select an action.

        @param state: float64 numpy array with shape (state_dim,)
            state of the environment

        @return action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        state = torch.from_numpy(state).float().to(device=self.device).unsqueeze(0)

        if self.mode == "train":
            action, _, _, _ = self.policy_network.sample(state)
        else:
            _, _, _, mean = self.policy_network.sample(state)
            action = torch.tanh(mean)

        action = action.detach().cpu().numpy()[0].astype("float64")

        return action

    def agent_set_policy_mode(self, mode):
        """
        Set the policy mode (deterministic or stochastic).
        Set policy model mode (train or eval).

        Note: Summary of effects of this method

        'eval':
        - deterministic policy (see self.agent_select_action() method)
        - no new experiences added to the replay buffer (see self.agent_step() and self.agent_end() methods)
        - network parameters not updated (see self.agent_set_policy_mode() method)
        - set policy model to evaluation mode - output of policy network may be changed if it contains dropout and
          batch normalization layers

        'train':
        - stochastic policy (see self.agent_select_action() method)
        - new experiences added to the replay buffer (see self.agent_step() and self.agent_end() methods)
        - network parameters are updated (see self.agent_set_policy_mode() method)
        - set policy model to training mode

        @param mode: string
            mode: training mode ('train') during learning, evaluation mode ('eval') during inference
        """

        assert (mode == "train" or mode == "eval"), "sacv2_agent.agent_set_policy_mode: mode must be either 'train' or 'eval'"

        if mode == "train":
            self.mode = "train"
            self.policy_network.train()
        else:
            self.mode = "eval"
            self.policy_network.eval()

    def agent_update_network_parameters(self):
        """
        Update the parameters for the NN(s).

        Note: This is performed in the following order:
        - value network
        - both q value network's
        - policy network
        """

        self.num_updates += 1

        state, action, reward, next_state, terminal = self.replay_buffer.sample(self.batch_size)

        state = torch.from_numpy(state).to(device=self.device)
        action = torch.from_numpy(action).to(device=self.device)
        reward = torch.from_numpy(reward).unsqueeze(1).to(device=self.device)
        next_state = torch.from_numpy(next_state).to(device=self.device)
        terminal = torch.from_numpy(terminal).unsqueeze(1).to(device=self.device)

        # q_value network

        predicted_q_value_1, predicted_q_value_2 = self.q_network(state, action)

        with torch.no_grad():

            next_state_sampled_action, next_state_log_prob, _, _ = self.policy_network.sample(next_state)

            predicted_target_q_value_1, predicted_target_q_value_2 = self.target_q_network(next_state, next_state_sampled_action)
            estimated_value = torch.min(predicted_target_q_value_1, predicted_target_q_value_2) - self.alpha * next_state_log_prob
            estimated_q_value = reward + self.gamma * (1 - terminal) * estimated_value

        q_value_loss_1 = self.q_criterion_1(predicted_q_value_1, estimated_q_value)
        q_value_loss_2 = self.q_criterion_2(predicted_q_value_2, estimated_q_value)

        self.q_optimizer_1.zero_grad()
        q_value_loss_1.backward()
        self.q_optimizer_1.step()

        self.q_optimizer_2.zero_grad()
        q_value_loss_2.backward()
        self.q_optimizer_2.step()

        # policy network
        sampled_action, log_prob, entropy, _ = self.policy_network.sample(state)

        entropy = entropy.mean()

        sampled_q_value_1, sampled_q_value_2 = self.q_network(state, sampled_action)
        sampled_q_value = torch.min(sampled_q_value_1, sampled_q_value_2)
        policy_loss = ((self.alpha * log_prob) - sampled_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # adjust temperature
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        # (soft update) target q_value network
        if self.num_updates % self.target_update_interval == 0:
            for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.loss_data[self.loss_index] = [self.num_updates, q_value_loss_1.item(), q_value_loss_2.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item(), entropy.item()]
        self.loss_index += 1