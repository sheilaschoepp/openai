import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam

from controllers.sacv1.sacv1_networks import ValueNetwork, TwinnedQNetwork, GaussianPolicyNetwork
from utils.replay_buffer import ReplayBuffer
from utils.rl_glue import BaseAgent


class SACv1(BaseAgent):
    """
    Soft Actor Critic (SAC) Agent (based on the paper at https://arxiv.org/abs/1801.01290).

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
        @param device: string
            indicates whether using 'cuda' or 'cpu'
        @param loss_data: float64 numpy zeros array with shape (n_time_steps - batch_size, 5)
            numpy array to store agent loss data
        """

        super(SACv1, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.model_updates_per_step = model_updates_per_step
        self.replay_buffer_size = replay_buffer_size
        self.target_update_interval = target_update_interval
        self.device = device
        self.loss_data = loss_data

        self.num_updates = 0

        # value network
        self.value_network = ValueNetwork(state_dim, hidden_dim).to(device=self.device)
        self.target_value_network = ValueNetwork(state_dim, hidden_dim).to(device=self.device)

        # critic network
        self.q_network = TwinnedQNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)

        # actor network
        self.policy_network = GaussianPolicyNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)

        # (hard update) target value network
        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.value_optimizer = Adam(self.value_network.parameters(), lr=lr)
        self.q_optimizer_1 = Adam(self.q_network.Q1.parameters(), lr=lr)
        self.q_optimizer_2 = Adam(self.q_network.Q2.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.policy_network.parameters(), lr=lr)

        # mse loss criteria
        self.value_criterion = nn.MSELoss()
        self.q_criterion_1 = nn.MSELoss()
        self.q_criterion_2 = nn.MSELoss()

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

        self.agent_load_models(dir_, t)
        self.agent_load_num_updates(dir_)
        self.agent_load_replay_buffer(dir_)

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
            self.value_network.load_state_dict(checkpoint["value_network_state_dict"])
            self.target_value_network.load_state_dict(checkpoint["target_value_network_state_dict"])
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])

            # send to GPU
            self.value_network.to(self.device)
            self.target_value_network.to(self.device)
            self.q_network.to(self.device)
            self.policy_network.to(self.device)

        else:

            # send to CPU
            checkpoint = torch.load(tar_foldername + "/{}.tar".format(t), map_location=self.device)

            # load neural network(s)
            self.value_network.load_state_dict(checkpoint["value_network_state_dict"])
            self.target_value_network.load_state_dict(checkpoint["target_value_network_state_dict"])
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])

        # set network(s) to training mode
        self.value_network.train()
        self.target_value_network.train()
        self.q_network.train()
        self.policy_network.train()

        # load Adam optimizers
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.q_optimizer_1.load_state_dict(checkpoint["q_optimizer_1_state_dict"])
        self.q_optimizer_2.load_state_dict(checkpoint["q_optimizer_2_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

    def agent_load_num_updates(self, dir_):
        """
        Load number of updates.

        File format: .npy

        @param dir_: string
            load directory
        """

        numpy_foldername = dir_ + "/npy"

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

    def agent_save(self, dir_, t):
        """
        Save agent's data.

        @param dir_: string
            save directory
        @param t: int
            number of time steps into training
        """

        self.agent_save_models(dir_, t)
        self.agent_save_num_updates(dir_)
        self.agent_save_replay_buffer(dir_)

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

        torch.save({
            "value_network_state_dict": self.value_network.state_dict(),
            "target_value_network_state_dict": self.target_value_network.state_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "policy_network_state_dict": self.policy_network.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
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
            _, _, mean, _ = self.policy_network.sample(state)
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

        assert (mode == "train" or mode == "eval"), "sacv1_agent.agent_set_policy_mode: mode must be either 'train' or 'eval'"

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
        - target value network (Polyak averaging)
        """

        self.num_updates += 1

        state, action, reward, next_state, terminal = self.replay_buffer.sample(self.batch_size)

        state = torch.from_numpy(state).to(device=self.device)
        action = torch.from_numpy(action).to(device=self.device)
        reward = torch.from_numpy(reward).unsqueeze(1).to(device=self.device)
        next_state = torch.from_numpy(next_state).to(device=self.device)
        terminal = torch.from_numpy(terminal).unsqueeze(1).to(device=self.device)

        # q_value network
        sampled_action, log_prob, mean, log_std = self.policy_network.sample(state)

        predicted_q_value_1, predicted_q_value_2 = self.q_network(state, action)

        with torch.no_grad():

            target_value = self.target_value_network(next_state)
            estimated_q_value = reward + (1 - terminal) * self.gamma * target_value

        q_value_loss_1 = self.q_criterion_1(predicted_q_value_1, estimated_q_value)
        q_value_loss_2 = self.q_criterion_2(predicted_q_value_2, estimated_q_value)

        self.q_optimizer_1.zero_grad()
        q_value_loss_1.backward()
        self.q_optimizer_1.step()

        self.q_optimizer_2.zero_grad()
        q_value_loss_2.backward()
        self.q_optimizer_2.step()

        # value network
        predicted_value = self.value_network(state)

        sqv1, sqv2 = self.q_network(state, sampled_action)
        sampled_q_value = torch.min(sqv1, sqv2)

        with torch.no_grad():

            target_v_function = sampled_q_value - (self.alpha * log_prob)

        value_loss = self.value_criterion(predicted_value, target_v_function)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # policy network
        policy_loss = ((self.alpha * log_prob) - sampled_q_value).mean()
        regularization_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        policy_loss += regularization_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # (soft update) target value network
        if self.num_updates % self.target_update_interval == 0:
            for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        index = self.num_updates - 1
        self.loss_data[index] = [self.num_updates, value_loss.item(), q_value_loss_1.item(), q_value_loss_2.item(), policy_loss.item()]
