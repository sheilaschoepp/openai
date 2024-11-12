import os
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam

from controllers.ppo.ppo_networks import ActorCriticNetwork
from utils.memory import Memory
from utils.rl_glue import BaseAgent


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent (based on the paper at https://arxiv.org/abs/1707.06347).

    The agent program implements the learning algorithm and action-selection mechanism.  -Brian Tanner & Adam White
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 log_std,
                 lr,
                 linear_lr_decay,
                 slow_lrd,
                 gamma,
                 time_steps,
                 num_samples,
                 mini_batch_size,
                 epochs,
                 epsilon,
                 vf_loss_coef,
                 policy_entropy_coef,
                 clipped_value_fn,
                 max_grad_norm,
                 use_gae,
                 gae_lambda,
                 device,
                 loss_data,
                 resume):
        """
        Initialize agent variables.
        Initialize agent neural network.

        @param state_dim: int
            environment state dimension
        @param action_dim: int
            action dimension
        @param hidden_dim: int
            hidden dimension
        @param log_std: float
            log standard deviation of the policy distribution
        @param lr: float
            learning rate
        @param linear_lr_decay: bool
            if true, decrease the learning rate linearly
        @param slow_lrd: float
            slow linear learning rate decay by this percentage
        @param gamma: float
            discount factor
        @param time_steps: int
            number of time steps in normal (non-malfunctioning) MuJoCo Gym environment
        @param num_samples: int
            number of samples used to update the network(s)
        @param mini_batch_size: int
            number of samples per mini-batch
        @param epochs: int
            number of epochs when updating the network(s)
        @param epsilon: float
            clip parameter
        @param vf_loss_coef: float
            c1 - coefficient for the squared error loss term
        @param policy_entropy_coef: float
            c2 - coefficient for the entropy bonus term
        @param clipped_value_fn: bool
            if true, clip value function
        @param max_grad_norm: float
            max norm of gradients
        @param use_gae: bool
            if true, use generalized advantage estimation
        @param gae_lambda: float
            generalized advantage estimation smoothing parameter
        @param device: string
            indicates whether using 'cuda' or 'cpu'
        @param loss_data: float64 numpy zeros array with shape (n_timesteps / num_samples * epochs, 3)
            numpy array to store agent loss data
        @param resume: bool
            if true, we are resuming the experiment
        """

        super(PPO, self).__init__()

        assert num_samples >= mini_batch_size, "agent.__init__: the number of samples must be greater than or equal to the mini-batch size"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std = log_std
        self.lr = lr
        self.linear_lr_decay = linear_lr_decay
        self.slow_lrd = slow_lrd

        self.time_steps = time_steps
        self.num_samples = num_samples
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs

        self.epsilon = epsilon
        self.vf_loss_coef = vf_loss_coef
        self.policy_entropy_coef = policy_entropy_coef
        self.clipped_value_fn = clipped_value_fn
        self.max_grad_norm = max_grad_norm

        self.device = device
        self.loss_data = loss_data

        self.resume = resume

        # network
        self.actor_critic_network = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.log_std).to(device=self.device)

        # optimizer
        self.actor_critic_optimizer = Adam(self.actor_critic_network.parameters(), lr=self.lr)

        # mse loss criterion
        self.actor_critic_criterion = nn.MSELoss()

        # memory
        self.memory = Memory(num_samples, self.state_dim, self.action_dim, gamma, use_gae, gae_lambda, mini_batch_size, device=self.device)
        self.memory_init_samples = 0

        # loss_index
        self.loss_index = 0

        # total number of network updates that will occur
        self.total_num_updates = self.time_steps // self.num_samples

        # number of full updates of the network(s)
        self.num_updates = 0
        self.num_old_updates = 0

        # number of epochs to update the network(s)
        self.num_epoch_updates = 0

        # number of mini batch updates of the networks
        self.num_mini_batch_updates = 0

        # RL problem
        self.state = None
        self.action = None
        self.log_prob = None

        # mode - learn or evaluation
        # if train, select an action using a policy distribution; add experience to the memory; policy model mode is 'train'
        # if eval, select a deterministic action (distribution mean); do not add experience to the memory; policy mode is 'eval'
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

        @return action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        action = self.agent_select_action(state)

        return action

    def agent_step(self, reward, next_state, terminal):
        """
        A step taken by the agent.
        Store the (s, a, lp, r, t) tuple into the memory:
        s: state,
        a: action,
        lp: log probability of action,
        r: reward, and
        t: terminal.
        If it is time to update, update the parameters for the NN(s), then clear the memory.
        Agent selects an action.

        @param reward: float64
            reward received for taking action
        @param next_state: float64 numpy array with shape (state_dim,)
            the state of the environment after taking action
        @param terminal: boolean
            true if the goal state has been reached after taking action; otherwise false

        @return action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        if self.mode == "train":

            self.memory.add(self.state, self.action, self.log_prob, reward, terminal)

            if len(self.memory) == 0:  # memory is full with new samples
                self.agent_memory_compute(next_state)
                self.agent_update_network_parameters()

        action = self.agent_select_action(next_state)

        return action

    def agent_end(self, reward, next_state, terminal):
        """
        The goal state has been reached.  Terminate the agent.
        Store the (s, a, lp, r, t) tuple into the memory:
        s: state,
        a: action,
        lp: log probability of action,
        r: reward, and
        t: terminal.
        If it is time to update, update the parameters for the NN(s), then clear the memory.

        @param reward: float64
            reward received for taking action
        @param next_state: float64 numpy array with shape (state_dim,)
            the state of the environment after taking action
        @param terminal: boolean
            true if the goal state has been reached after taking action; otherwise false

        @return self.action: None
        """

        if self.mode == "train":

            self.memory.add(self.state, self.action, self.log_prob, reward, terminal)

            if len(self.memory) == 0:  # memory is full with new samples
                self.agent_memory_compute(next_state)
                self.agent_update_network_parameters()

        self.state = None
        self.action = None
        self.log_prob = None

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

        if split_message[0] == "clear_memory":
            self.agent_clear_memory()
            self.memory_init_samples = 0
            self.agent_load_total_num_updates()
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
        if split_message[0] == "reset_lr":
            self.agent_reset_lr()
        if split_message[0] == "save":
            data_dir = split_message[1]
            time_steps = int(split_message[2])
            self.agent_save(data_dir, time_steps)
        if split_message[0] == "save_model":
            data_dir = split_message[1]
            time_steps = int(split_message[2])
            self.agent_save_models(data_dir, time_steps)

        return response

    def agent_clear_memory(self):
        """
        Clear the memory.
        """

        self.memory.clear()

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
        self.agent_load_memory(dir_)
        self.agent_load_memory_inti_samples(dir_)  # must come after agent_load_memory
        self.agent_load_total_num_updates()  # must come after agent_load_memory

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
            self.actor_critic_network.load_state_dict(checkpoint["actor_critic_network_state_dict"])

            # send to GPU
            self.actor_critic_network.to(device=self.device)

        else:

            # send to CPU
            checkpoint = torch.load(tar_foldername + "/{}.tar".format(t), map_location=torch.device(self.device))

            # load neural network(s)
            self.actor_critic_network.load_state_dict(checkpoint["actor_critic_network_state_dict"])

        # set network(s) to training mode
        self.actor_critic_network.train()

        # load Adam optimizers
        self.actor_critic_optimizer.load_state_dict(checkpoint["actor_critic_optimizer_state_dict"])

    def agent_load_num_updates(self, dir_):
        """
        Load number of updates.

        File format: .pickle

        @param dir_: string
            load directory
        """

        pickle_foldername = dir_ + "/pickle"

        with open(pickle_foldername + "/num_updates.pickle", "rb") as f:
            update_dic = pickle.load(f)

        self.loss_index = update_dic["loss_index"]
        self.total_num_updates = update_dic["total_num_updates"]
        self.num_updates = update_dic["num_updates"]

        if self.resume:
            self.num_old_updates = update_dic["num_old_updates"]
        else:
            self.num_old_updates = update_dic["num_updates"]  # number of updates with n_controller

        self.num_epoch_updates = update_dic["num_epoch_updates"]
        self.num_mini_batch_updates = update_dic["num_mini_batch_updates"]

    def agent_load_memory(self, dir_):
        """
        Load memory.

        File format: .pickle

        @param dir_: string
            load directory
        """

        pickle_foldername = dir_ + "/pickle"

        with open(pickle_foldername + "/memory.pickle", "rb") as f:
            self.memory = pickle.load(f)

    def agent_load_memory_inti_samples(self, dir_):
        """
        Load the number of samples stored in memory at the start of learning with the abnormal controller.

        File format: .pickle

        @param dir_: string
            load directory
        """

        if self.resume:

            pickle_foldername = dir_ + "/pickle"

            with open(pickle_foldername + "/memory_init_samples.pickle", "rb") as f:
                memory_dic = pickle.load(f)

            self.memory_init_samples = memory_dic["memory_init_samples"]

        else:

            self.memory_init_samples = self.memory.step

    def agent_load_total_num_updates(self):
        """
        Update instance attribute total_num_updates - the number of updates that will occur in this experiment.
        This value is used to linearly decay the learning rate in the agent_update_network_parameters method.

        Note: If we load the memory, there may be samples stored within the memory.
        Note: If we clear the memory, any samples are no longer stored within the memory.
        """

        if self.resume:
            pass
        else:
            self.total_num_updates = (self.memory.step + self.time_steps) // self.num_samples

    def agent_memory_compute(self, next_state):
        """
        Compute the values of all states in memory and save them to memory.
        Update the memory by computing returns and (normalized) advantages for the collected samples.

        @param next_state: float64 numpy array with shape (state_dim,)
            state of the environment
        """

        # save next state and terminal to memory
        self.memory.states[-1].copy_(torch.from_numpy(next_state).float())

        # compute value for all states in memory (incl. next state)
        with torch.no_grad():
            values = self.actor_critic_network.value(self.memory.states)

        # save values to memory
        self.memory.values.copy_(values)

        # compute returns and (normalized) advantages
        self.memory.compute_returns_and_advantages()

    def agent_reinitialize_networks(self):
        """
        Randomly re-initialize the network(s) and optimizer(s).
        """

        self.actor_critic_network = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.log_std).to(device=self.device)
        self.actor_critic_optimizer = Adam(self.actor_critic_network.parameters(), lr=self.lr)

        self.num_updates = 0
        self.num_old_updates = 0
        self.num_epoch_updates = 0
        self.num_mini_batch_updates = 0

    def agent_reset_lr(self):
        """
        Reset the agent's learning rate to its original value.
        """
        for param_group in self.actor_critic_optimizer.param_groups:  # reset learning rate
            param_group["lr"] = self.lr

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
        self.agent_save_memory(dir_)
        self.agent_save_memory_init_samples(dir_)

    def agent_save_memory(self, dir_):
        """
        Save memory.

        File format: .pickle

        @param dir_: string
            save directory
        """

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(pickle_foldername + "/memory.pickle", "wb") as f:
            pickle.dump(self.memory, f)

    def agent_save_memory_init_samples(self, dir_):
        """
        Save number of samples in memory at the start of learning with the ab_controller..

        File format: .pickle

        @param dir_: string
            save directory
        """

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        memory_dic = {"memory_init_samples": self.memory_init_samples}

        with open(pickle_foldername + "/memory_init_samples.pickle", "wb") as f:
            pickle.dump(memory_dic, f)

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
            "actor_critic_network_state_dict": self.actor_critic_network.state_dict(),
            "actor_critic_optimizer_state_dict": self.actor_critic_optimizer.state_dict()
        }, foldername + "/{}.tar".format(t))

    def agent_save_num_updates(self, dir_):
        """
        Save number of updates.

        File format: .pickle

        @param dir_: string
            save directory
        """

        pickle_foldername = dir_ + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        update_dic = {"loss_index": self.loss_index,
                      "total_num_updates": self.total_num_updates,
                      "num_updates": self.num_updates,
                      "num_old_updates": self.num_old_updates,
                      "num_epoch_updates": self.num_epoch_updates,
                      "num_mini_batch_updates": self.num_mini_batch_updates}

        with open(pickle_foldername + "/num_updates.pickle", "wb") as f:
            pickle.dump(update_dic, f)

    def agent_select_action(self, state):
        """
        Select an action.
        Store data in instance variables (self.state, self.action and self.log_prob).

        Note: The deterministic flag indicates how an action is selected.
        deterministic=False: an action is sampled from a multivariate normal distribution.
        deterministic=True: action is mean of multivariate normal distribution.

        @param state: float64 numpy array with shape (state_dim,)
            state of the environment

        @return action: float64 numpy array with shape (action_dim,)
            action selected by the agent
        """

        if self.mode == "train":
            deterministic = False
        else:
            deterministic = True

        self.state = torch.from_numpy(state).float().unsqueeze(0).to(device=self.device)

        with torch.no_grad():
            self.action, self.log_prob = self.actor_critic_network.sample(self.state, deterministic=deterministic)

        action = self.action.cpu().numpy()[0].astype("float64")

        return action

    def agent_set_policy_mode(self, mode):
        """
        Set the policy mode (deterministic or stochastic).
        Set policy model mode (train or eval).

        Note: Summary of effects of this method

        'eval':
        - deterministic policy (see self.agent_select_action() method)
        - no new experiences added to the memory (see self.agent_step() and self.agent_end() methods)
        - network parameters not updated (see self.agent_set_policy_mode() method)
        - set policy model to evaluation mode - output of policy network may be changed if it contains dropout and
          batch normalization layers

        'train':
        - stochastic policy (see self.agent_select_action() method)
        - new experiences added to the memory (see self.agent_step() and self.agent_end() methods)
        - network parameters are updated (see self.agent_set_policy_mode() method)
        - set policy model to training mode

        @param mode: string
            mode: training mode ('train') during learning, evaluation mode ('eval') during inference
        """

        assert (mode == "train" or mode == "eval"), "ppo_agent.agent_set_policy_mode: mode must be either 'train' or 'eval'"

        if mode == "train":
            self.mode = "train"
            self.actor_critic_network.train()
        else:
            self.mode = "eval"
            self.actor_critic_network.eval()

    def agent_update_network_parameters(self):
        """
        Update the learning rate.
        Update the parameters for the NN(s).

        Note: The learning rate starts at its full value (as passed by the argument 'lr').  The learning rate is
        gradually decreased by an equal amount each update until it reaches 0.0.
        Note: At the start of learning in the normal environment, the learning rate has its full value.  At the start of
        learning in the abnormal environment, the learning rate has its full value.
        """

        if self.linear_lr_decay:

            lr = self.lr - (self.lr * ((self.num_updates - self.num_old_updates) / self.total_num_updates) * self.slow_lrd)  # self.num_old_updates is > 0 only if we loaded data from normal environment

            for param_group in self.actor_critic_optimizer.param_groups:
                param_group["lr"] = lr

        self.num_updates += 1

        avg_clip_loss = 0
        avg_vf_loss = 0
        avg_entropy = 0
        avg_clip_vf_s_loss = 0
        total_num_clips = 0

        for _ in range(self.epochs):

            self.num_epoch_updates += 1

            mini_batches = self.memory.generate_mini_batches()

            for mb in mini_batches:

                self.num_mini_batch_updates += 1

                states_batch, values_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch = mb

                new_log_probs_batch, new_dist_entropies_batch = self.actor_critic_network.evaluate(states_batch, actions_batch)
                new_values_batch = self.actor_critic_network.value(states_batch)

                ratios = torch.exp(new_log_probs_batch - old_log_probs_batch)

                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages_batch

                clip_loss = -1 * torch.min(surr1, surr2).mean()

                clipped = ratios.lt(1 - self.epsilon) | ratios.gt(1 + self.epsilon)
                num_clips = int(clipped.float().sum())
                total_num_clips += num_clips

                if self.clipped_value_fn:

                    values_batch_clipped = values_batch + (new_values_batch - values_batch).clamp(-self.epsilon, self.epsilon)
                    values_loss = (new_values_batch - returns_batch).pow(2)
                    clip_values_loss = (values_batch_clipped - returns_batch).pow(2)
                    vf_loss = -1 * torch.max(values_loss, clip_values_loss).mean()

                else:

                    vf_loss = -1 * self.actor_critic_criterion(returns_batch, new_values_batch)

                new_dist_entropy = -1 * new_dist_entropies_batch.mean()

                clip_vf_s_loss = clip_loss - self.vf_loss_coef * vf_loss + self.policy_entropy_coef * new_dist_entropy

                self.actor_critic_optimizer.zero_grad()
                clip_vf_s_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic_network.parameters(), self.max_grad_norm)
                self.actor_critic_optimizer.step()

                avg_clip_loss += clip_loss.item()
                avg_vf_loss += vf_loss.item()
                avg_entropy += -1 * new_dist_entropy.item()
                avg_clip_vf_s_loss += clip_vf_s_loss.item()

        num_updates = self.epochs * (self.num_samples // self.mini_batch_size)

        avg_clip_loss /= num_updates
        avg_vf_loss /= num_updates
        avg_entropy /= num_updates
        avg_clip_vf_s_loss /= num_updates

        clip_fraction = total_num_clips / (self.num_epoch_updates * self.mini_batch_size * (self.num_samples // self.mini_batch_size))

        # self.loss_data[self.loss_index] = [self.num_updates, self.num_epoch_updates, self.num_mini_batch_updates, avg_clip_loss, avg_vf_loss, avg_entropy, avg_clip_vf_s_loss, clip_fraction]
        self.loss_index += 1
