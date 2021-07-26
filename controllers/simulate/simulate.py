import argparse
import pickle
import random
import time

import numpy as np
import torch

from controllers.ppov2.ppov2_agent import PPOv2
from controllers.sacv2.sacv2_agent import SACv2
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs

parser = argparse.ArgumentParser(description="Simulate Arguments")

parser.add_argument("-f", "--file", default="",
                    help="absolute path of the folder containing data")

parser.add_argument("-t", "--time_steps", default="",
                    help="the number of time steps into learning")

parser.add_argument("-e", "--env_name", default=None,  # run the learned model in a environment different from the environment found in the filename (learning environment)
                    help="name of MuJoCo Gym environment (default: None)")

args = parser.parse_args()


class Simulate:
    """
    Controller for simulating SAC or PPO learning progress.
    """

    LINE = "--------------------------------------------------------------------------------"

    def __init__(self):

        self.load_data_dir = args.file
        self.t = int(args.time_steps)

        self.parameters = None
        self.load_parameters()

        if "ab_env_name" in self.parameters:
            env_name = self.parameters["ab_env_name"]
        else:
            env_name = self.parameters["n_env_name"]
            if env_name == "Ant-v2":
                env_name = "AntEnv-v0"  # use our own ant class so that we can view rendering of ant properly
            elif env_name == "FetchReach-v1":
                env_name = "FetchReachEnv-v0"

        if args.env_name is not None:  # if we have specified an ab_env_name argument, we are attempting to see the policy learned in the normal environment in the abnormal env
            env_name = args.env_name

        # seeds

        random.seed(self.parameters["seed"])
        np.random.seed(self.parameters["seed"])
        torch.manual_seed(self.parameters["seed"])
        if self.parameters["device"] == "cuda":
            torch.cuda.manual_seed(self.parameters["seed"])

        # env is seeded in Environment __init__() method

        # rl problem

        # normal environment used for training
        self.env = Environment(env_name,
                               self.parameters["seed"],
                               render=True)

        # agent
        if "SAC" in self.load_data_dir:

            self.agent = SACv2(self.env.env_state_dim(),
                               self.env.env_action_dim(),
                               self.parameters["gamma"],
                               self.parameters["tau"],
                               self.parameters["alpha"],
                               self.parameters["lr"],
                               self.parameters["hidden_dim"],
                               self.parameters["replay_buffer_size"],
                               self.parameters["batch_size"],
                               self.parameters["model_updates_per_step"],
                               self.parameters["target_update_interval"],
                               self.parameters["automatic_entropy_tuning"],
                               self.parameters["device"],
                               None)

        elif "PPO" in self.load_data_dir:

            self.agent = PPOv2(self.env.env_state_dim(),
                               self.env.env_action_dim(),
                               self.parameters["hidden_dim"],
                               self.parameters["log_std"],
                               self.parameters["lr"],
                               self.parameters["linear_lr_decay"],
                               self.parameters["gamma"],
                               self.parameters["n_time_steps"],
                               self.parameters["num_samples"],
                               self.parameters["mini_batch_size"],
                               self.parameters["epochs"],
                               self.parameters["epsilon"],
                               self.parameters["vf_loss_coef"],
                               self.parameters["policy_entropy_coef"],
                               self.parameters["clipped_value_fn"],
                               self.parameters["max_grad_norm"],
                               self.parameters["use_gae"],
                               self.parameters["gae_lambda"],
                               self.parameters["device"],
                               None,
                               False)

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = {"num_episodes": 0, "num_steps": 0, "total_reward": 0}

        # load agent model

        self.rlg.rl_agent_message("load_model, {}, {}".format(self.load_data_dir, self.t))  # load agent data
        self.rlg.rl_agent_message("mode, eval")  # set mode to eval: no learning, deterministic policy

    def cleanup(self):
        """
        Close environment.
        """

        self.rlg.rl_env_message("close")

    def load_parameters(self):
        """
        Load normal experiment parameters.

        File format: .pickle
        """

        pickle_foldername = self.load_data_dir + "/pickle"

        with open(pickle_foldername + "/parameters.pickle", "rb") as f:
            self.parameters = pickle.load(f)

    def run(self):

        self.rlg.rl_init(total_reward=self.rlg_statistics["total_reward"],
                         num_steps=self.rlg_statistics["num_steps"],
                         num_episodes=self.rlg_statistics["num_episodes"])

        for _ in range(10):

            self.rlg.rl_start()

            terminal = False

            max_steps_this_episode = 1000
            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, _, terminal, _ = self.rlg.rl_step()

                time.sleep(0.1)


if __name__ == "__main__":

    sim = Simulate()

    try:

        sim.run()

    except KeyboardInterrupt as e:

        print("keyboard interrupt")
        sim.cleanup()

# normal ant
# -f "/media/sschoepp/easystore/shared/ant/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed2" -t 20000000

# fetchreach normal
# -f "/media/sschoepp/easystore/shared/fetchreach/normal/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21/seed0" -t 2000000

# faulty ant before learning (retaining all data)
# -f "/media/sschoepp/easystore/shared/ant/faulty/sac/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_r/seed2" -t 20000000

# faulty ant after learning (best setting)
# -f "/media/sschoepp/easystore/shared/ant/faulty/sac/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:True_a:True_d:cuda_resumed/seed0" -t 40000000