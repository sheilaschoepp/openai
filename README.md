# openai

## Required Software and Code

1. [Download](https://mujoco.org/) and [install](https://github.com/openai/mujoco-py#install-mujoco) MuJoCo.
2. Clone [this](https://github.com/sheilaschoepp/openai.git) repository into a directory of your choice (on Linux, clone into the Documents directory).\
`git clone https://github.com/sheilaschoepp/openai.git`

## Anaconda Environment Setup

#### Notes for Ubuntu 20.04

I ran into errors creating an anaconda environment on machines running Ubuntu 20.04.  The failure was in the installation of mujoco-py.  I used the following two resources and commands to resolve the problems:
1. [Source 1](https://github.com/openai/mujoco-py/issues/297) \
`sudo apt install libosmesa6-dev`
2. [Source 2](https://github.com/openai/mujoco-py/issues/147) \
`sudo apt install patchelf`

#### Option 1: Quick Setup for Linux with CUDA 11.0 (currently broken due to pytorch version stored in yml file)
3. Navigate to the openai/yml folder.\
`cd openai/yml`
4. [Create openai anaconda environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). \
`conda env create -f environment.yml`
5. Activate openai anaconda environment.\
`conda activate openai`

OR

#### Option 2: Manual Setup
3. Create anaconda environment with python 3.9.\
`conda create -n openai python=3.9`
4. Activate openai anaconda environment.\
`conda activate openai`
5. [Install](https://pytorch.org/get-started/) pytorch.\
If using CUDA 11.1: `pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
6. Install [OpenAI gym](https://gym.openai.com/docs/), [pandas](https://pandas.pydata.org/), [termcolor](https://pypi.org/project/termcolor/), [matplotlib](https://matplotlib.org/), [tqdm](https://pypi.org/project/tqdm/), [seaborn](https://seaborn.pydata.org/installing.html), [dm_control](https://github.com/deepmind/dm_control) and [pickle5](https://pypi.org/project/pickle5/). \
`pip install gym[all] pandas termcolor matplotlib tqdm seaborn dm_control pickle5`
7. [(Re)install](https://github.com/openai/mujoco-py#install-and-use-mujoco-py) mujoco-py 2.0. \
`pip install -U 'mujoco-py<2.1,>=2.0'`

## Project Structure

#### Branches

* main branch - most recent stable version
* develop branch
* server branch - used to pass scripts to servers

#### Folders

* controllers/sacv2: SAC from paper 'Soft Actor-Critic Algorithms and Applications'
* controllers/ppov2: PPO from paper 'Proximal Policy Optimization Algorithms', upgraded PPO according to paper 'Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO'
* custom_gym_envs: malfunctioning MuJoCo envs (based on Ant-v2 and FetchReach-v1)
* environment: environment python file(s)
* papers: papers that the code is based on
* utils: utilities (i.e. rl_glue.py, replay_buffer.py, plot_style_settings.py)
* yml: environment.yml file for ubuntu machines

## Running an Experiment

#### Controller Files

* The controllers/sacv2/sacv2_n_controller.py file runs SAC v2 algorithm in the normal environment.
* The controllers/sacv2/sacv2_ab_controller.py file runs SAC v2 algorithm in the abnormal environment, after learning in the normal environment has been completed.
* The controllers/ppov2/ppov2_n_controller.py file runs PPO v2 algorithm in the normal environment.
* The controllers/ppov2/ppov2_ab_controller.py file runs PPO v2 algorithm in the abnormal environment, after learning in the normal environment has been completed.

#### Before Running an Experiment

1. In your activated virtual environment within terminal, run `python <ALG>_n_controller.py --help` and observe the different parameters and their default settings.  The <ALG>_n_controller.py file has various arguments, including the name of the normal environment and the number of timesteps to take within this environment.\
`usage: sacv2_n_controller.py [-h] [-e N_ENV_NAME] [-t N] [--gamma G] [--tau G]
                             [--alpha G] [--lr G] [--hidden_dim N] [-rbs N]
                             [--batch_size N] [--model_updates_per_step N]
                             [--target_update_interval N] [-tef N] [-ee N] [-tmsf N]
                             [-a] [-c] [-s N] [-d] [--resumable] [--resume]
                             [-rf RESUME_FILE] [-tl N] [-ps] [-pss N]`
2. In your activated virtual environment within terminal, run `python <ALG>_ab_controller.py --help` and observe the different parameters and their default settings.  The <ALG>_ab_controller.py file has various arguments, including the name of the abnormal environment and the number of timesteps to take within this environment.  The <ALG>_ab_controller.py file is used only after <ALG>_n_controller.py has been run.  The filename argument is the absolute path of the seed# folder containing the data obtained from running <ALG>_n_controller.py to completion.  Note that <ALG>_ab_controller.py has fewer arguments than <ALG>_n_controller.py.  Many of the arguments set in <ALG>_n_controller.py are loaded from file and are retained in <ALG>_ab_controller.py.\
`usage: sacv2_ab_controller.py [-h] [-e AB_ENV_NAME] [-t N] [-crb] [-rn] [-c] [-f FILE]
                              [-d] [--resumable] [--resume] [-rf RESUME_FILE] [-tl N]`

#### To Run an Experiment in Terminal

1. Add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<path_to_mujoco_folder>/.mujoco/mujoco200/bin` and
`export PYTHONPATH="<path_to_openai_folder>/openai/"` to your ~/.bashrc file.
2. Run `python <ALG>_n_controller.py <arguments>` where \<arguments\> is your selected arguments settings that are different from the default settings.
3. Allow <ALG>_n_controller.py to run to completion.
4. Run `python <ALG>_ab_controller.py <arguments>` where \<arguments\> is your selected arguments settings that are different from the default settings.  The filename argument is the absolute path of the seed# folder containing the data obtained from running <ALG>_n_controller.py to completion.  Note that <ALG>_ab_controller.py has fewer arguments than <ALG>_n_controller.py.  Many of the arguments set in <ALG>_n_controller.py are loaded from file and are retained in <ALG>_ab_controller.py.
5. Allow <ALG>_ab_controller.py to run to completion.

#### To Run an Experiment in PyCharm

1. For the selected controller, select *Edit Configurations...* to open the *Edit Run/Debug configurations dialog*.
2. Add *LD_LIBRARY_PATH=/<path_to_mujoco_folder>/.mujoco/mujoco200/bin* to your environment variables.
3. In the top right corner is a drop down menu where the name of the selected controller is listed.  (If you see "Add Configuration" instead of your controller name, run your desired controller using the default settings and immediately stop it.  The name should now appear in the top right.)  Add the arguments in your call to the controller by selecting 'Edit Configurations' from the drop down menu.  Alternatively, you can adjust their settings within the controller files.

#### Making an Experiment Resumable and Resuming an Experiment

The openai experiments require millions of time steps.  For this reason, the functionality to make experiments resumable and to resume experiments was added.

To make an experiment resumable:
* When running the selected controller, add the --resumable argument.  In this case, your experiment will run normally until the number of time steps remaining has reached 1000 or less (note: the maximum episode length is 1000).  At this point, all data is saved at the end of the first completed episode.
* E.g. `python <ALG>_n_controller.py <arguments> --resumable`

To resume a (resumable) experiment:
* When resuming an experiment, several arguments are overwritten and MUST be specified to resume an experiment correctly.  These include: --n_time_steps (or --ab_time_steps), --cuda, --resume and --resume_file.  All remaining arguments are unmodified.
* To complete a resumable experiment without adding additional time steps:
  * set --n_time_steps (or --ab_time_steps) to the original value while specifying the remaining required arguments.
* To add additional time steps to a resumable experiment:
  * set --n_time_steps (or --ab_time_steps) to the new value, while specifying the remaining required arguments.  (You may wish to make the extended experiment resumable by adding the --resumable flag.)
* E.g. `python <ALG>_n_controller.py --cuda --n_time_steps 20000000 --resume --resume_file <absolute_path_of_seedX_folder>`

## Notes

#### Design

* It was a design choice to separate <ALG>_n_controller.py and <ALG>_ab_controller.py.  In doing this, I only had to run <ALG>_n_controller.py once for each set of parameters.  The model learned from running <ALG>_n_controller.py could then be used to continue learning in different malfunction environments with <ALG>_ab_controller.py.

#### Create a Malfunctioning Ant

To create a malfunctioning Ant, the following steps must be taken:
* xml file
  * within the openai/custom_gym_envs/envs/ant/xml folder, copy and paste AntEnv_v0_Normal.xml, editing the version of the .xml filename
  * add malfunction to one or more joints
* python file
  * within the openai/custom_gym_envs/envs/ant/ folder, copy and paste AntEnv_v0_Normal.py, editing the version of the .py filename
  * update the class name with the version number
  * update the filepath instance variable for the class with the path to the appropriate xml file
* init file
  * add new environment to openai/custom_gym_envs/__init__.py

To create a malfunctioning FetchReach, the following steps must be taken:
* copy openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_normal/ folder
* modify one or more of the following files to apply a fault: 
  * assets/fetch/robot.xml
  * fetch/reach.py
  * fetch_env.py
  * robot_env.py
* init file
  * add new environment to openai/custom_gym_envs/__init__.py

Do not edit:
* openai/custom_gym_envs/envs/ant/AntEnv_v0_Normal.py
* openai/custom_gym_envs/envs/ant/xml/AntEnv_v0_Normal.xml
* openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/*

These are the files for a normal Ant and FetchReach environments.  You may copy and paste them with a new name to use as a base for a new malfunctioning MuJoCo environment.
