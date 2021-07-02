import os
import argparse
import numpy as np
from termcolor import colored

parser = argparse.ArgumentParser(description="Missing")

parser.add_argument("-c", "--hpsc", default=False, action="store_true",
                    help="if True, only check for missing csv files (and not for missing tar files)")

args = parser.parse_args()


def run(algorithm, env_name):
    """
    Determine the missing files for a given experiment.

    Finds:
    - missing pss values
    - missing seeds
    - missing final models (if tar folder is available)
    """

    if args.hpsc:
        directory = os.path.join(DATA_DIR, env_name, "hpsc", algorithm)
    else:
        directory = os.path.join(DATA_DIR, env_name, "hps", algorithm)
    dirs = os.listdir(directory)

    print(directory)
    missing_pss_values = list(np.arange(0, 100))

    for dir in dirs:

        if algorithm == "sac" and env_name == "ant" and "_resumed" not in dir:  # skip resumable folders
            pass

        else:

            pss_value = None
            params = dir.split("_")
            for p in params:
                if "pss:" in p:
                    pss_value = int(p[4:])
                    missing_pss_values.remove(pss_value)  # remove the pss value from the missing pss values list
            model = params[1].split(":")[1] + ".tar"

            # check if all seeds are present for given pss value
            for s in range(10):
                seed_foldername = os.path.join(directory, dir, "seed" + str(s))
                if os.path.exists(seed_foldername):

                    # check if final model is present for given pss value and given seed
                    tar_foldername = os.path.join(seed_foldername, "tar")
                    if os.path.exists(tar_foldername):
                        models = os.listdir(tar_foldername)
                        if model not in models:
                            print(colored("missing final model: pss {} seed {} tar {}".format(pss_value, s, model), "red"))

                else:
                    print(colored("missing seed: pss {} seed {}".format(pss_value, s), "red"))

    print("missing pss values for {} {}:".format(env_name, algorithm.upper()), missing_pss_values)


if __name__ == "__main__":
    """
    Find missing files for all experiments.
    """

    DATA_DIR = "/media/sschoepp/easystore/shared"
    # DATA_DIR = "/mnt/DATA/shared"

    RUNS = 10

    # run("ppo", "fetchreach")
    # run("sac", "fetchreach")
    run("ppo", "ant")
    run("sac", "ant")

