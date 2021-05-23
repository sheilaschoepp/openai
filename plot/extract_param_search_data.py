import os


if __name__ == "__main__":

    DATA_DIR = "/mnt/DATA"
    RUNS = 5

    dirs = os.listdir(DATA_DIR)

    for dir_ in dirs:

        prefix = dir_[0:3]

        if prefix == "SAC":

            pass

        elif prefix == "PPO":

            pass