import os


def ant():

    ant_data_dir = DATA_DIR + "/ant"

    os.listdir(ant_data_dir)


def fetchreach():

    pass


if __name__ == "__main__":

    DATA_DIR = "/mnt/DATA/shared"

    RUNS = 10

    ant()
    fetchreach()
