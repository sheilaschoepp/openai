module load python/3.7
virtualenv --no-download ~/envs/openai
pip install --no-index --upgrade pip
pip install numpy Cython pandas termcolor matplotlib cffi imageio pycparser lockfile imageio torch torchvision --no-index
pip install gym[all]