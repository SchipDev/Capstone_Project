#! /bin/bash
#
#SBATCH --account=cmda_capstone_2021
#SBATCH -t 06:00:00
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH -p a100_dev_q
#

#Load modules
module reset
export PYTHONUSERBASE=/projects/cmda_capstone_2021_ti/pythonlibs
module load Anaconda3/2020.11
module load TensorFlow

#Run beginner tutorial
echo "TENSORFLOW_TINKERCLIFFS_A100: Normal beginning of execution."
python trainingset_extractor.py 10000
python model_iter1/train.py
echo "TENSORFLOW_TINKERCLIFFS_A100: Normal end of execution."
