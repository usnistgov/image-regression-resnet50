#!/bin/bash

# **************************
# START - MODIFY THESE OPTIONS

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --job-name=resnet50
#SBATCH -o rn50_%N.%j.out
#SBATCH --time=24:0:0

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# job configuration
test_every_n_steps=1000
batch_size=8 # 4x across the gpus

train_lmdb_file="train.lmdb"
test_lmdb_file="test.lmdb"

input_data_directory="/wrk/mmajursk/tmp"
output_directory="/wrk/mmajursk/tmp"

experiment_name="resnet50-$(date +%Y%m%dT%H%M%S)"

number_classes=2
learning_rate=3e-4
use_augmentation=1
balance_classes=1

# END - MODIFY THESE OPTIONS
# **************************


echo "Experiment: $experiment_name"
scratch_dir="/scratch/${SLURM_JOB_ID}"

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
term_handler()
{
        echo "function term_handler called.  Cleaning up and Exiting"
        # Do nothing
        exit -1
}

# associate the function "term_handler" with the TERM signal
trap 'term_handler' TERM

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf2

# make working directory
mkdir -p ${scratch_dir}
echo "Created Directory: $scratch_dir"

# copy data to node
echo "Copying data to Node"
cp -r ${input_data_directory}/${train_lmdb_file} ${scratch_dir}/${train_lmdb_file}
cp -r ${input_data_directory}/${test_lmdb_file} ${scratch_dir}/${test_lmdb_file}
echo "data copy to node complete"
echo "Working directory contains: "
ls ${scratch_dir}

results_dir="$output_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

mkdir -p "$results_dir/src"
cp -r . "$results_dir/src"

# launch training script with required options
echo "Launching Training Script"

python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$scratch_dir/$train_lmdb_file" --test_database="$scratch_dir/$test_lmdb_file" --output_dir="$results_dir" --number_classes=${number_classes} --learning_rate=${learning_rate}  --use_augmentation=${use_augmentation} --balance_classes=${balance_classes} | tee "$results_dir/log.txt"

echo "Job completed"
