#!/bin/bash
#SBATCH --job-name=regressor_train
#SBATCH -o logs/%x_%J.out
#SBATCH -e logs/%x_%J.err
#SBATCH --mem=256g
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

# Usage: sbatch -J <EXPERIMENT_NAME> sbatch_train_regressor.sh <CONFIG_PATH>
#   sbatch -J pitch sbatch_train_regressor.sh code/body_latent_regressor_pitch_config.json
#   sbatch -J full  sbatch_train_regressor.sh code/body_latent_regressor_config.json
CONFIG_PATH=$1

# Safety checks
if [ -z "$CONFIG_PATH" ]; then
  echo "Error: No config file path provided."
  echo "Usage: sbatch -J <EXPERIMENT_NAME> sbatch_train_regressor.sh path/to/config.json"
  exit 1
fi

echo "started"

# Navigate to the project root
cd /cs/labs/tsevi/lior.kotlar/headon-flight

# Create logs and analysis directories if they don't exist
mkdir -p logs
mkdir -p data/analysis

# Activate the virtual environment
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Config File: $CONFIG_PATH"
echo "Experiment Name: $SLURM_JOB_NAME"

# Execute the body->latent regressor grid search
python code/body_latent_regressor.py --config "$CONFIG_PATH" --job_name "$SLURM_JOB_NAME"

echo "finished working"
