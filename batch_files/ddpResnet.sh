#!/bin/bash

#SBATCH --time=12:00:00         	# walltime
#SBATCH --nodes=1               	# number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4            	# number of GPUS
#SBATCH --cpus-per-task=10       	# number of processor cores (i.e. threads)
#SBATCH --partition=alpha          	# defines access to hardware and software modules
#SBATCH --mem-per-cpu=10000      	# memory per CPU core
#SBATCH -J "ddpResNet"       		# job name
#SBATCH -A p_ml_birds   				# credit to your project 

#SBATCH -o /lustre/scratch2/ws/0/s4030475-ml_birds_project/output/slurm-%j.out     # save output messages %j is job-id
#SBATCH -e /lustre/scratch2/ws/0/s4030475-ml_birds_project/output/slurm-%j.err     # save error messages %j is job-id

#SBATCH --mail-type=end# send email notification when job finished
#SBATCH --mail-user=an.dang_thanh@mailbox.tu-dresden.de
export MASTER_PORT=13370
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
module load modenv/hiera # loads the alpha environment
module load GCC/10.2.0
module load CUDA/11.1.1
module load OpenMPI/4.0.5
module load PyTorch/1.9.0
module load matplotlib
module load scikit-learn

source /lustre/scratch2/ws/0/s4030475-ml_birds_project/virt_envs/alpha-test-env2/bin/activate
srun python /lustre/scratch2/ws/0/s4030475-ml_birds_project/code/ddpResNet.py

exit 0