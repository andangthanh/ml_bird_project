#!/bin/bash

#SBATCH --time=24:00:00         	# walltime
#SBATCH --nodes=1               	# number of nodes
#SBATCH --gres=gpu:1            	# number of GPUS
#SBATCH --ntasks=1              	# limit to one node
#SBATCH --cpus-per-task=10       	# number of processor cores (i.e. threads)
#SBATCH --partition=alpha          	# defines access to hardware and software modules
#SBATCH --mem-per-cpu=8000      	# memory per CPU core
#SBATCH -J "5mask"       		# job name
#SBATCH -A p_ml_birds   				# credit to your project 

#SBATCH -o /lustre/scratch2/ws/0/s4030475-ml_birds_project/output/slurm-%j.out     # save output messages %j is job-id
#SBATCH -e /lustre/scratch2/ws/0/s4030475-ml_birds_project/output/slurm-%j.err     # save error messages %j is job-id

#SBATCH --mail-type=end# send email notification when job finished
#SBATCH --mail-user=an.dang_thanh@mailbox.tu-dresden.de

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
module load modenv/hiera # loads the alpha environment
module load GCC/10.2.0
module load CUDA/11.1.1
module load OpenMPI/4.0.5
module load PyTorch/1.9.0
module load matplotlib
module load scikit-learn

source /lustre/scratch2/ws/0/s4030475-ml_birds_project/virt_envs/alpha-test-env2/bin/activate
python /lustre/scratch2/ws/0/s4030475-ml_birds_project/code/specAug_5maskLen.py.py

exit 0