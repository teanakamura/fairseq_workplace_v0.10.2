## Initialize module
source /etc/profile.d/modules.sh

module load python/3.6/3.6.5
#module load cuda/9.1/9.1.85.3
module load cuda/10.1/10.1.243
#module load cudnn/7.1/7.1.3
module load cudnn/7.6/7.6.4
#module load openmpi/2.1.6
#module load nccl/2.1/2.1.15-1

source ~/venv/fairseq_v0.10.2/bin/activate
