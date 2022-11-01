#!/bin/bash
#SBATCH --job-name=1-cpu-test         
#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4        
#SBATCH --mem-per-cpu=4G         
#SBATCH --gres=gpu:1             
#SBATCH --time=00:30:00          
#SBATCH --mail-type=end          
#SBATCH --mail-user=melhabr@princeton.edu

module purge
module load anaconda3/2022.5
conda activate torch-env

python ../train_record_loss.py -a resnet50 /scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_2017_face_obfuscation