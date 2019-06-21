#!/usr/bin/env sh
# Distributed under MIT license


#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-80:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH
# this script evaluates the best model (according to BLEU early stopping)
# on newstest2017, using detokenized BLEU (equivalent to evaluation with
# mteval-v13a.pl)
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

script_dir=/home/s1852803/unmt/wmt17/wmt17-transformer-scripts/training/scripts
main_dir=$script_dir/../
data_dir=$main_dir/data
working_dir=$main_dir/model

# variables (toolkits; source and target language)
. $main_dir/vars

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
# Currently translate.py only uses a single GPU so there is no point
# assigning more than one.
devices=0

test_prefix=newstest2019
test=$test_prefix.bpe.$src
ref=$test_prefix.$trg
model=$working_dir/model.best-valid-script

# decode
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
     -m $model \
     -i $data_dir/$test \
     -o $working_dir/$test.output.dev \
     -k 12 \
     -n 0.6 \
     -b 10

# postprocess
$script_dir/postprocess.sh < $working_dir/$test.output.dev > $working_dir/$test.output.postprocessed.dev

# postprocess (no detokenization)
$script_dir/postprocess_tokenized.sh < $working_dir/$test.output.dev > $working_dir/$test.output.tokenized.dev

# evaluate with detokenized BLEU (same as mteval-v13a.pl)
echo "$test_prefix (detokenized BLEU)"
$nematus_home/data/multi-bleu-detok.perl $data_dir/$ref < $working_dir/$test.output.postprocessed.dev

# evaluate with tokenized BLEU
echo "$test_prefix (tokenized BLEU)"
$nematus_home/data/multi-bleu.perl $data_dir/$test_prefix.tok.$trg < $working_dir/$test.output.tokenized.dev
