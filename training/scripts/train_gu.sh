#!/usr/bin/env sh
# Distributed under MIT license


#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH



#script_dir=`dirname $0`
script_dir=/home/s1852803/unmt/wmt17/wmt17-transformer-scripts/training/scripts
main_dir=$script_dir/..
data_dir=$main_dir/data
working_dir=$main_dir/model

# variables (toolkits; source and target language)
. $main_dir/vars

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
devices=0,1,2,3

# Training command that closely follows the 'base' configuration from the
# paper
#
#  "Attention is All you Need" in Advances in Neural Information Processing
#  Systems 30, 2017. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
#  Uszkoreit, Llion Jones, Aidan N Gomez, Lukadz Kaiser, and Illia Polosukhin.
#
# Depending on the size and number of available GPUs, you may need to adjust
# the token_batch_size parameter. The command used here was tested on a
# machine with four 12 GB GPUS.
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $data_dir/corpus.bpe.$src \
    --target_dataset $data_dir/corpus.bpe.$trg \
    --dictionaries $data_dir/corpus.bpe.both.json \
                   $data_dir/corpus.bpe.both.json \
    --save_freq 30000 \
    --model $working_dir/model \
    --reload latest_checkpoint \
    --model_type transformer \
    --embedding_size 512 \
    --state_size 512 \
    --tie_encoder_decoder_embeddings \
    --tie_decoder_embeddings \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --learning_schedule transformer \
    --warmup_steps 4000 \
    --maxlen 100 \
    --batch_size 128 \
    --token_batch_size 4096 \
    --valid_source_dataset $data_dir/newstest2013.bpe.$src \
    --valid_target_dataset $data_dir/newstest2013.bpe.$trg \
    --valid_batch_size 64 \
    --valid_token_batch_size 1024 \
    --valid_freq 10000 \
    --valid_script $script_dir/validate.sh \
    --disp_freq 1000 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 4 \
    --translation_maxlen 100 \
    --normalization_alpha 0.6
