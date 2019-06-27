#!/usr/bin/env sh
# Distributed under MIT license


#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:3
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

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

#script_dir=`dirname $0`
script_dir=/home/s1852803/unmt/wmt17/wmt17-transformer-scripts/training/scripts
main_dir=$script_dir/..
data_dir=$main_dir/data
working_dir=$main_dir/model

# variables (toolkits; source and target language)
. $main_dir/vars

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
devices=0,1,2

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
    --dictionaries $data_dir/corpus.bpe.$src.json \
                   $data_dir/corpus.bpe.$trg.json \
    --save_freq 1000 \
    --model $working_dir/model \
    --model_type rnn \
    --embedding_size 512 \
    --state_size 1024 \
    --rnn_enc_depth 1 \
    --rnn_enc_transition_depth 2 \
    --rnn_dec_depth 1 \
    --rnn_dec_base_transition_depth 2 \
    --tie_decoder_embeddings \
    --rnn_layer_normalisation \
    --rnn_dropout_hidden 0.5 \
    --rnn_dropout_embedding 0.5 \
    --rnn_dropout_source 0.3 \
    --rnn_dropout_target 0.3 \
    --loss_function cross-entropy \
    --label_smoothing 0.2 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-09 \
    --learning_schedule constant\
    --learning_rate 0.0005 \
    --patience 10 \
    --maxlen 200 \
    --token_batch_size 4000 \
    --valid_source_dataset $data_dir/newsdev2019.bpe.$src \
    --valid_target_dataset $data_dir/newsdev2019.bpe.$trg \
    --valid_token_batch_size 1000 \
    --valid_freq 1000 \
    --valid_script $script_dir/validate.sh \
    --disp_freq 10 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 5 \
    --translation_maxlen 200 \
