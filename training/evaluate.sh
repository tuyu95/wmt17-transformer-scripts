#!/bin/sh

# this script evaluates the best model (according to BLEU early stopping)
# on newstest2017, using detokenized BLEU (equivalent to evaluation with
# mteval-v13a.pl)

script_dir=`dirname $0`
data_dir=$script_dir/data
working_dir=$script_dir/model

#language-independent variables (toolkit locations)
. $script_dir/../vars

#language-dependent variables (source and target language)
. $script_dir/vars

test=newstest2017.bpe.$src
ref=newstest2017.$tgt
model=$working_dir/model.npz.best_bleu


# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,gpuarray.preallocate=0.1 time python $nematus_home/nematus/translate.py \
     -m $model \
     -i $data_dir/$testv -o $working_dir/$test.output.dev -k 12 -n -p 1 --suppress-unk

# postprocess
$script_dir/postprocess.sh < $working_dir/$test.output.dev > $working_dir/$test.output.postprocessed.dev

# evaluate with detokenized BLEU (same as mteval-v13a.pl)
$nematus_home/data/multi-bleu-detok.perl $data_dir/$ref < $working_dir/$test.output.postprocessed.dev
