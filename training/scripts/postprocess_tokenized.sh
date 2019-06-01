#!/bin/sh
# Distributed under MIT license

# this sample script postprocesses the MT output,
# including merging of BPE subword units,
# detruecasing, and detokenization

script_dir=/home/s1852803/unmt/wmt17/wmt17-transformer-scripts/training/scripts
main_dir=$script_dir/../

# variables (toolkits; source and target language)
. $main_dir/vars

sed -r 's/\@\@ //g' |
$moses_scripts/recaser/detruecase.perl
