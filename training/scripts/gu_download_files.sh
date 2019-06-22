#!/bin/bash
# Downloads WMT17 training and test data for GU-EN
# Distributed under MIT license

script_dir=/home/s1852803/unmt/wmt17/wmt17-transformer-scripts/training/scripts
main_dir=$script_dir/..

# variables (toolkits; source and target language)
. $main_dir/vars

# get EN-DE training data for WMT17

if [ ! -f $main_dir/bible.gu-en.tsv.gz ];
then
  wget http://data.statmt.org/wmt19/translation-task/bible.gu-en.tsv.gz -O $main_dir/downloads/gu-en.tsv.gz
  gzip -d $main_dir/downloads/gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $main_dir/downloads/gu-en.tsv > $main_dir/downloads/gu-en.gu
  awk -F$'\t' '{print $2}' $main_dir/downloads/gu-en.tsv > $main_dir/downloads/gu-en.en
fi

if [ ! -f $main_dir/govin-clean.gu-en.tsv.gz ];
then
  wget http://data.statmt.org/wmt19/translation-task/govin-clean.gu-en.tsv.gz -O $main_dir/downloads/govin-clean.gu-en.tsv.gz
  gzip -d $main_dir/downloads/govin-clean.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $main_dir/downloads/govin-clean.gu-en.tsv > $main_dir/downloads/govin-clean.gu-en.gu
  awk -F$'\t' '{print $2}' $main_dir/downloads/govin-clean.gu-en.tsv > $main_dir/downloads/govin-clean.gu-en.en
fi

if [ ! -f $main_dir/wikipedia.gu-en.tsv.gz ];
then
  wget http://data.statmt.org/wmt19/translation-task/wikipedia.gu-en.tsv.gz -O $main_dir/downloads/wikipedia.gu-en.tsv.gz
  gzip -d $main_dir/downloads/wikipedia.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $main_dir/downloads/wikipedia.gu-en.tsv > $main_dir/downloads/wikipedia.gu-en.gu
  awk -F$'\t' '{print $2}' $main_dir/downloads/wikipedia.gu-en.tsv > $main_dir/downloads/wikipedia.gu-en.en
fi

if [ ! -f $main_dir/opus.gu-en.tsv.gz ];
then
  wget http://data.statmt.org/wmt19/translation-task/opus.gu-en.tsv.gz -O $main_dir/downloads/opus.gu-en.tsv.gz
  gzip -d $main_dir/downloads/opus.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $main_dir/downloads/opus.gu-en.tsv > $main_dir/downloads/opus.gu-en.gu
  awk -F$'\t' '{print $2}' $main_dir/downloads/opus.gu-en.tsv > $main_dir/downloads/opus.gu-en.en
fi

if [ ! -f $main_dir/downloads/dev.tgz ];
then
  wget http://data.statmt.org/wmt19/translation-task/dev.tgz -O $main_dir/downloads/dev.tgz
  tar -xf $main_dir/downloads/dev.tgz -C $main_dir/downloads
fi

if [ ! -f $main_dir/downloads/test.tgz ];
then
  wget http://data.statmt.org/wmt19/translation-task/test.tgz -O $main_dir/downloads/test.tgz
  tar -xf $main_dir/downloads/test.tgz -C $main_dir/downloads
fi


# concatenate all training corpora
cat $main_dir/downloads/gu-en.gu $main_dir/downloads/govin-clean.gu-en.gu $main_dir/downloads/wikipedia.gu-en.gu $main_dir/downloads/opus.gu-en.gu > $main_dir/data/corpus.gu
cat $main_dir/downloads/gu-en.en $main_dir/downloads/govin-clean.gu-en.en $main_dir/downloads/wikipedia.gu-en.en $main_dir/downloads/opus.gu-en.en > $main_dir/data/corpus.en

$moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newsdev2019-guen-ref.en.sgm > $main_dir/data/newsdev2019.en
$moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newsdev2019-guen-src.gu.sgm > $main_dir/data/newsdev2019.gu

$moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/sgm/newstest2019-guen-ref.en.sgm > $main_dir/data/newstest2019.en
$moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/sgm/newstest2019-guen-src.gu.sgm > $main_dir/data/newstest2019.gu


cd ..
