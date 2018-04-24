#!/usr/bin/env bash

#THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train.py --prototype prototype_ubuntu_HRED > Model_Output.txt 
#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu0 python sample.py Output/1496891647.84_UbuntuModel  /export/sdb/home/chenhongshen/MVHRED/data/UbuntuData/ResponseContextPairs/raw_testing_contexts.txt ubuntu_response.HRED --beam_search --n-samples=5 --ignore-unk --verbose 1>Test_Output.txt 2>&1

THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32 python train.py --prototype prototype_jd_MVHRED --pretrained Output/1500877901.41_JDModel> Model_Output.txt

#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu3 python  sample.py Output/1499074821.87_DoubanModel /export/sdb/home/chenhongshen/MVHRED/data/Douban/test.txt.raw.context douban.MVHRED.response --beam_search --n-samples=5 --ignore-unk --verbose 2>Douban.MVHRED.Test_Output.txt 
