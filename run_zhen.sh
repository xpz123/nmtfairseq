#!/bin/bash

#deal data
#python preprocess.py --source-lang zh --target-lang en --trainpref data_zhen/train --validpref data_zhen/val --testpref data_zhen/test --destdir mydata/data-bin/charZh_wordEn --workers 5 --nwordstgt 50000
#python preprocess.py --source-lang zh --target-lang en --trainpref ../../../data/UM-Corpus/data/train --validpref ../../../data/UM-Corpus/data/val --testpref ../../../data/UM-Corpus/data/test --destdir mydata/data-bin/charZh_wordEn_um --workers 5 --nwordstgt 50000 --nwordssrc 6000


#train lstm model (um)
#CUDA_VISIBLE_DEVICES=2  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_myNMT --save-dir checkpoints/ZhEn_lstm/um_charWord_exp1 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.005 --lr-shrink 0.5 --max-tokens 4096 --dropout 0.3

#CUDA_VISIBLE_DEVICES=3  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_wiseman_iwslt_de_en --save-dir checkpoints/ZhEn_lstm/um_charWord_exp2 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.005 --lr-shrink 0.5 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=3  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_myNMT --save-dir checkpoints/ZhEn_lstm/um_charWord_exp3 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.001 --lr-shrink 0.5 --max-tokens 4096 --dropout 0.3

#CUDA_VISIBLE_DEVICES=2  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_myNMT --save-dir checkpoints/ZhEn_lstm/um_charWord_exp4 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.0005 --lr-shrink 0.5 --max-tokens 4096 --dropout 0.3

#exp5 in shixiong

#CUDA_VISIBLE_DEVICES=3  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_myNMT --save-dir checkpoints/ZhEn_lstm/um_charWord_exp6 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.0001 --lr-shrink 0.5 --max-tokens 4096 --encoder-embed-dim 256 --encoder-layers 2 --encoder-hidden-size 512 --decoder-embed-dim 256 --decoder-layers 2 --decoder-out-embed-dim  256 --decoder-hidden-size 512 --dropout 0.3

#CUDA_VISIBLE_DEVICES=1  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_myNMT --save-dir checkpoints/ZhEn_lstm/um_charWord_exp7 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.001 --lr-shrink 0.5 --max-tokens 4096 --encoder-embed-dim 512 --encoder-layers 2 --encoder-hidden-size 512 --decoder-embed-dim 512 --decoder-layers 2 --decoder-out-embed-dim  512 --decoder-hidden-size 512  --dropout 0.3

#exp8,exp9,exp10 in shixiong

CUDA_VISIBLE_DEVICES=2  python train.py mydata/data-bin/charZh_wordEn_um --arch lstm_myNMT --save-dir checkpoints/ZhEn_lstm/um_charWord_exp11 --log-interval 1000 --no-progress-bar --log-format simple --optimizer adam --lr 0.0005 --lr-shrink 0.5 --max-tokens 4096 --encoder-embed-dim 512 --encoder-layers 2 --encoder-hidden-size 512 --decoder-embed-dim 512 --decoder-layers 2 --decoder-out-embed-dim  512 --decoder-hidden-size 512  --dropout 0.3



