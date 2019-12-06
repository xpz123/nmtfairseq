#!/bin/bash

#evaluate experiment model

#only have source txt.
#CUDA_VISIBLE_DEVICES=3 python interactive.py mydata/data-bin/charTi_charZh --task translation --source-lang ti --target-lang zh --path checkpoints/charChar_exp1/checkpoint_best.pt --buffer-size 500 --batch-size 1 --beam 5 --remove-bpe --input ../../../data/CCMT2019/data_20191101/charTi_charZh/test-2019.ti

#have source txt and target txt.
#--------Exp1------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp1/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp1/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp1/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp2------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp2/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp2/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp2/checkpoint_best.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp3------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp3/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp3/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp3/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp4------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp4/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp4/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp4/checkpoint_best.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp5------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp5/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp5/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp5/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp6------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp6/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp6/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp6/checkpoint_best.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp7------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp7/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp7/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp7/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp8------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp8/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp8/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp8/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp9------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp9/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp9/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp9/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp10------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp10/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp10/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp10/checkpoint_best.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp11------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp11/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp11/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp11/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp12------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp12/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp12/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp12/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp13------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp13/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp13/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp13/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp14------------------
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp14/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp14/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp14/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp15------------------
#CUDA_VISIBLE_DEVICES=2 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp15/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=2 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp15/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=2 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp15/checkpoint_best.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#--------Exp16------------------
CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_charZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp16/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/charTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/charWord_exp16/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/wordTi_wordZh_v2 --task translation --log-format simple --log-interval 10 --path checkpoints/wordWord_exp16/checkpoint_last.pt --batch-size 1 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50


