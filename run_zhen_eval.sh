#!/bin/bash

#evaluate experiment model
#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/charWord_exp1/checkpoint33.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/charWord_exp2/checkpoint_best.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#um data
#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp1/checkpoint_best.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp2/checkpoint_best.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp3/checkpoint_best.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp4/checkpoint_last.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp6/checkpoint_best.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp7/checkpoint10.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

CUDA_VISIBLE_DEVICES=0 python generate.py  mydata/data-bin/charZh_wordEn_um --task translation --log-format simple --log-interval 10 --path checkpoints/ZhEn_lstm/um_charWord_exp11/checkpoint_best.pt --batch-size 16 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50



