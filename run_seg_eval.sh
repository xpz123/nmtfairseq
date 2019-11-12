#!/bin/bash

#evaluate experiment model

#have source txt and target txt.
#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp1/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp2/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp3/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp4/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp5/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp6/checkpoint_last.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50

#CUDA_VISIBLE_DEVICES=3 python generate.py mydata/data-bin/msr --task translation --log-format simple --log-interval 10 --path checkpoints/exp3/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --results-path temp --lenpen 0.6 --max-len-a 1 --max-len-b 50


