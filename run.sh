#!/bin/bash

#deal data
#python preprocess.py --source-lang ti --target-lang zh --trainpref ../../../data/CCMT2019/data_20191101/charTi_charZh/train --validpref ../../../data/CCMT2019/data_20191101/charTi_charZh/valid --testpref ../../../data/CCMT2019/data_20191101/charTi_charZh/test-2017 --destdir data/data-bin/charTi_charZh --workers 5

#python preprocess.py --source-lang ti --target-lang zh --trainpref ../../../data/CCMT2019/data_20191101/charTi_wordZh/train --validpref ../../../data/CCMT2019/data_20191101/charTi_wordZh/valid --testpref ../../../data/CCMT2019/data_20191101/charTi_wordZh/test-2017 --destdir data/data-bin/charTi_wordZh --workers 5

#python preprocess.py --source-lang ti --target-lang zh --trainpref ../../../data/CCMT2019/data_20191101/wordTi_wordZh/train --validpref ../../../data/CCMT2019/data_20191101/wordTi_wordZh/valid --testpref ../../../data/CCMT2019/data_20191101/wordTi_wordZh/test-2017 --destdir data/data-bin/wordTi_wordZh --workers 5

#python preprocess.py --source-lang ti --target-lang ch --trainpref data/train --validpref data/valid --testpref data/test --destdir data/data-bin/charCCMT_new --workers 5


#train transformer model
#CUDA_VISIBLE_DEVICES=0 python train.py data/data-bin/charTi_charZh --arch transformer_iwslt_de_en --save-dir checkpoints/charChar_exp1 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=1 python train.py data/data-bin/charTi_wordZh --arch transformer_iwslt_de_en --save-dir checkpoints/charWord_exp1 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=2 python train.py data/data-bin/wordTi_wordZh --arch transformer_iwslt_de_en --save-dir checkpoints/wordWord_exp1 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#evaluate experiment model
CUDA_VISIBLE_DEVICES=3 python generate.py data/data-bin/wordTi_wordZh --task translation --log-format simple --log-interval 10 --path checkpoints/charChar_exp1/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --results-path temp


#CUDA_VISIBLE_DEVICES=1 python train.py data-bin/ti_ch --arch transformer_vaswani_wmt_en_de_big --save-dir checkpoints/exp2 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000

#CUDA_VISIBLE_DEVICES=2 python train.py data-bin/ti_ch --arch transformer_vaswani_wmt_en_fr_big --save-dir checkpoints/exp3 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000

#CUDA_VISIBLE_DEVICES=3 python train.py data-bin/ti_ch --arch transformer_vaswani_wmt_en_fr_big --save-dir checkpoints/exp4 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0001 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000

#CUDA_VISIBLE_DEVICES=1 python train.py data-bin/ti_ch --arch transformer_wmt_en_de_big_t2t --save-dir checkpoints/exp5 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0001 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000

#CUDA_VISIBLE_DEVICES=2 python train.py data-bin/ti_ch --arch transformer_wmt_en_de_big_t2t --save-dir checkpoints/exp6 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000

#CUDA_VISIBLE_DEVICES=1 python train.py data-bin/ti_ch --arch transformer_wmt_en_de_big_t2t --save-dir checkpoints/exp7 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2048

#CUDA_VISIBLE_DEVICES=2 python train.py data-bin/ti_ch --arch transformer_wmt_en_de_big_t2t --save-dir checkpoints/exp8 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 1024

#CUDA_VISIBLE_DEVICES=3 python train.py data-bin/ti_ch --arch transformer_wmt_en_de_big_t2t --save-dir checkpoints/exp8_2 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang ti --target-lang zh --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000  --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 1024


#evaluate experiment model
CUDA_VISIBLE_DEVICES=7 python generate.py data/data-bin/charCCMT --task translation --log-format simple --log-interval 10 --path result/exp2/checkpoint_last.pt --batch-size 128 --beam 5 --remove-bpe --results-path temp

