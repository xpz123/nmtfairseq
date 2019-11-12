#!/bin/bash

#deal data
#python preprocess.py --source-lang raw --target-lang seg --trainpref ../../../data/sighan_seg/msr_sed_data/train --validpref ../../../data/sighan_seg/msr_sed_data/valid --testpref ../../../data/sighan_seg/msr_sed_data/test --destdir mydata/data-bin/msr --workers 5


#train transformer model
#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/msr --arch transformer_iwslt_de_en --save-dir checkpoints/exp1 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/msr --arch transformer_vaswani_wmt_en_de_big --save-dir checkpoints/exp2 --no-progress-bar --no-epoch-checkpoints --log-interval 300 --log-format simple  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 0.0005 --source-lang raw --target-lang seg --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' --max-update 50000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2048

#CUDA_VISIBLE_DEVICES=2 python train.py mydata/data-bin/msr --arch transformer_wmt_en_de --save-dir checkpoints/exp3 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/msr --arch transformer_wmt_en_de --save-dir checkpoints/exp4 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/msr --arch transformer_wmt_en_de --save-dir checkpoints/exp5 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas  '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --dropout 0.3

#CUDA_VISIBLE_DEVICES=2 python train.py mydata/data-bin/msr --arch transformer_wmt_en_de --save-dir checkpoints/exp6 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas  '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --dropout 0.1


CUDA_VISIBLE_DEVICES=0 python train.py mydata/data-bin/msr --arch transformer_iwslt_de_en --save-dir checkpoints/exp7 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#CUDA_VISIBLE_DEVICES=1 python train.py mydata/data-bin/msr --arch transformer_iwslt_de_en --save-dir checkpoints/exp8 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --encoder-layers 3 --decoder-layers 3

#CUDA_VISIBLE_DEVICES=2 python train.py mydata/data-bin/msr --arch transformer_iwslt_de_en --save-dir checkpoints/exp9 --log-interval 300 --no-progress-bar --no-epoch-checkpoints --log-format simple --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --encoder-layers 2 --decoder-layers 2



