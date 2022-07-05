#!/bin/bash

## Code by LSS(github: teddy309)
## date: 2022.06.01 ~ Current(on updating)

#. conda/bin/activate
#conda activate beir

# 이건 뭐하는거지....
#run='srun -p gpu --gres=gpu:1'  # if slurm is available



USE_WANDB=False
RUN_PREPROCESS=True

DISABLE_CUDA=False
declare -i GPU_NUM=1
declare -i BATCH_SIZE=1

# 0- preprocess
#echo run printFEVER.py!!
# $run python beir_printFEVERcorpus.py
#echo run preprocessColloquial2Beir.py!!
#$run python preprocess_colloquial2beir.py -preprocess $RUN_PREPROCESS --colloquial_index 2

echo $USE_WANDB $RUN_PREPROCESS $DISABLE_CUDA $GPU_NUM $BATCH_SIZE

echo run beir_demo.py!!
# 1- basic(by tensorboard)
#$run python beir_demo.py
#$run python beir_demo.py --dataset_name colloquial --search_method DRES --model_name DPR --model_batch_size=$BATCH_SIZE
$run python beir_demo_ds.py --dataset_name colloquial --search_method DRES --model_name DPR --disable_cuda $DISABLE_CUDA --cuda_gpuNum=$GPU_NUM --model_batch_size=$BATCH_SIZE
#$run python beir_demo.py --dataset_name colloquial_0 #이거 하는중.(fever:OK, colloquial:하는중.)
#echo run beir_demo.py!! dataset colloquial_1
#$run python beir_demo.py --dataset_name colloquial_1
#echo run beir_demo.py!! dataset colloquial_2
#$run python beir_demo.py --dataset_name colloquial_2
# 2- with logger: (outdir at path './runs/')
#$run python beir_demo.py --use_wandb $USE_WANDB --dataset_name colloquial
# 3- with wandb
#$run python beir_demo.py --use_wandb $USE_WANDB