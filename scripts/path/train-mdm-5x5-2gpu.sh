#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=3
#SBATCH --nodes=1
#SBATCH --mem=36G
#SBATCH --output="/home/mila/b/buvanesa/code/nano-vppo/logs/out_%j.out"
#SBATCH --error="/home/mila/b/buvanesa/code/nano-vppo/logs/err_%j.err"
#SBATCH --partition=long
#SBATCH --job-name=test
#SBATCH --constraint="80gb"

export WANDB_DISABLED=true
module load anaconda/3
module load cuda/12.4.0/cudnn/8.9
conda activate diffusion
cd /home/mila/b/buvanesa/code/mini-rl-project/diff-vs-ar/

exp=output/path/mdm-alpha0.25-gamma1-bs1024-lr3e-4-ep1200-T20-ep-1200-5x5-medium`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu  --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20199 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_medium \
    --do_train \
    --dataset path_train \
    --max_samples 200000 \
    --finetuning_type full \
    --cutoff_len 250 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 5000.0 \
    --plot_loss \
    --run_name ${dataset}_prefix \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 1 \
    > $exp/train.log

for dataset in path_test
do
topk_decoding=True
mkdir $exp/$dataset
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 250 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $exp/${dataset}/eval-TopK$topk_decoding.log
    
python scripts/path/cal_path_acc.py data/path_test.jsonl $exp/${dataset}/generated_predictions.jsonl > $exp/pd_acc.log
done

