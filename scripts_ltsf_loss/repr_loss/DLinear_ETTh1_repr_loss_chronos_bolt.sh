#!/bin/bash

#SBATCH --job-name=DLinear_ETTh1_repr_loss_chronos_bolt_alpha_0.5_predict_336_batch_not_shuffle
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10G
#SBATCH --nodelist=laal2
#SBATCH --partition=laal_a6000
#SBATCH --output=/home/jinkwanjang/Time-Series-Library/log/DLinear_input_96_repr_loss_chronos/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate timeSeries

export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

# python -u /home/jinkwanjang/Time-Series-Library/run_repr_loss.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --batch_size 8 \
#   --seq_len 96 \
#   --label_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --repr_loss \
#   --repr_model chronos_bolt \
#   --base MSE \
#   --alpha 0.5 \


python -u /home/jinkwanjang/Time-Series-Library/run_repr_loss.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --batch_size 32 \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --repr_loss \
  --repr_model chronos_bolt \
  --base MSE \
  --alpha 0.5 \
  --sample_stride \
  --batch_not_shuffle True \
