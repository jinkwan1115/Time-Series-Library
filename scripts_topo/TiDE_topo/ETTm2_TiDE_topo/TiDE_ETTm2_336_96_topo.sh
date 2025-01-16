#!/bin/bash

#SBATCH --job-name=TiDE_ETTm2_336_96_topo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10G
#SBATCH --nodelist=laal1
#SBATCH --partition=laal_3090
#SBATCH --output=/home/jinkwanjang/Time-Series-Library/log/TiDE_topo/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate timeSeries

export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

python -u /home/jinkwanjang/Time-Series-Library/run_tide_topo.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.3 \
  --batch_size 512 \
  --learning_rate 0.1 \
  --patience 5 \
  --train_epochs 30 \
  --des 'Exp' \
  --itr 1 \
  --freq t \
  --use_ph True \

# python -u /home/jinkwanjang/Time-Series-Library/run_tide.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_336_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 336 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 256 \
#   --dropout 0.3 \
#   --batch_size 512 \
#   --learning_rate 0.1 \
#   --patience 5 \
#   --train_epochs 10 \
#   --des 'Exp' \
#   --itr 1 \
#   --freq h \
#   --embed 'timeF' \
#   --combined_loss \
#   --alpha_additional_loss 0 \
#   --use_freq \
#   --freq_loss_type complex \
#   --use_ph \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 256 \
#   --dropout 0.3 \
#   --batch_size 512 \
#   --learning_rate 0.1 \
#   --patience 5 \
#   --train_epochs 10 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 256 \
#   --dropout 0.3 \
#   --batch_size 512 \
#   --learning_rate 0.1 \
#   --patience 5 \
#   --train_epochs 10 \




# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 8 \
#   --d_model 256 \
#   --d_ff 256 \
#   --dropout 0.3 \
#   --batch_size 512 \
#   --learning_rate 0.1 \
#   --patience 5 \
#   --train_epochs 10 \

