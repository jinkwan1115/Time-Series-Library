#!/bin/bash

#SBATCH --job-name=DLinear_individual_false_ETTh1_fourier_complex_MAE_alpha_0.0_input_336
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10G
#SBATCH --nodelist=laal1
#SBATCH --partition=laal_3090
#SBATCH --output=/home/jinkwanjang/Time-Series-Library/log/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate timeSeries

export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

python -u /home/jinkwanjang/Time-Series-Library/run_mae.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --combined_loss \
  --alpha_additional_loss 0 \
  --use_freq \
  --freq_loss_type complex \
  --loss 'MAE' \
  --use_dilate True \
  --alpha_dilate 0.5 \
  --gamma_dilate 0.01 \

python -u /home/jinkwanjang/Time-Series-Library/run_mae.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --combined_loss \
  --alpha_additional_loss 0 \
  --use_freq \
  --freq_loss_type complex \
  --loss 'MAE' \
  --use_dilate True \
  --alpha_dilate 0.5 \
  --gamma_dilate 0.01 \

python -u /home/jinkwanjang/Time-Series-Library/run_mae.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --combined_loss \
  --alpha_additional_loss 0 \
  --use_freq \
  --freq_loss_type complex \
  --loss 'MAE' \
  --use_dilate True \
  --alpha_dilate 0.5 \
  --gamma_dilate 0.01 \

python -u /home/jinkwanjang/Time-Series-Library/run_mae.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --combined_loss \
  --alpha_additional_loss 0 \
  --use_freq \
  --freq_loss_type complex \
  --loss 'MAE' \
  --use_dilate True \
  --alpha_dilate 0.5 \
  --gamma_dilate 0.01 \