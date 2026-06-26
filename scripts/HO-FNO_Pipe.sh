${PYTHON_BIN:-python3} hofno_pipe.py \
--gpu 1 \
--width 128 \
--depth 8 \
--in_channels 4 \
--out_channels 1 \
--modes1 50 \
--modes2 24 \
--order 2 \
--p_drop_MLP 0 \
--drop_path_rate 0.1 \
--expansion_MLP 4 \
--lr 0.001 \
--epochs "${EPOCHS:-500}" \
--wandb_mode "${WANDB_MODE:-online}" \
--eta_min 0.000005 \
--batch-size 8 \
--eval 0 \
--save_name HO-FNO_Pipe_2_3_zero_pad \
--2_3_zero_padding



