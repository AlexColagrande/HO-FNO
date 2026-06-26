${PYTHON_BIN:-python3} hofno_airfoil.py \
--gpu 1 \
--width 128 \
--depth 8 \
--in_channels 4 \
--out_channels 1 \
--modes1 24 \
--modes2 12 \
--order 2 \
--p_drop_MLP 0 \
--drop_path_rate 0.1 \
--expansion_MLP 4 \
--lr 0.001 \
--epochs "${EPOCHS:-500}" \
--wandb_mode "${WANDB_MODE:-online}" \
--eta_min 0.000005 \
--batch-size 20 \
--eval 0 \
--save_name HO_FNO_Airfoil_2_3_zero_pad \
--2_3_zero_padding 



