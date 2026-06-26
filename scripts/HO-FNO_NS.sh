${PYTHON_BIN:-python3} hofno_ns.py \
--gpu 0 \
--width 256 \
--depth 8 \
--in_channels 12 \
--out_channels 1 \
--modes1 16 \
--modes2 16 \
--order 2 \
--p_drop_MLP 0 \
--drop_path_rate 0 \
--expansion_MLP 4 \
--lr 0.001 \
--epochs "${EPOCHS:-500}" \
--wandb_mode "${WANDB_MODE:-online}" \
--eta_min 0.000005 \
--batch-size 8 \
--eval 0 \
--save_name HO_FNO_NS_time \
#--2_3_zero_padding



