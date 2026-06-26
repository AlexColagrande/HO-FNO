${PYTHON_BIN:-python3} hofno_darcy.py \
--gpu 1 \
--width 256 \
--depth 8 \
--in_channels 3 \
--out_channels 1 \
--modes1 32 \
--modes2 16 \
--order 2 \
--p_drop_MLP 0 \
--drop_path_rate 0.1 \
--expansion_MLP 4 \
--lr 0.001 \
--epochs "${EPOCHS:-500}" \
--wandb_mode "${WANDB_MODE:-online}" \
--eta_min 0.000005 \
--max_grad_norm 0.1 \
--batch-size 32 \
--slice_num 64 \
--eval 0 \
--downsample 5 \
--save_name HO_FNO_Darcy \
--2_3_zero_padding \

# Matrix ablation: Diagonal=Depthwise, Lowrank=conv(in_C, r)conv(r, out_C), no_proj=identity, no_bias, SAME MATRICES. Other initializations 
