#! /bin/bash

dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=128
tr_max_sample_points=1000
te_max_sample_points=1000
lr=2e-3
output_steps=1
epochs=2000
ds=dynabs
env_name='MultiTool-v1'
data_dirs='data/autobot/1024_gen_multitool'
# data_dirs='data/autobot/1024_gen_multitool/1024_gen_multitool_2022_10_25_00_49_30_0001/dataset.gz'
log_name="0514_${output_steps}steps_fulldata"

python PointFlow/train.py \
    --log_name ${log_name} \
    --env_name ${env_name} \
    --lr ${lr} \
    --train_T False \
    --dataset_type ${ds} \
    --filter_buffer False \
    --cached_state_path 'datasets/1025_multitool' \
    --data_dirs ${data_dirs} \
    --tr_max_sample_points=${tr_max_sample_points}\
    --te_max_sample_points=${te_max_sample_points}\
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 5 \
    --viz_freq 5 \
    --log_freq 1 \
    --val_freq 1 \
    --output_steps ${output_steps} \
    --use_latent_flow \
    --distributed \

echo "Done"
# exit 0
