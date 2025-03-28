export TASK_NAME=RTE

torchrun --nproc_per_node=4 --master_port=12456 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir /scratch/gpfs/zs8839/cos5682/cos568proj2/data/glue_data/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir /scratch/gpfs/zs8839/cos5682/cos568proj2/tmp \
  --overwrite_output_dir \
  --master_ip 127.0.0.1 \
  --master_port 12456 \
  --world_size 4
