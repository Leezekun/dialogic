CUDA_VISIBLE_DEVICES=1 python ../../../learn.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --data_version 2.3\
    --model_name t5-base\
    --pretrained_path ../../../../checkpoints/base/\
    --ckpt_save_path ../../../ckpt23/base/full_training/\
    --use_db_as_input True\
    --epoch_num 50\
    --gradient_accumulation_steps 2\
    --number_of_gpu 1\
    --batch_size_per_gpu 32\
    --train_data_ratio 1.0