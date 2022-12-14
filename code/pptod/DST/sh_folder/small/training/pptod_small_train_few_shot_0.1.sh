CUDA_VISIBLE_DEVICES=2 python ../../../learn.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-2.3-fine-processed/\
    --model_name t5-small\
    --delx_response False\
    --pretrained_path ../../../../checkpoints/small/\
    --ckpt_save_path ../../../ckpt23/small/few_shot_0.1/2_shot_augx2\
    --epoch_num 100\
    --gradient_accumulation_steps 4\
    --number_of_gpu 1\
    --batch_size_per_gpu 32\
    --train_data_ratio 0.01\
    --aug_data_balance True\
    --aug_train_data_file multi-woz-fine-processed-train-2_shot_augment_x2_dst_turn_info_train_ratio_0.1.json