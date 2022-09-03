CUDA_VISIBLE_DEVICES=0 python ../../../dialogic_simulation.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --data_version 2.3\
    --model_name t5-base\
    --pretrained_path ../../../ckpt23/base/full_training/\
    --output_save_path ../../../simulation_result23/base/full_training/\
    --train_data_ratio 1.0\
    --augment_dialog_path ../../../../data/multiwoz/data/multi-woz-2.3-dialogic-processed/simplified_combine_0.2_2_shot_augment_dialog_turn_info_train_ratio_1.0.json\
    --log_path ../../../simulation_result23/base/full_training/dialogic_simulation.log\
    --max_turn_num 12\
    --max_dialog_num 85\
    --max_aug_time 1\
    --verify_bs True\
    --verify_da True\
    --use_db_as_input True\
    --n_user 1\
    --n_system 1\
    --debug True\
    --save True