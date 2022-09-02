CUDA_VISIBLE_DEVICES=0 python ../../../dialogic_simulation.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --data_version 2.3\
    --model_name t5-large\
    --pretrained_path ../../../ckpt23/large/few_shot_0.1/\
    --output_save_path ../../../simulation_result23/large/few_shot_0.1/\
    --train_data_ratio 0.1\
    --augment_dialog_path ../../../../data/multiwoz/data/multi-woz-2.3-dialogic-processed/combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.1.json\
    --log_path ../../../simulation_result23/large/few_shot_0.1/dialogic_simulation.log\
    --max_turn_num 10\
    --max_dialog_num 843\
    --max_aug_time 1\
    --verify_bs True\
    --verify_da True\
    --use_db_as_input True\
    --n_user 1\
    --n_system 1\
    --debug True\
    --save True