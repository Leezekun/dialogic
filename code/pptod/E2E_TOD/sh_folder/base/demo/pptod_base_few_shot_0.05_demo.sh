CUDA_VISIBLE_DEVICES=0 python ../../../dialogic_demo.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --data_version 2.3\
    --model_name t5-base\
    --pretrained_path ../../../ckpt23/base/few_shot_0.05/\
    --output_save_path ../../../simulation_result23/base/few_shot_0.05/\
    --use_db_as_input True\
    --train_data_ratio 0.05\
    --augment_dialog_path ../../../../data/multiwoz/data/multi-woz-2.3-dialogic-processed/combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.05.json\
    --log_path ../../../simulation_result23/base/few_shot_0.05/dialogic_simulation.log\
    --max_turn_num 10\
    --verify_bs True\
    --verify_da True\
    --n_user 1\
    --n_system 1\
    --debug True\
    --save True\
    --pause True\
    --input_user_goal True\
    --k_shot 2\
    --temperature 0.2