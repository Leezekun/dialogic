CUDA_VISIBLE_DEVICES=2,3 python ../../../inference_pptod.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --data_version 2.3\
    --model_name t5-base\
    --use_db_as_input True\
    --pretrained_path ../../../ckpt23/base/few_shot_0.1/\
    --output_save_path ../../../inference_result23/base/few_shot_0.1/\
    --number_of_gpu 2\
    --batch_size_per_gpu 128\
    --train_data_ratio 0.1\
    --eva_mode train