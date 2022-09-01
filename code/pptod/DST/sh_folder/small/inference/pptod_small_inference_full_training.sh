CUDA_VISIBLE_DEVICES=5,6 python ../../../inference.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-2.3-fine-processed/\
    --model_name t5-small\
    --pretrained_path ../../../ckpt/small/full_training/\
    --output_save_path ../../../inference_result23/small/full_training/\
    --number_of_gpu 2\
    --batch_size_per_gpu 64\
    --input_test_path ../../../../E2E_TOD/inference_result23/small/full_training/train_0.01_inference_result_e2e_evaluation_inform_64.71_success_56.47_bleu_16.15_combine_score_76.74.json\
    --eva_mode test