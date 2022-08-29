from functools import total_ordering
import os
from selectors import EpollSelector
import sys
import json
import torch
import random
import argparse
import operator
import progressbar
import torch.nn as nn
from operator import itemgetter
import torch.nn.functional as F

def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    ckpt_name = ""
    for name in file_names:
        if name.startswith('epoch'):
            ckpt_name = name
            if 'best' in name:
                print (name)
                return name
    if ckpt_name:
        print (ckpt_name)
        return ckpt_name
    else:
        raise Exception("Cannot find ckpt!")
            

def zip_result(prediction):
    result = {}
    for turn in prediction:
        dial_id = turn['dial_id']
        turn_idx = turn['turn_num']
        try:
            result[dial_id][turn_idx] = turn
        except KeyError:
            result[dial_id] = {}
            result[dial_id][turn_idx] = turn
    return result

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')
    parser.add_argument('--pretrained_path', type=str, help='the path that stores pretrained checkpoint.')
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--eva_mode", type=str, default="test", help="evaluation mode: test, dev, train or all.")

    parser.add_argument("--output_save_path", type=str, help="directory to save the model output.")
    parser.add_argument("--input_test_path", type=str, help="directory to the tested file.")

    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
 
    args = parse_config()
    device = torch.device('cuda')

    # **********************************************************************
    # --- evaluation --- #
    from inference_utlis import batch_generate
    print ('Start evaluation...')

    # --- evaluation on the input data --- #
    if args.input_test_path:
        import json
        with open(args.input_test_path, 'r') as inputfile:
            all_dev_result = json.load(inputfile)

    # --- evaluation on all the inference data --- #
    else:
        print ('Start loading data...')
        assert args.model_name.startswith('t5')

        ckpt_name = get_checkpoint_name(args.pretrained_path)
        pretrained_path = args.pretrained_path + '/' + ckpt_name

        from transformers import T5Tokenizer
        print ('Loading Pretrained Tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(pretrained_path)

        if args.add_prefix == 'True':
            add_prefix = True
        elif args.add_prefix == 'False':
            add_prefix = False
        else:
            raise Exception('Wrong Prefix Mode!!!')

        if args.eva_mode in ['test', 'dev', 'train', 'all']:
            eva_mode = args.eva_mode
        else:
            raise Exception('Wrong Eva Mode!!!')

        if args.add_special_decoder_token == 'True':
            add_special_decoder_token = True
        elif args.add_special_decoder_token == 'False':
            add_special_decoder_token = False
        else:
            raise Exception('Wrong Add Special Token Mode!!!')

        from dataclass import DSTMultiWozData
        data = DSTMultiWozData(args.model_name, tokenizer, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
                            data_mode='train', train_data_ratio=0.005)

        print ('Start loading model...')
        assert args.model_name.startswith('t5')

        from modelling.T5Model import T5Gen_Model
        model = T5Gen_Model(pretrained_path, data.tokenizer, data.special_token_list, dropout=0.0, 
            add_special_decoder_token=add_special_decoder_token, is_training=True)

        if cuda_available:
            if multi_gpu_training:
                model = nn.DataParallel(model) # multi-gpu training
            else:
                pass
            model = model.to(device)
        else:
            pass
        model.eval()
        print ('Model loaded')

        # --- start inference --- #
        with torch.no_grad():
            dev_batch_list = \
            data.build_all_evaluation_batch_list(eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode=eva_mode)
            dev_batch_num_per_epoch = len(dev_batch_list)

            p = progressbar.ProgressBar(dev_batch_num_per_epoch)
            print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
            p.start()
            all_dev_result = []
            for p_dev_idx in range(dev_batch_num_per_epoch):
                p.update(p_dev_idx)
                one_inference_batch = dev_batch_list[p_dev_idx]
                dev_batch_parse_dict = batch_generate(model, one_inference_batch, data)
                for item in dev_batch_parse_dict:
                    all_dev_result.append(item)
            p.finish()

    # --- start evaluation and save results --- #
    from compute_joint_acc import compute_jacc
    all_dev_result = zip_result(all_dev_result)
    dev_score, over_gen_rate, de_gen_rate, error_dials = compute_jacc(data=all_dev_result)
    dev_score *= 100

    error_dialog_num = len(error_dials)
    total_dialog_num = len(all_dev_result)
    print ('Total {} dialogs, {} error dialogs, error rate {}'.format(total_dialog_num, error_dialog_num, error_dialog_num/total_dialog_num))

    if args.input_test_path:
        input_test = args.input_test_path.replace(".json", "")
        one_dev_str = '{}_joint_accuracy_{}'.format(input_test, round(dev_score,2))
    else:
        one_dev_str = '{}_joint_accuracy_{}'.format(eva_mode, round(dev_score,2))

    print ('Test Accuracy is {}'.format(dev_score), 'Over-generation rate is {}'.format(over_gen_rate), 'De-generation rate is {}'.format(de_gen_rate))

    output_save_path = args.output_save_path + '/' + one_dev_str + '.json'
    import os
    if os.path.exists(args.output_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(args.output_save_path, exist_ok=True)

    import json
    with open(output_save_path, 'w') as outfile:
        json.dump(all_dev_result, outfile, indent=4)
    
    output_save_path = args.output_save_path + '/' + one_dev_str + '_error_dials.json'
    import os
    if os.path.exists(args.output_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(args.output_save_path, exist_ok=True)

    import json
    with open(output_save_path, 'w') as outfile:
        json.dump(error_dials, outfile, indent=4)

    print ('Evaluation Completed!')
