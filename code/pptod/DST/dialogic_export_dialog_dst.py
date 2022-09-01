
from ast import arg
import enum
from pickle import NONE
import progressbar
import argparse
import logging
import time
import json
import random
import re
import copy
import os
import numpy as np


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration        
    parser.add_argument('--data_path_prefix', type=str, default='../data/multiwoz/data/multi-woz-2.3-fine-processed/', help='The path where the data stores.')
    parser.add_argument('--save_data_path_prefix', type=str, default='../../trade-dst/generated_data/dst_2.3_1.0/', help='The path where the data stores.')

    parser.add_argument('--shuffle_mode', type=str, default='unshuffle', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--delx_response', type=str, default='True', 
        help="True or False, whether we use delx system response or nodelx system response.")

    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")

    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    parser.add_argument('--aug_train_data_file', type=str, default="", help='the augmented data file path')
    parser.add_argument('--aug_data_balance', type=str, default="True", help='whether to balance the original data and the augmented data')
    
    # model configuration
    parser.add_argument('--model_name', type=str, default='t5-base', help='t5-base or t5-large or facebook/bart-base or facebook/bart-large')
    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    return parser.parse_args()


import argparse
if __name__ == '__main__':
    args = parse_config()
    
    print ('Start loading data...')
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    if args.pretrained_path != 'None':
        print ('Loading Pretrained Tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    if args.delx_response == 'True':
        delx_response = True
    elif args.delx_response == 'False':
        delx_response = False
    else:
        raise Exception('Wrong Delex Mode!!!')

    if args.aug_data_balance == 'True':
        aug_data_balance = True
    elif args.delx_response == 'False':
        aug_data_balance = False
    else:
        raise Exception('Wrong Aug Data Balance Mode!!!')

    if args.add_prefix == 'True':
        add_prefix = True
    elif args.add_prefix == 'False':
        add_prefix = False
    else:
        raise Exception('Wrong Prefix Mode!!!')
    
    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    from dataclass import DSTMultiWozData
    if args.aug_train_data_file:
        data = DSTMultiWozData(args.model_name, tokenizer, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
                          data_mode='train', delx=delx_response, train_data_ratio=args.train_data_ratio, aug_train_data_file=args.aug_train_data_file, augment_data_balance=aug_data_balance)
    else:
        data = DSTMultiWozData(args.model_name, tokenizer, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
                          data_mode='train', delx=delx_response, train_data_ratio=args.train_data_ratio)

    print("Start converting raw multiwoz data......")
    train_dials, dev_dials, test_dials, train_dial_id_list, dev_dial_id_list, test_dial_id_list = data.convert_to_raw_data()

    print("Start saving raw multiwoz data......")
    # save the dialog turn info
    if not os.path.exists(args.save_data_path_prefix):
        os.makedirs(args.save_data_path_prefix)
        
    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "train_raw_dials.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(train_dials, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "dev_raw_dials.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(dev_dials, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "test_raw_dials.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(test_dials, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "trainListFile.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(train_dial_id_list, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "valListFile.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(dev_dial_id_list, f)
    f.close()

    save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "testListFile.json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(test_dial_id_list, f)
    f.close()




