import os
import random
import json
import time
import numpy as np
import os
import sys
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import progressbar
import argparse
from eval import MultiWozEvaluator
from transformers import *
import openai
import re
import copy
import pprint
import logging
import time
from colorama import Fore, Back, Style

from dialogic_utils import system_prefix, user_prefix
from dialogic_utils import *

def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    selected_name = ""
    for name in file_names:
        if name.startswith('epoch'):
            if 'best' in name:
                print (name)
                return name
            selected_name = name
    print (selected_name)
    return selected_name

def create_logger(args):
    """
    print the logs to console and file
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(args.log_path):
        os.makedirs("/".join(args.log_path.split("/")[:-2]), exist_ok=True)

    # create a handler to write logs in files
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--data_version', type=str, help='The version of used multiwoz data, 2.0, 2.1, 2.3, 2.4')

    # the configuration of verifier
    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--use_db_as_input', type=str, default='False', 
        help="True or False, whether includes db result as part of the input when generating response.")
    parser.add_argument('--cascaded', type=str, default='False', 
        help="True or False, whether includes action when generating response.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')
    parser.add_argument('--gpt3_version', type=str, default='text-davinci-002', help='options: [text-davinci-002, text-davinci-001, text-curie-001, text-babbage-001, or text-ada-001]')

    # simulation configuration
    parser.add_argument('--max_aug_time', type=int, default=1, help='the size of augment data: x original data size.')
    parser.add_argument('--max_dialog_num', type=int, default=0, help='the maximum dialog using gpt-3, if 0 depends on max_aug_size')
    parser.add_argument('--max_turn_num', type=int, default=10, help='the maximum turns of each dialog.')
    parser.add_argument('--max_repeat_time', type=int, default=3, help='the maximum time of repeat generation of GPT-3.')

    # verifier
    parser.add_argument('--verify_bs', type=str, default='True', help='only simulate user.')
    parser.add_argument('--verify_da', type=str, default='True', help='simulate both user and system.')
    parser.add_argument('--n_user', type=int, default=1, help='how many user utterances for each bs, 1 means no rewrite.')
    parser.add_argument('--n_system', type=int, default=1, help='how many system response for each da, 1 means no rewrite.')

    # data
    parser.add_argument('--augment_dialog_path', type=str, default=None, help='the path that stores the error test cases in dst.')
    parser.add_argument("--output_save_path", type=str, help="directory to save the model output.")
    parser.add_argument('--log_path', type=str, help='the path that stores the log information.')
    
    # debug
    parser.add_argument('--debug', default='True', type=str, help='Whether to print in the process.')   
    parser.add_argument('--save', default='True', type=str, help='Whether to save the interaction results.')   

    return parser.parse_args()

def convert_db_to_pointer(text):
    pointer_id = re.findall("\d+", text)
    pointer = [0] * 7
    if pointer_id: 
        pointer_id = int(pointer_id[0])
        pointer[pointer_id] = 1
    return pointer

def get_turn_domain(text, q):

    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text
    else:
        raise Exception("Wrong text when extracting turn domain!")
    
    from ontology import all_domains
    for text in texts:
        domains = re.findall(r"\[.+?\]", text)
        for domain in domains:
            if domain not in q and domain[1:-1] in all_domains:
                q.append(domain)
                turn_domain = q[-1:]
                return turn_domain
    return q[-1:]

def save_dialogs(args, all_dialogs, one_dev_str):

    output_save_path = os.path.join(args.output_save_path, one_dev_str + f'_simulation_result.json')
    
    if os.path.exists(args.output_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(args.output_save_path, exist_ok=True)

    # rearrange the order of keys of the dialogue data
    for dialog in all_dialogs:
        dialog_turns = dialog["turns"]
        new_dialog_turns = [rearrange_dict(turn) for turn in dialog_turns]
        dialog["turns"] = new_dialog_turns

    # save the dialogs
    with open(output_save_path, 'w') as outfile:
        json.dump(all_dialogs, outfile, indent=4)
    print(f"Saving dialogues, current num: {len(all_dialogs)}!")

def load_dialogs(args, one_dev_str):

    output_save_path = os.path.join(args.output_save_path, one_dev_str + f'_simulation_result.json')

    if not os.path.exists(output_save_path):
        print(f"No dialogues so far in {output_save_path}!")
        return []
    
    with open(output_save_path, 'r') as inputfile:
        all_dialogs = json.load(inputfile)
    print(f"Loading dialogues, current num: {len(all_dialogs)}!")
    return all_dialogs


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
    logger = create_logger(args)

    device = torch.device('cuda')

    print ('Start loading data...')
    from dataclass import MultiWozData

    if args.data_version == "2.0":
        from config import Config
    elif args.data_version == "2.1":
        from config21 import Config
    elif args.data_version == "2.3":
        from config23 import Config
    elif args.data_version == "2.4":
        from config24 import Config
    else:
        raise Exception("Wrong MultiWOZ version!")

    cfg = Config(args.data_path_prefix)
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

    if args.cascaded == 'True':
        cascaded = True
    elif args.cascaded == 'False':
        cascaded = False
    else:
        raise Exception('Wrong Use Cascaded Mode!!!')

    if args.verify_bs == 'True':
        verify_bs = True
    elif args.verify_bs == 'False':
        verify_bs = False
    else:
        raise Exception('Wrong verify_bs Mode!!!')

    if args.verify_da == 'True':
        verify_da = True
    elif args.verify_da == 'False':
        verify_da = False
    else:
        raise Exception('Wrong verify_da Mode!!!')

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

    if args.debug == 'True':
        debug = True
    elif args.debug == 'False':
        debug = False
    else:
        raise Exception('Wrong debug Mode!!!')

    if args.pretrained_path != 'None':
        ckpt_name = get_checkpoint_name(args.pretrained_path)
        pretrained_path = args.pretrained_path + '/' + ckpt_name

    if args.pretrained_path != 'None':
        print ('Loading Pretrained Tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='interact', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio)

    print ('Data loaded')
    evaluator = MultiWozEvaluator(data.reader, cfg)

    print ('Start loading model...')
    if verify_bs or verify_da:
        assert args.model_name.startswith('t5')
        from modelling.T5Model import T5Gen_Model
        if args.pretrained_path != 'None':
            model = T5Gen_Model(pretrained_path, data.tokenizer, data.special_token_list, dropout=0.0, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)
        else:
            model = T5Gen_Model(args.model_name, data.tokenizer, data.special_token_list, dropout=0.0, 
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

    from e2e_inference_utlis import e2e_batch_interactive_generate
    with torch.no_grad():
        input_contain_db=use_db_as_input

        """
        interact with system (verifier) using GPT-3

        """
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = OPENAI_API_KEY
        
        # load the augment dialogs with prompts
        assert args.augment_dialog_path is not None
        f = open(args.augment_dialog_path, "r")
        augment_dialogs = json.load(f) # dict, dict[dial_id] is dialog_dict, dialog_dict[turn_id] is a turn_dict
        augment_dialogs = random_dict(augment_dialogs) # shuffle the dict
        f.close()
        
        # save file name
        one_dev_str = args.augment_dialog_path.split("/")[-1].split(".json")[0].strip()

        # record all the interaction dialogs
        all_dialogs = load_dialogs(args, one_dev_str)

        # num_augment_dialogs = 0
        num_augment_dialogs = len(all_dialogs)

        # record the dst generation performance of gpt-3
        total_turn_num, over_gen_turn_num, de_gen_turn_num = 0, 0, 0

        def not_aug_time_satisfied(all_dialogs, augment_dialogs, max_aug_time):
            # initialize aug time for each dialog
            dialog_aug_times = {}
            # only keep the valid dialogs
            for dial_id, dialog_turn_with_info in augment_dialogs.items():
                dialog_aug_times[dial_id] = 0

            # update aug time for each dialog
            for dialog in all_dialogs:
                dial_id = dialog['dial_id']
                if dial_id in dialog_aug_times:
                    dialog_aug_times[dial_id] += 1
                else:
                    pass
                    # raise Exception("Out of training set dialogs!")
            
            # check whether each dialog has been interacted
            not_satisfied_dials = []
            for dial_id, dial_aug_time in dialog_aug_times.items():
                if dial_aug_time < max_aug_time:
                    not_satisfied_dials.append(dial_id)
                elif dial_aug_time > max_aug_time:
                    raise Exception(f"Exceed max aug time: {dial_id}")
                else:
                    pass
            return not_satisfied_dials

        while not_aug_time_satisfied(all_dialogs, augment_dialogs, args.max_aug_time):
        
            for dial_id, dialog_turn_with_info in augment_dialogs.items():
            
                if dial_id not in not_aug_time_satisfied(all_dialogs, augment_dialogs, args.max_aug_time):
                    print(f"Dialogue {dial_id} already exist!")
                    continue

                real_goal = {}
                dialog_info = {}
                dialog_turns = []
                turn_id = 0

                print()
                print(f"Start interaction based on {dial_id}!")

                prompt = dialog_turn_with_info['prompt']
                logger.info(dial_id)
                logger.info(prompt)

                augment_goal = dialog_turn_with_info['augment_goal']
                augment_domains = list(augment_goal.keys()) # include [general] domain

                user = ""
                context = ""
                last_da = ""
                history = [] # [(user, response)]
                domain_queue = []

                while turn_id < args.max_turn_num:
                    total_turn_num += 1

                    error_turn = False
                    end_of_dialog = False
                    turn_info = {}
                    turn_info["dial_id"] = dial_id
                    turn_info["turn_num"] = turn_id
                    turn_goal = {}
                    turn_domain = []

                    user = ""
                    repeat_time = 0
                    while not user and repeat_time < args.max_repeat_time:
                        repeat_time += 1
                        if "[offerbook]" in last_da: # ensure booking
                            _user_prefix = user_prefix(bs_reform="[general]", user="yes, ")
                        else:
                            _user_prefix = user_prefix()
                        user_with_bs = openai.Completion.create(
                                    engine=args.gpt3_version,
                                    prompt=prompt + "\n" + _user_prefix,
                                    temperature=0.7,
                                    max_tokens=64,
                                    n=1,
                                    top_p=1,
                                    frequency_penalty=1.0,
                                    presence_penalty=0,
                                    stop=["Assistant"]
                                    )["choices"][0]["text"].lower().replace("\n", "").replace("you:", "").replace("*", "").strip()
                        user_with_bs = _user_prefix + user_with_bs # You require([domain] slot_name is slot_value): user utterance

                        # extract user's utterance
                        if "):" in user_with_bs and len(user_with_bs.split("):")) == 2: 
                            user = user_with_bs.split("):")[1].strip()
                    
                    not_mentioned_domain = ""
                    for d in augment_domains:
                            if d != '[general]' and d not in real_goal:
                                not_mentioned_domain = d
                                break
                    
                    # if '[general]' in user_with_bs:
                    #     # if gpt-3 tries to end the conversation before mentioning all the domain, add a start sequence
                    #     not_mentioned_domain = ""
                    #     for d in augment_domains:
                    #         if d != '[general]' and d not in real_goal:
                    #             not_mentioned_domain = d
                    #             break

                        # if there is domain that hasn't been mentioned, regenerate the user utterance requiring the not mentioned domain
                        # if not_mentioned_domain:
                        #     pass
                        # else:
                        #     end_of_dialog = True
                        
                    if debug: print(Fore.MAGENTA + f"Original GPT-3 generation of User: {user_with_bs}" + Style.RESET_ALL)

                    # extract gpt3_bs_reform and verifier_bs_reform
                    if "require(" in user_with_bs:
                        gpt3_bspn_reform = user_with_bs.split("require(")[1].split("):")[0].strip()
                        gpt3_turn_goal = paser_bs_reform_to_dict(gpt3_bspn_reform)
                        # if debug: print(Fore.GREEN + f"GPT-3 predicted belief text: {gpt3_turn_goal}" + Style.RESET_ALL)
                    else:
                        gpt3_bspn_reform = ""
                        gpt3_turn_goal = {}
                    # turn_info["bspn_gpt3_current_turn"] = paser_dict_to_bs(gpt3_turn_goal)

                    """
                    Start to interact with TOD !!!
                    """
                    if verify_bs or verify_da:
                        # construct context_ids
                        # user = '<sos_u> {} <eos_u>'.format(user)
                        context = context + ' ' + '<sos_u> {} <eos_u>'.format(user)
                        context_ids = data.tokenizer.convert_tokens_to_ids(data.tokenizer.tokenize(context))

                        # construct bs input ids
                        one_bs_token_id_input = data.bs_prefix_id + [data.sos_context_token_id] + context_ids[-900:] + [data.eos_context_token_id]
                        batch_bs_token_id_input = [one_bs_token_id_input] # change to a batch with batch_size=1, to use batch_generate()
                        # generate bs using debugged tod
                        batch_generated_bs = e2e_batch_interactive_generate(model, 'bs', batch_bs_token_id_input, data)
                        one_bs_text = batch_generated_bs[0]
                        gen_goal = paser_bs_to_dict(one_bs_text)

                        if debug: print(Fore.BLUE + f"Verifier predicted belief state: {one_bs_text}" + Style.RESET_ALL)
                        # print(Fore.BLUE + f"Predicted belief text: {gen_goal}" + Style.RESET_ALL)
                        # record turn info
                        turn_info["bspn_verifier"] = one_bs_text

                    """
                    determine turn_domain, priority: [general] > gpt3 > debugged tod
                    """
                    if '[general]' in gpt3_bspn_reform:
                        turn_domain = ['[general]']
                    else: # detect if there is new domain, if not, return the most recently mentioned domain
                        turn_domain = get_turn_domain(gpt3_bspn_reform, domain_queue)

                    # if debug: print(f"Predicted domain: {turn_domain}")
                    # record turn info
                    turn_info["turn_domain"] = turn_domain
                    turn_info["dspn"] = turn_domain
                    
                    """
                    Start analyzing the bs generated by GPT-3 and verifier
                    """
                    for domain in turn_domain:
                        """
                        obtain current domain_bs/turn_bs
                        """
                        if domain in real_goal:
                            domain_bs = copy.deepcopy(real_goal[domain])
                        else:
                            domain_bs = {}
                        if domain in turn_goal:
                            turn_bs = copy.deepcopy(turn_goal[domain])
                        else:
                            turn_bs = {}
                        
                        """
                        determine bs and update real_goal/turn_goal based on the multi-turn prediction of debugged TOD
                        """
                        if verify_bs:
                            if domain in gen_goal:
                                gen_domain_bs = copy.deepcopy(gen_goal[domain])
                            else:
                                gen_domain_bs = {}

                            for slot_name, slot_value in gen_domain_bs.items():
                                # check if the slot appears in user's utterance of this turn
                                mentioned_in_this_turn = False
                                if len(slot_value)==1 and slot_value.isdigit():
                                    if slot_value in user.split() or slot_value+"." in user or slot_value+"," in user or slot_value+"?" in user:
                                        mentioned_in_this_turn = True
                                    else:
                                        if slot_value in num2word:
                                            if num2word[slot_value] in user:
                                                mentioned_in_this_turn = True
                                elif slot_value != "yes" and ((len(slot_value)==1 and (slot_value in user.split() or slot_value+"." in user or slot_value+"," in user or slot_value+"?" in user)) or (len(slot_value)>1 and slot_value in user)):
                                    mentioned_in_this_turn = True
                                elif slot_value == "yes" and slot_name == 'internet' and any([_ in user for _ in ['wifi', 'internet']]) and not any([_ in user for _ in ["don't", "donot", "don 't", "dont", "doesn't"]]):
                                    mentioned_in_this_turn = True
                                elif slot_value == "yes" and slot_name == 'parking' and 'parking' in user and not any([_ in user for _ in ["don't", "donot", "don 't", "dont", "doesn't"]]):
                                    mentioned_in_this_turn = True
                                elif any([_ in user for _ in ["same"]]): # deal with in the same group, in the same place, don't care situation
                                    appear_time = 0 
                                    for d, b in gen_goal.items():
                                        for s in b.values():
                                            if s == slot_value: 
                                                appear_time += 1
                                    if appear_time >= 2: # should appear at least 2 times
                                        mentioned_in_this_turn = True
                                elif slot_value in ['dont care', "don't care", "do nt care", "doesn't care", "dontcare"] or "care" in slot_value:
                                    mentioned_in_this_turn = True
                                else:
                                    for norm_slot_value, typo in GENERAL_TYPO.items():
                                        if slot_value == typo:
                                            if ((len(norm_slot_value)==1 and (norm_slot_value in user.split() or norm_slot_value+"." in user or norm_slot_value+"," in user or norm_slot_value+"?" in user)) or (len(norm_slot_value)>1 and norm_slot_value in user)):
                                                mentioned_in_this_turn = True
                                                break
                                        if slot_value == norm_slot_value:
                                            if ((len(typo)==1 and (typo in user.split() or typo+"." in user or typo+"," in user or typo+"?" in user)) or (len(typo)>1 and typo in user)):
                                                mentioned_in_this_turn = True
                                                break

                                # check if this slot was mentioned in last turn
                                mentioned_in_last_turn = False
                                if history: 
                                    last_user, last_response = history[-1]
                                    last_user += " " + last_response
                                    if len(slot_value)==1 and slot_value.isdigit():
                                        if slot_value in last_user.split() or slot_value+"." in last_user or slot_value+"," in last_user or slot_value+"?" in last_user:
                                            mentioned_in_last_turn = True
                                        else:
                                            if slot_value in num2word:
                                                if num2word[slot_value] in last_user:
                                                    mentioned_in_last_turn = True
                                    elif slot_value != "yes" and ((len(slot_value)==1 and (slot_value in last_user.split() or slot_value+"." in last_user or slot_value+"," in last_user or slot_value+"?" in last_user)) or (len(slot_value)>1 and slot_value in last_user)):
                                        mentioned_in_last_turn = True
                                    elif slot_value == "yes" and slot_name == 'internet' and any([_ in last_user for _ in ['wifi', 'internet']]) and not any([_ in last_user for _ in ["don't", "donot", "don 't", "dont", "doesn't"]]):
                                        mentioned_in_last_turn = True
                                    elif slot_value == "yes" and slot_name == 'parking' and 'parking' in last_user and not any([_ in last_user for _ in ["don't", "donot", "don 't", "dont", "doesn't"]]):
                                        mentioned_in_last_turn = True
                                    elif any([_ in last_user for _ in ["same"]]): # deal with in the same group, in the same place, don't care situation
                                        appear_time = 0 
                                        for d, b in gen_goal.items():
                                            for s in b.values():
                                                if s == slot_value: 
                                                    appear_time += 1
                                        if appear_time >= 2: # should appear at least 2 times
                                            mentioned_in_last_turn = True
                                    else:
                                        for norm_slot_value, typo in GENERAL_TYPO.items():
                                            if slot_value == typo:
                                                if ((len(norm_slot_value)==1 and (norm_slot_value in last_user.split() or norm_slot_value+"." in last_user or norm_slot_value+"," in last_user or norm_slot_value+"?" in last_user)) or (len(norm_slot_value)>1 and norm_slot_value in last_user)):
                                                    mentioned_in_last_turn = True
                                                    break
                                            if slot_value == norm_slot_value:
                                                if ((len(typo)==1 and (typo in last_user.split() or typo+"." in last_user or typo+"," in last_user or typo+"?" in last_user)) or (len(typo)>1 and typo in last_user)):
                                                    mentioned_in_last_turn = True
                                                    break

                                if mentioned_in_this_turn: # can update in domain_bs and turn_bs
                                    # check if this slot is in this domain
                                    from ontology import informable_slots
                                    domain_pure_text = domain.replace("[","").replace("]", "").strip() # [taxi] -> taxi
                                    if domain_pure_text in informable_slots:
                                        if slot_name in informable_slots[domain_pure_text]:
                                            if slot_value and not re.findall(r"\[.+?\]", slot_value):
                                                domain_bs[slot_name] = slot_value
                                                turn_bs[slot_name] = slot_value

                                if mentioned_in_last_turn: # can only update in domain_bs, not turn_bs!
                                    # check if this slot is in this domain
                                    from ontology import informable_slots
                                    domain_pure_text = domain.replace("[","").replace("]", "").strip() # [taxi] -> taxi
                                    if domain_pure_text in informable_slots:
                                        if slot_name in informable_slots[domain_pure_text]:
                                            if slot_value and not re.findall(r"\[.+?\]", slot_value): 
                                                domain_bs[slot_name] = slot_value

                        """
                        update real_goal/turn_goal based on the single turn prediction of GPT-3 and verifier
                        """
                        predicted_turn_goals = []
                        predicted_turn_goals.append(gpt3_turn_goal)

                        # start analyzing domain_bs_text
                        for predicted_turn_goal in predicted_turn_goals:         
                            if predicted_turn_goal and domain in predicted_turn_goal:
                                predicted_domain_bs = predicted_turn_goal[domain]
                                for slot_name, slot_value in predicted_domain_bs.items():
                                    mentioned_in_this_turn = False
                                    # check if the slot appears in user's utterance of this turn
                                    user_words = [word.strip() for word in user.split()]
                                    if len(slot_value)==1 and slot_value.isdigit():
                                        if slot_value in user_words or slot_value+"." in user or slot_value+"," in user or slot_value+"?" in user:
                                            mentioned_in_this_turn = True
                                        else:
                                            if slot_value in num2word:
                                                if num2word[slot_value] in user:
                                                    mentioned_in_this_turn = True
                                    elif slot_value != "yes" and ((len(slot_value)==1 and (slot_value in user_words or slot_value+"." in user or slot_value+"," in user or slot_value+"?" in user)) or (len(slot_value)>1 and slot_value in user)):
                                        mentioned_in_this_turn = True
                                    elif slot_value == "yes" and slot_name == 'internet' and any([_ in user for _ in ['wifi', 'internet']]) and not any([_ in user for _ in ["don't", "donot", "don 't", "dont", "doesn't"]]):
                                        mentioned_in_this_turn = True
                                    elif slot_value == "yes" and slot_name == 'parking' and 'parking' in user and not any([_ in user for _ in ["don't", "donot", "don 't", "dont", "doesn't"]]):
                                        mentioned_in_this_turn = True
                                    elif slot_value in ['dont care', "don't care", "do nt care", "doesn't care", "dontcare"] or any([_ in slot_value for _ in ["care"]]):
                                        mentioned_in_this_turn = True
                                    else:
                                        for norm_slot_value, typo in GENERAL_TYPO.items():
                                            if slot_value == typo:
                                                if ((len(norm_slot_value)==1 and (norm_slot_value in user.split() or norm_slot_value+"." in user or norm_slot_value+"," in user or norm_slot_value+"?" in user)) or (len(norm_slot_value)>1 and norm_slot_value in user)):
                                                    mentioned_in_this_turn = True
                                                    break
                                            if slot_value == norm_slot_value:
                                                if ((len(typo)==1 and (typo in user.split() or typo+"." in user or typo+"," in user or typo+"?" in user)) or (len(typo)>1 and typo in user)):
                                                    mentioned_in_this_turn = True
                                                    break

                                    if mentioned_in_this_turn:
                                        # check the slots valid before updating the tracked bs
                                        from ontology import informable_slots
                                        domain_pure_text = domain.replace("[","").replace("]", "").strip() # [taxi] -> taxi
                                        if domain_pure_text in informable_slots:
                                            if slot_name in informable_slots[domain_pure_text]:
                                                if slot_value and not re.findall(r"\[.+?\]", slot_value):
                                                    domain_bs[slot_name] = slot_value
                                                    turn_bs[slot_name] = slot_value
                                    else:
                                        print(f"Slot {slot_name}: {slot_value} not in user utterance: {user}")
                        
                        """
                        update real_goal and turn_goal, based on the prediction of TOD, gpt-3 and verifier
                        """
                        real_goal[domain] = domain_bs
                        turn_goal[domain] = turn_bs

                    """
                    evaluate the difference between gpt-3 generated goal with real goal
                    """
                    gpt3_turn_goal_list = paser_dict_to_list(gpt3_turn_goal)
                    turn_goal_list = paser_dict_to_list(turn_goal)
                    for i in turn_goal_list:
                        if i not in gpt3_turn_goal_list:
                            de_gen_turn_num += 1
                            break
                    for i in gpt3_turn_goal_list:
                        if i not in turn_goal_list:
                            over_gen_turn_num += 1
                            break

                    """
                    reconstruct the generated user_with_bs
                    """
                    bs_text = paser_dict_to_bs(reverse_dict(real_goal))
                    reform_bs_text = paser_dict_to_bs_reform(reverse_dict(real_goal))
                    bsdx_text = paser_dict_to_bsdx(reverse_dict(real_goal))
                    reform_bsdx_text = paser_dict_to_bsdx_reform(reverse_dict(real_goal))

                    # correct belief state                    
                    if debug: print(Fore.RED + f"Corrected belief state: {bs_text}" + Style.RESET_ALL)
                    # correct belief state
                    one_bs_text = bs_text
                    # record turn info
                    turn_info["bspn"] = bs_text
                    turn_info["bsdx"] = bsdx_text
                    turn_info["bspn_reform"] = reform_bs_text
                    turn_info["bsdx_reform"] = reform_bsdx_text

                    # bs for this turn, for gpt-3
                    turn_bs_text = paser_dict_to_bs_reform(turn_goal, ignore_none_bs=False)
                    
                    # record turn info
                    turn_info['user'] = user
                    # ignore the delex process
                    turn_info['usdx'] = user

                    """
                    The format in the prompt, should be consistent with the generate_prompt function
                    Example: 
                    You require([taxi] destination is pizza hut fen ditton , departure is saint john 's college): i would like a taxi from saint john 's college to pizza hut fen ditton . 
                    """ 
                    # update prompt
                    user_text = user_prefix(turn_bs_text, user) # You require(turn_bs_text): user
                    if debug: print(Fore.MAGENTA + f"Corrected GPT-3 generation of User: {user_text}" + Style.RESET_ALL)
                    prompt += "\n" + user_text
                    logger.info("\n" + user_text)
                    
                    """
                    Continue the generation of dialog action and response, given the correct belief state !!!!!!
                    """
                    one_queried_db_result = data.reader.bspan_to_DBpointer(one_bs_text, turn_domain)
                    db_pointer = convert_db_to_pointer(one_queried_db_result)
                    turn_info["db"] = one_queried_db_result
                    turn_info["pointer"] = db_pointer
                    
                    # whether we need to query the db base
                    if input_contain_db:
                        # record turn info
                        if debug: print(Fore.LIGHTGREEN_EX + f"DB search result: {one_queried_db_result}" + Style.RESET_ALL)
                        one_db_text = '<sos_db> ' + one_queried_db_result + ' <eos_db>'
                        one_db_token_id_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_db_text))
                    else:
                        one_db_token_id_input = []
                        # record turn info

                    one_da_text = ""

                    if verify_da:
                        # then we generate the dialogue action
                        one_da_token_id_input = data.da_prefix_id + [data.sos_context_token_id] + context_ids[-900:] + [data.eos_context_token_id] + one_db_token_id_input
                        batch_da_token_id_input = [one_da_token_id_input] # change to a batch with batch_size=1, to use batch_generate()
                        # generate da
                        batch_generated_da = e2e_batch_interactive_generate(model, 'da', batch_da_token_id_input, data)
                        one_da_text = batch_generated_da[0]

                        one_da_token_id_output = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_da_text))
                        if debug: print(Fore.YELLOW + f"Verifier generated Action: {one_da_text}" + Style.RESET_ALL)

                        if one_da_text:

                            # finally we generate the response
                            if not cascaded: # only needs db
                                one_nlg_token_id_input = data.nlg_prefix_id + [data.sos_context_token_id] + context_ids[-900:] + [data.eos_context_token_id] + one_db_token_id_input
                            else: # needs db and da
                                one_nlg_token_id_input = data.nlg_prefix_id + [data.sos_context_token_id] + context_ids[-900:] + [data.eos_context_token_id] + one_db_token_id_input + one_da_token_id_output
                            batch_nlg_token_id_input = [one_nlg_token_id_input] # change to a batch with batch_size=1, to use batch_generate()
                            # generate nlg
                            batch_generated_nlg = e2e_batch_interactive_generate(model, 'nlg', batch_nlg_token_id_input, data)
                            one_nlg_text = batch_generated_nlg[0]
                            if debug: print(Fore.CYAN + f"Verifier generated Response: {one_nlg_text}" + Style.RESET_ALL)

                            # record turn info
                            turn_info["aspn"] = one_da_text
                            turn_info["aspn_verifier"] = one_da_text
                            turn_info["aspn_reform"] = one_da_text

                            # using gpt-3 generation
                            system_based_on_da = openai.Completion.create(
                                        # engine="text-davinci-002",
                                        engine=args.gpt3_version,
                                        prompt=prompt + "\n" + system_prefix(one_da_text),
                                        temperature=0.8,
                                        max_tokens=64,
                                        n=1,
                                        top_p=1,
                                        frequency_penalty=1.0,
                                        presence_penalty=0,
                                        stop=["You require"]
                                        )["choices"][0]["text"].lower().replace("\n", "").replace("you:", "").replace("*", "").strip()

                            # make sure that the entity is proposed.
                            if '_name' in one_nlg_text and '_name' not in system_based_on_da:
                                system_based_on_da = one_nlg_text

                            # record turn info
                            turn_info["resp"] = system_based_on_da
                            turn_info["resp_verifier"] = one_nlg_text
                            prompt += "\n" + system_prefix(one_da_text, system_based_on_da)
                            logger.info("\n" + system_prefix(one_da_text, system_based_on_da))
                            if debug: print(Fore.LIGHTMAGENTA_EX + f"Corrected GPT-3 generation of System: {system_prefix(one_da_text, system_based_on_da)}" + Style.RESET_ALL)

                            # determine if it is the end
                            if ("[bye]" in gpt3_aspn_reform or "[welcome]" in gpt3_aspn_reform) and not not_mentioned_domain: 
                                end_of_dialog = True
                    
                    if not one_da_text:
                        # using gpt-3 generation
                        system_based_on_da = ""
                        repeat_time = 0
                        while not system_based_on_da and repeat_time < args.max_repeat_time:
                            repeat_time += 1
                            system_with_da = openai.Completion.create(
                                    # engine="text-davinci-002",
                                    engine=args.gpt3_version,
                                    prompt=prompt + "\n" + system_prefix(),
                                    temperature=0.7,
                                    max_tokens=64,
                                    n=1,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0,
                                    stop=["You require"]
                                    )["choices"][0]["text"].lower().replace("\n", "").replace("you:", "").replace("*", "").strip()
                            system_with_da = system_prefix() + system_with_da

                            if "):" in system_with_da and len(system_with_da.split("):")) == 2: 
                                system_based_on_da  = system_with_da.split("):")[1].strip()
                        
                        if debug: print(Fore.LIGHTMAGENTA_EX + f"Original GPT-3 Generated System: {system_with_da}" + Style.RESET_ALL)

                        # extract gpt3_da_reform 
                        if "Assistant(" in system_with_da:
                            gpt3_aspn_reform = system_with_da.split("Assistant(")[1].split("):")[0].strip()
                            if debug: print(Fore.GREEN + f"GPT-3 predicted dialog action: {gpt3_aspn_reform}" + Style.RESET_ALL)
                        else:
                            gpt3_aspn_reform = ""

                        # record turn info
                        turn_info["aspn"] = gpt3_aspn_reform
                        # turn_info["aspn_gen"] = one_da_text
                        turn_info["aspn_reform"] = gpt3_aspn_reform

                        # record turn info
                        turn_info["resp"] = system_based_on_da
                        # turn_info["resp_gen"] = one_nlg_text
                        prompt += "\n" + system_prefix(gpt3_aspn_reform, system_based_on_da)
                        logger.info("\n" + system_prefix(gpt3_aspn_reform, system_based_on_da))
                    
                        # determine if it is the end
                        if ("[bye]" in gpt3_aspn_reform or "[welcome]" in gpt3_aspn_reform) and not not_mentioned_domain: 
                            end_of_dialog = True

                    # add response to context
                    system = '<sos_r> {} <eos_r>'.format(turn_info["resp"])
                    context = context + ' ' + system

                    # add it to history
                    history.append((turn_info["user"], turn_info["resp"]))

                    # Print generated response
                    print(Fore.WHITE)
                    print(f"--------------------------------- turn {turn_id} ---------------------------------")
                    print(f"User: {turn_info['user']}")
                    print(f"Assistant: {turn_info['resp']}")
                    print("--------------------------------------------------------------------------")
                    print(Style.RESET_ALL)

                    # rearrange the orders and record this turn's info
                    dialog_turns.append(turn_info)
                    turn_id += 1

                    # determine whether to end this dialog
                    if end_of_dialog:
                        break
                
                # record this dialog's info
                dialog_info['dial_id'] = dial_id
                dialog_info['turns'] = dialog_turns
                dialog_info['prompt'] = dialog_turn_with_info['prompt']
                dialog_info['goal'] = real_goal
                all_dialogs.append(dialog_info) 
                print(f"Dialogue {dial_id} simulation finished !!!")
                print()

                # save dialogs
                if args.save:
                    save_dialogs(args, all_dialogs, one_dev_str)

                num_augment_dialogs += 1
                if num_augment_dialogs >= args.max_dialog_num:
                    break

            if num_augment_dialogs >= args.max_dialog_num:
                break
        
        # save dialogs
        if args.save:
            save_dialogs(args, all_dialogs, one_dev_str)

        print(f"Simulate {num_augment_dialogs} dialogs, {total_turn_num} turns, over-generation turns {over_gen_turn_num}, de-generation turns {de_gen_turn_num}.")        

    

