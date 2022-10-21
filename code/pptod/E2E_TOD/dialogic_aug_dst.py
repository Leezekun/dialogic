
import enum
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
import openai
import torch
from transformers import *

from colorama import Fore, Back, Style

from dialogic_utils import *

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration    
    parser.add_argument('--multiwoz_schema_path', type=str, default="../data/multiwoz/data/multi-woz-2.3-fine-processed/schema.json", help='the path that stores the schema for multiwoz dataset.')
    parser.add_argument('--possible_slot_values_path', type=str, default="../data/multiwoz/data/multi-woz-2.3-fine-processed/possible_slot_values.json", help='the path that stores the possible slot values for multiwoz dataset.')

    parser.add_argument('--data_path_prefix', type=str, default='../data/multiwoz/data', help='The path where the data stores.')
    parser.add_argument('--data_version', type=str, default='2.3', help='The version of used multiwoz data, 2.0, 2.1, 2.3, 2.4')
    
    # the data information
    parser.add_argument('--eva_mode', type=str, default='train', 
        help="test or dev, or train, or all, evaluation on test or dev dataset")
    parser.add_argument('--train_data_ratio', type=float, default=0.01, help='the ratio of training data used for training the model')

    # how to build the prompt
    parser.add_argument('--augment_time', type=int, default=1, help='the augment size compared with the original dataset')
    parser.add_argument('--n_user', type=int, default=1, help='how many user utterances for each bs.')
    parser.add_argument('--k_shot', type=int, default=2, help='the maximum number of demo dialogs.')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature in softmax, for the sampling in combine.')

    # debug
    parser.add_argument('--debug', default='True', type=str, help='Whether to print in the process.')  

    return parser.parse_args()


def softmax(x, temperature=1.0):
    x = [_/temperature for _ in x]
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def calculate_turn_similarity(single_turn_info, query_goal, metric="jaccard"):
    
    def jaccard(list1, list2):
        intersection = list(set(list1) & set(list2))
        unionset = list(set(list1).union(set(list2)))
        if unionset:
            return float(len(intersection) / len(unionset))
        else:
            return 0.0

    goal1 = query_goal
    goal_list1 = paser_bs_from_dict_to_list(goal1)
    goal_domain1 = list(goal1.keys())
    if "[general]" in goal_domain1:
        goal_domain1.remove("[general]")

    other_turn_similarities = []
    other_turns = []

    for turn_info in single_turn_info:

        goal2 = turn_info['bs']
        has_valid_bs, has_dontcare = detect_valid_bs(goal2)

        if goal1 == goal2 or not has_valid_bs:
            continue

        goal_list2 = paser_bs_from_dict_to_list(goal2)
        goal_domain2 = list(goal2.keys())
        if "[general]" in goal_domain2:
            goal_domain2.remove("[general]")

        similarity = jaccard(goal_list1, goal_list2)
        if_same_domain = float(goal_domain1 == goal_domain2) 
        if_same_bs = float(similarity == 1.0) 
        if_error = turn_info["bs_error"]

        # only select same domain, don't select totally same goal, or error turns
        similarity = similarity * if_same_domain * (1-if_same_bs) * (1-if_error)
        
        if has_dontcare: similarity *= 2.0 # increase the possibility of selecting a turn with dontcare slot value

        other_turn_similarities.append(similarity)
        other_turns.append(turn_info)

    similarity_dict = {"turns": other_turns, "turn_similarities": other_turn_similarities}

    return similarity_dict


def sample_demo_turns(single_turn_info, augment_turn_goal, args):

    not_mention_goal = copy.deepcopy(augment_turn_goal)
    
    augment_demos = []

    # find the similar real dialogs as demo
    similarities_dict = calculate_turn_similarity(single_turn_info, not_mention_goal)
    other_turns = similarities_dict['turns']
    other_turn_similarities = similarities_dict['turn_similarities']

    if np.sum(other_turn_similarities) == 0:
        return augment_demos

    other_turn_weights = softmax([sim if sim !=0 else -1e9 for sim in other_turn_similarities], args.temperature)

    # select the one with highest similarity
    attempt_time = 0
    while len(augment_demos) < args.k_shot and attempt_time < 5:
        k_shot = args.k_shot - len(augment_demos)
        another_turns = random.choices(other_turns, weights=other_turn_weights, k=k_shot)
        for another_turn in another_turns:
            if another_turn not in augment_demos:
                augment_demos.append(another_turn) 
        attempt_time += 1      

    return augment_demos


def add_dontcare(augment_demos, augment_turn_goal):
    for augment_demo in augment_demos:
        bs = augment_demo['bs']
        user = augment_demo['user']
        for domain, domain_bs in bs.items():
            for slot, slot_value in domain_bs.items():
                if slot_value == "dontcare" and not user.startswith("no"):
                    for slot in augment_turn_goal[domain]:
                        if slot not in ["people", "stay", "destination"] and f"{domain}-{slot}" not in ["[hotel]-type", "[hotel]-stay", "[hotel]-people", "[train]-destination"]:
                            print(f"{domain}-{slot} change to dontcare!")
                            augment_turn_goal[domain][slot] = 'dontcare'
                            return augment_turn_goal
    return augment_turn_goal 


def get_slot_info(norm_schema, domain, slot):
    # obtain slot description
    slot_description = ""
    slot_possible_values = []
    domain = domain.replace("[", "").replace("]", "")
    for service in norm_schema:
        if service["service_name"] == domain:
            slots = service["slots"]
            for s in slots:
                if f"{domain}-{slot}" == s["name"]:
                    if "description" in s:
                        slot_description = copy.deepcopy(s["description"])
                    if "possible_values" in s:
                        slot_possible_values = copy.deepcopy(s["possible_values"])
                    return slot_description, slot_possible_values
    return slot_description, slot_possible_values


def generate_prompt_for_dst(norm_schema, augment_turn_goal, augment_demos, prompt_startswith):

    prompt = []  

    assert len(augment_turn_goal) == 1
    domain = list(augment_turn_goal.keys())[0]
    domain_text = domain.replace("[", "").replace("]", "")

    # intro = f"You want to book a {domain}. Tell the assistant your requirements."
    # intro = f"The following is a conversation with an booking assistant. The human wants to book a {domain}, and the assistant asks for his requirements."
    # intro = f"Write sentences to express your requirements when booking a {domain_text}. Mention and only mention the requirement in the bracket."
    # intro = f"Write sentences to answer the assistant's question on your requirements when booking a {domain_text}."
    # intro = f"Translate requirements when booking a {domain_text} to natural language, mention and only mention all the feature." 
    intro = f"Answer the assistant's question on each feature you require when booking a {domain_text}. Also mention no preference on a feature when your requirement on it is \"dontcare\"."
    prompt.append(intro)

    # add slot description for mentioned slots
    prompt.append("Features:")
    mentioned_slots = list(augment_turn_goal[domain].keys())
    for demo in augment_demos:
        bs = demo['bs']
        mentioned_slots.extend(list(bs[domain].keys()))
    mentioned_slots =list(set(mentioned_slots))

    for slot in mentioned_slots:
        slot_description, slot_possible_values = get_slot_info(norm_schema, domain, slot)
        if len(slot_possible_values) > 1 and len(slot_possible_values) <= 5:
            slot_possible_values[-1] = "or " + slot_possible_values[-1]
            # slot_possible_values.append("or dontcare (any is ok)")
            slot_possible_values = ", ".join(slot_possible_values)
            prompt.append(f"{slot}: {slot_description}, {slot_possible_values};")
        else:
            prompt.append(f"{slot}: {slot_description};")

    # add examples
    prompt.append("Examples:")

    for demo in augment_demos:
        bs = demo['bs']
        user = demo['user']

        bsdx = paser_dict_to_bsdx(bs)
        bsdx = bsdx.split()[1:]
        bsdx = ", ".join(bsdx)
        prompt.append(f"Assistant: what is your requirement on {bsdx}?")

        bspn_reform = paser_dict_to_bs_reform(bs)
        prompt.append(f"You({bspn_reform}): {user}")

    # this sample
    bsdx = paser_dict_to_bsdx(augment_turn_goal)
    bsdx = bsdx.split()[1:]
    bsdx = ", ".join(bsdx)
    prompt.append(f"Assistant: what is your requirement on {bsdx}?")

    bspn_reform = paser_dict_to_bs_reform(augment_turn_goal) 
    
    if prompt_startswith:
        prompt.append(f"You({bspn_reform}): {prompt_startswith}")
    else:
        prompt.append(f"You({bspn_reform}):")
        
    prompt = "\n".join(prompt)

    return prompt


def construct_augment_dst(dialogs_with_turn_info, orig_augment_dst_turn_info, augment_time, schema, possible_slot_values, args):

    # first normalize multiwoz2.2's schema file to this format
    norm_schema = normalize_domain_slot(schema)
    
    augment_dst_turn_info = copy.deepcopy(orig_augment_dst_turn_info)
    assert isinstance(augment_dst_turn_info, dict)

    total_turn_num, total_aug_turn_num, total_aug_slot_num = 0, 0, 0
    type_aug_num = [0, 0, 0, 0]

    for dial_id, dialog_with_turn_info in dialogs_with_turn_info.items():
        
        print(f"Current dialog id: {dial_id}")
        dialog_type_aug_num = [0, 0, 0, 0]

        orig_turns = dialog_with_turn_info['orig_turns']
        info_turns = dialog_with_turn_info['info_turns']

        if dial_id in augment_dst_turn_info:
            augment_turns = copy.deepcopy(augment_dst_turn_info[dial_id])
        else:
            augment_turns = {}

        """
        not based on message, use bs_reform as prompt
        type: [substitute, drop, combine, random]
        """
        for turn_id in orig_turns:
            
            orig_turn = orig_turns[turn_id]
            info_turn = info_turns[turn_id]

            if turn_id in augment_turns:
                augment_turn = copy.deepcopy(augment_turns[turn_id])
                augment_turn_list = copy.deepcopy(augment_turn["augment_turns"])
            else:
                augment_turn = copy.deepcopy(orig_turn)
                augment_turn_list = [] # a list a augment turns

            # obtain this turn's information
            orig_turn_goal = info_turn['bs']
            orig_turn_user = info_turn['user']
            # obtain last turn's information
            if int(turn_id) > 0:
                last_turn = orig_turns[str(int(turn_id)-1)]
                last_turn_bs_reform = last_turn['bspn_reform']
                last_turn_aspn = last_turn['aspn']
                last_turn_resp = last_turn['resp']
                orig_goal = paser_bs_reform_to_dict(last_turn_bs_reform)
                system_act = paser_aspn_to_dict(last_turn_aspn)
            else:
                last_turn_bs_reform = ""
                last_turn_aspn = ""
                last_turn_resp = ""
                orig_goal = {}
                system_act = {}
            
            # check mentioned_slots, not_mentioned_slots, and not_mentioned_domains
            mentioned_slots = {}
            if orig_goal:
                for domain, domain_slot in orig_goal.items():
                    if domain_slot:
                        mentioned_slots[domain] = list(domain_slot.keys()) # list
            mentioned_domains = []

            not_mentioned_slots = {}
            for domain in informable_slots:
                all_domain_slots = informable_slots[domain]
                domain = f"[{domain}]"
                if domain in mentioned_slots:
                    mentioned_domain_slots = mentioned_slots[domain]
                else:
                    mentioned_domain_slots = []
                not_mentioned_domain_slots = []
                for slot in all_domain_slots:
                    if slot not in mentioned_domain_slots:
                        not_mentioned_domain_slots.append(slot)
                not_mentioned_slots[domain] = not_mentioned_domain_slots

            # check request_slots
            request_slots = {}
            if system_act:
                for domain, domain_act in system_act.items():
                    if "[request]" in domain_act:
                        domain_request_slots = []
                        for slot in domain_act["[request]"]:
                            if slot == "price":
                                domain_request_slots.append("pricerange")
                            else:
                                domain_request_slots.append(slot)
                        request_slots[domain] = domain_request_slots
                        break
            assert len(request_slots) <= 1 # only 0 or 1 domain
            
            """
            start generate augment_goal
            """
            for i in range(augment_time): 
        
                augment_turn_goal = {}
                augment_demos = []
                prompt_startswith = ""
                
                type = 0

                # # augment the turns when dialog act at last turn is [request]
                if ("[request]" in last_turn_aspn and request_slots):
                    type = 2
                    # generate augment_turn_goal
                    for domain, domain_request_slots in request_slots.items():
                        # Must include: select slots from request slots, at least 1
                        random.shuffle(domain_request_slots)
                        selected_slot_num = random.randint(1, len(domain_request_slots))
                        selected_slots = domain_request_slots[:selected_slot_num]
                        # ADD: select slots from not mentioned slots, at least 0, at most 2
                        domain_not_mentioned_slots = not_mentioned_slots[domain]
                        random.shuffle(domain_not_mentioned_slots)
                        selected_slot_num = random.randint(0, min(2, len(domain_not_mentioned_slots)))
                        selected_slots += domain_not_mentioned_slots[:selected_slot_num]
                        # UPDATE: select one slot from mentioned slots, prob=0.2
                        # if domain in mentioned_slots:
                        #     domain_mentioned_slots = mentioned_slots[domain]
                        #     if random.random() < 0.2 and domain_mentioned_slots:
                        #         selected_slots += [random.choice(domain_mentioned_slots)]
                        # remove repeated slots
                        selected_slots = list(set(selected_slots))

                        # construct augment_turn_goal
                        augment_domain_bs = {}
                        for slot in selected_slots:
                            augment_domain_bs[slot] = ""
                        
                        # substitute domain value
                        augment_domain = domain.split("[")[1].split("]")[0].strip() # [hotel] -> hotel
                        augment_domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs)
                        augment_turn_goal[domain] = augment_domain_bs
                        total_aug_slot_num += slot_num

                        augment_turn_goal[domain] = augment_domain_bs

                # this is a new start, either in the begining of dialogue, or the end of a domain in a dialogue
                elif "[reqmore]" in last_turn_aspn or int(turn_id) == 0:
                    type = 3
                    not_mentioned_domains = []
                    for domain in all_domain:
                        if domain not in mentioned_domains and domain not in ["[police]", "[hospital]"]:
                            not_mentioned_domains.append(domain)
                    selected_domain = random.choice(not_mentioned_domains)
                    possible_slots = not_mentioned_slots[selected_domain]
                    random.shuffle(possible_slots)
                    selected_slot_num = random.randint(min(1, len(possible_slots)), min(4, len(possible_slots))) # at least 1, at most 4
                    selected_slots = possible_slots[:selected_slot_num]

                    augment_domain_bs = {}
                    for slot in selected_slots:
                        augment_domain_bs[slot] = ""

                    augment_domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, selected_domain, augment_domain_bs)
                    total_aug_slot_num += slot_num
                    augment_turn_goal[selected_domain] = augment_domain_bs   

                    if  "[reqmore]" in last_turn_aspn:
                        prompt_startswith = "i also need a "
                    elif int(turn_id) == 0:
                        prompt_startswith = "i need a "

                # no last turn's guidance, can only substitute value and add not mentioned slots for the turns with valid bs
                elif detect_valid_bs(orig_turn_goal)[0]:
                    type = 4
                    # generate augment_turn_goal
                    for domain, domain_bs in orig_turn_goal.items():
                        if domain_bs:
                            # DROP: drop some slots from the orig_goal, only keep part of the slots, at least 1
                            turn_mentioned_slots = list(domain_bs.keys())
                            if turn_mentioned_slots:
                                random.shuffle(turn_mentioned_slots)
                                selected_slot_num = random.randint(1, len(turn_mentioned_slots))
                                selected_slots = turn_mentioned_slots[:selected_slot_num]
                            # ADD: select slots from not mentioned slots, at least 0, at most 2
                            domain_not_mentioned_slots = []
                            for _ in not_mentioned_slots[domain]:
                                if _ not in domain_bs:
                                    domain_not_mentioned_slots.append(_) 
                            if domain_not_mentioned_slots:
                                random.shuffle(domain_not_mentioned_slots)
                                selected_slot_num = random.randint(1, min(2, len(domain_not_mentioned_slots)))
                                selected_slots += domain_not_mentioned_slots[:selected_slot_num]
                            # UPDATE: select one slot from mentioned slots, prob=0.2
                            # if domain in mentioned_slots:
                            #     domain_mentioned_slots = mentioned_slots[domain]
                            #     if random.random() < 0.2 and domain_mentioned_slots:
                            #         selected_slots += [random.choice(domain_mentioned_slots)]
                            # remove repeated slots
                            selected_slots = list(set(selected_slots))

                            # construct augment_turn_goal, add new added slots
                            augment_domain_bs = {}
                            for slot in selected_slots:
                                augment_domain_bs[slot] = ""
                            
                            # substitute domain value
                            augment_domain = domain.split("[")[1].split("]")[0].strip() # [hotel] -> hotel
                            augment_domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs)
                            augment_turn_goal[domain] = augment_domain_bs
                            total_aug_slot_num += slot_num

                            augment_turn_goal[domain] = augment_domain_bs

                            break # only mention one domain in each turn

                        # add this turn as one of the demos
                        augment_demos.append(info_turn)
                
                # given augment_turn_goal and augment_demos, using gpt-3 to start augmenting
                if augment_turn_goal:
                    
                    # select demos for gpt-3
                    if not augment_demos:
                        augment_demos = sample_demo_turns(single_turn_info, augment_turn_goal, args)
                        
                    # add dontcare if demos contain dontcare
                    # augment_turn_goal = add_dontcare(augment_demos, augment_turn_goal)
                    
                    if augment_demos:
                        prompt = generate_prompt_for_dst(norm_schema, augment_turn_goal, augment_demos, prompt_startswith)

                        # generate user utterance as seeds
                        outputs = openai.Completion.create(engine="text-davinci-002",
                                                            prompt=prompt,
                                                            temperature=0.7,
                                                            max_tokens=64,
                                                            n=args.n_user,
                                                            top_p=1,
                                                            frequency_penalty=1,
                                                            presence_penalty=0,
                                                            stop=["\n", "Assistant:", "You("]
                                                            )["choices"]

                        users = [output["text"].lower().replace("\n", "").strip() for output in outputs]
                        if users:

                            # change augment_turn_goal(turn-level) to augment_goal(dialogue-level)
                            augment_goal = copy.deepcopy(orig_goal)
                            for domain, domain_bs in augment_turn_goal.items():
                                if domain in augment_goal:
                                    for slot, slot_value in domain_bs.items():
                                        augment_goal[domain][slot] = slot_value
                                else:
                                    augment_goal[domain] = copy.deepcopy(domain_bs)
                                    
                            # change it to the pptod format
                            augment_bspn = paser_dict_to_bs(augment_goal)
                            augment_bsdx = paser_dict_to_bsdx(augment_goal)
                            augment_bspn_reform = paser_dict_to_bs_reform(augment_goal)
                            augment_bsdx_reform = paser_dict_to_bsdx_reform(augment_goal)

                            for user in users:
                                user = prompt_startswith + user

                                if not detect_error_turn(user, augment_turn_goal):

                                    if args.debug == "True":
                                        print(Fore.GREEN+ f"Prompt: {prompt}" + Style.RESET_ALL)
                                        print(Fore.YELLOW + f"Last turn dialogue action: {last_turn_aspn}" + Style.RESET_ALL)
                                        print(Fore.YELLOW + f"Last turn system response: {last_turn_resp}" + Style.RESET_ALL)
                                        print(Fore.CYAN + f"Augment_type: {type}" + Style.RESET_ALL)
                                        print(Fore.BLUE + f"Augment turn goal: {augment_turn_goal}" + Style.RESET_ALL)
                                        print(Fore.RED + f"Augment user: {user}" + Style.RESET_ALL)

                                    # record the augment_turn
                                    turn = {}
                                    turn['user'] = user
                                    turn['usdx'] = user
                                    turn['bspn'] = augment_bspn
                                    turn['bsdx'] = augment_bsdx
                                    turn['bspn_reform'] = augment_bspn_reform
                                    turn['bsdx_reform'] = augment_bsdx_reform
                                    augment_turn_list.append(turn)

                                    # record augment turn for each type
                                    if type:
                                        type_aug_num[type-1] += 1
                                        dialog_type_aug_num[type-1] += 1
                                        # if dialog_type_aug_num[1]>0 and dialog_type_aug_num[2]>0 and dialog_type_aug_num[3]>0:
                                        #     exit()
                                        

            # save the info in dict
            augment_turn["augment_turns"] = augment_turn_list
            augment_turns[turn_id] = augment_turn
            
            total_turn_num += 1
            total_aug_turn_num += len(augment_turn_list)

            print(Fore.GREEN+ f"Current turn num: {total_turn_num}, augment turn num: {total_aug_turn_num}" + Style.RESET_ALL)
            print(Fore.GREEN+ f"Augment turn for each type: {type_aug_num}." + Style.RESET_ALL)

        augment_dst_turn_info[dial_id] = augment_turns

    print(f"Total {total_turn_num} turns, augment {total_aug_turn_num} turns, {total_aug_slot_num} slots!")

    return augment_dst_turn_info

import argparse
if __name__ == '__main__':

    args = parse_config()
    
    print ('Start loading data...')
    from dataclass import MultiWozData

    if args.data_version == "2.0":
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-dialogic-processed")

    elif args.data_version == "2.1":
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.1-dialogic-processed")

    elif args.data_version == "2.3":
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.3-dialogic-processed")

    elif args.data_version == "2.4":
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.4-dialogic-processed")

    else:
        raise Exception("Wrong MultiWOZ version!")

    if args.eva_mode == 'dev':
        eva_mode = 'dev'
    elif args.eva_mode == 'test':
        eva_mode = 'test'
    elif args.eva_mode == 'train':
        eva_mode = 'train'
    elif args.eva_mode == 'all':
        eva_mode = 'all'
    else:
        raise Exception('Wrong Evaluation Mode!!!')

    if args.train_data_ratio > 1:
        raise Exception('Wrong Evaluation Mode!!!')
    elif args.train_data_ratio < 0:
        raise Exception('Wrong Evaluation Mode!!!')
    else:
        train_data_ratio = args.train_data_ratio

    if eva_mode == 'train':
        one_dev_str = f"{eva_mode}_ratio_{train_data_ratio}"
    else:
        one_dev_str = f"{eva_mode}"
    
    if args.debug == 'True':
        debug = True
    elif args.debug == 'False':
        debug = False
    else:
        raise Exception('Wrong debug Mode!!!')

    # load openai key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # load possible slot values
    assert args.possible_slot_values_path is not None
    f = open(args.possible_slot_values_path, "r")
    possible_slot_values = json.load(f)
    f.close()

    # load multiwoz schema
    assert args.multiwoz_schema_path is not None
    f = open(args.multiwoz_schema_path, "r")
    schema = json.load(f)
    f.close()

    assert save_output_path is not None

    print("Start loading the dialogs with single turn infos......")
    save_dialog_turn_info_path = os.path.join(save_output_path, "dialog_turn_info_" + one_dev_str + ".json")    
    f = open(save_dialog_turn_info_path, "r")
    dialogs_with_turn_info = json.load(f)
    f.close()

    print("Start loading single turn infos......")
    save_single_turn_info_path = os.path.join(save_output_path, "single_turn_info_" + one_dev_str + ".json")
    f = open(save_single_turn_info_path, "r")
    single_turn_info = json.load(f)
    f.close()

    print("Start loading existing augmented dialogs......")
    augment_dst_turn_info = {}
    augment_time = args.augment_time
    for i in range(args.augment_time, 1, -2):
        save_augment_dialog_turn_info_path = os.path.join(save_output_path, f"{args.k_shot}_shot_x{i}_dst_turn_info_" + one_dev_str + ".json")
        if os.path.exists(save_augment_dialog_turn_info_path):
            f = open(save_augment_dialog_turn_info_path, "r")
            augment_dst_turn_info = json.load(f)
            augment_time -= i
            print(f"Loaded augment dialogs, num of dialogs {len(augment_dst_turn_info)}, need {augment_time} augment time......")
            f.close()
            break
            
    """
    Start constructing the augmentation prompt for DST
    """
    print("Start augmenting dialogs' goal and message......")
    # augment dialogue data
    augment_dst_turn_info = construct_augment_dst(dialogs_with_turn_info, augment_dst_turn_info, augment_time, schema, possible_slot_values, args)

    print("Start saving augmented dialogs......")
    # save the augmented dialog turn info
    assert save_output_path is not None
    save_augment_dialog_turn_info_path = os.path.join(save_output_path, f"{args.k_shot}_shot_x{args.augment_time}_dst_turn_info_" + one_dev_str + ".json")
    f = open(save_augment_dialog_turn_info_path, "w")
    json.dump(augment_dst_turn_info, f)
    f.close()


