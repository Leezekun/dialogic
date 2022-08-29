
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
from transformers import AutoTokenizer

from dialogic_utils import system_prefix, user_prefix
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
    parser.add_argument('--simplified_prompt', type=bool, default=True, help="whether to use simplified message as prompt")
    parser.add_argument('--augment_type', type=str, default="combine", help='how to augment the dialog (only work using simplified prompt), choice: [substitute, drop, combine, random]')
    parser.add_argument('--augment_time', type=int, default=1, help='the augment size compared with the original dataset')
    parser.add_argument('--k_shot', type=int, default=2, help='the maximum number of demo dialogs')
    parser.add_argument('--temperature', type=float, default=0.2, help='the temperature in softmax, for the sampling in combine.')
    parser.add_argument('--length_limit', type=int, default=2048, help='the length limit of the prompt.')

    return parser.parse_args()

# intro = "You are talking with a booking assistant. Please tell the assistant your requirements to let him help you. "
intro = "The following are conversations between a user and an assistant.  The assistant can help the user to find things that satisfy his requirements. Try to speak differently in different conversations."
system_prefix_text = "Assistant: "


def softmax(x, temperature=1.0):
    x = [_/temperature for _ in x]
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def convert_turn_info_to_conversation(turn_info_dict):
    # generate conversation demo based the turn_info
    conversation_text = []
    for turn_id, turn_info in turn_info_dict.items():
        # construct text for bs
        turn_bs = turn_info['bs']
        bs_text = []
        for domain, bs in turn_bs.items():
            bs_text.append(domain)
            if bs:
                domain_text = []
                # bspn_reform format, for the ease of parsing
                for slot_name, slot_value in bs.items():
                    domain_text.append(f"{slot_name} is {slot_value}")
                domain_text = " , ".join(domain_text)
                bs_text.append(domain_text)
                
        bs_text = " ".join(bs_text) # example: [hotel] area east stars 4

        """
        Prompt conversation format

        Should be consistent with dialogic_simulation.py !!!

        Examples:
        You require([hotel] type is guest house, internet is yes): i would like it to be a guest house that has free wifi . 
        You require([train]): can you give me the reference number of the train ? 
        You require([general]): thank you. that's all i need today . 
        """
        user_text = user_prefix(bs_text, turn_info['user']) # f"You require({bs_reform}): {user}"
        system_text = system_prefix(turn_info['aspn'], turn_info['resp']) # f"Assistant({aspn}): {system}"

        conversation_text.append(user_text)
        conversation_text.append(system_text)
    
    conversation_text = "\n".join(conversation_text)
    return conversation_text


def generate_prompt_for_e2e(augment_dialog_with_turn_info, simplified_prompt):

    def paser_goal_to_message(goal):
        message = []   
        for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
            if domain != '[general]':
                domain_pure_text = domain.replace("[","").replace("]", "")
                domain_message = []
                if bs:
                    domain_message.append(domain)
                    domain_text = []
                    for slot_name, slot_value in bs.items():
                        domain_text.append(f"{slot_name} is {slot_value}")
                    domain_text = " , ".join(domain_text)
                    domain_message.append(domain_text)
                    domain_message = " ".join(domain_message)

                if domain_message:
                    if len(message) == 0:
                        domain_message = f"You are going to book a {domain_pure_text}, and your requirements for the {domain_pure_text} are ({domain_message})."
                    else:
                        domain_message = f"You also want to book a {domain_pure_text}, and your requirements for the {domain_pure_text} are ({domain_message})."
                else:
                    if len(message) == 0:
                        domain_message = f"You are going to book a {domain_pure_text}."
                    else:
                        domain_message = f"You also want to book a {domain_pure_text}."

                message.append(domain_message)

        message = " ".join(message)
        return message
    
    if not simplified_prompt:
        """
        use the original message as prompt
        """
        prompt_text = [intro]
        augment_message = augment_dialog_with_turn_info['augment_message']
        
        for idx, dialog_with_turn_info in enumerate(augment_dialog_with_turn_info['augment_demos']):
            message = dialog_with_turn_info['message']
            dialog_turn_info = dialog_with_turn_info['info_turns']

            # requirement
            requirement = f"Requirement{idx+1}: {message}"
            prompt_text.append(requirement)

            # conversation
            conversation_text = convert_turn_info_to_conversation(dialog_turn_info)
            conversation_text = f"Conversation{idx+1}:\n{conversation_text}"
            prompt_text.append(conversation_text)

        # requirement
        requirement = f"Requirement{idx+2}: {augment_message}"
        prompt_text.append(requirement)

        # conversation
        conversation_text = f"Conversation{idx+2}:"
        prompt_text.append(conversation_text)

        prompt = "\n".join(prompt_text)

    else:
        """
        use the message converted from goal (bs_reform) as prompt
        """
        prompt_text = [intro]
        augment_goal = augment_dialog_with_turn_info['augment_goal']
        augment_message = paser_goal_to_message(augment_goal)

        for idx, dialog_with_turn_info in enumerate(augment_dialog_with_turn_info['augment_demos']):
            goal = dialog_with_turn_info['goal']
            dialog_turn_info = dialog_with_turn_info['info_turns']

            message = paser_goal_to_message(goal) + " " + "Make sure you get the booking information once booked."

            # requirement
            requirement = f"Requirement{idx+1}: {message}"
            prompt_text.append(requirement)

            # conversation
            conversation_text = convert_turn_info_to_conversation(dialog_turn_info)
            conversation_text = f"Conversation{idx+1}:\n{conversation_text}"
            prompt_text.append(conversation_text)

        # requirement
        requirement = f"Requirement{idx+2}: {augment_message}"
        prompt_text.append(requirement)

        # conversation
        conversation_text = f"Conversation{idx+2}:"
        prompt_text.append(conversation_text)

        prompt = "\n".join(prompt_text)

    return prompt


def construct_augment_dialog(ref_dialogs_with_turn_info, base_dialogs_with_turn_info, possible_slot_values, args):
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # args configuration
    augment_type = args.augment_type
    augment_time = args.augment_time
    simplified_prompt = args.simplified_prompt

    # first normalize multiwoz2.2's schema file to this format
    norm_schema = normalize_domain_slot(schema)
    
    augment_dialogs_with_turn_info = {}

    # record not changed value slots
    not_change_slots = {}
    total_slot_num = 0

    # record augment dialogues num

    for idx in range(augment_time):

        while len(augment_dialogs_with_turn_info) < len(base_dialogs_with_turn_info)*(idx+1):

            base_dialogs_with_turn_info = random_dict(base_dialogs_with_turn_info)
            for dial_id, dialog_with_turn_info in base_dialogs_with_turn_info.items():

                message = dialog_with_turn_info['message']
                orig_goal = dialog_with_turn_info['goal']

                # the info stored in augment_dialog_with_turn_info
                augment_dialog_with_turn_info = {}
                augment_demos = [] # the dialogs for demonstrations (real dialogs)
                augment_goal = {}
                augment_message = ""

                if not simplified_prompt:
                    """
                    Modify the augment_goals and the message
                    only support substitute
                    """

                    aug_dial_id = "msg_" + dial_id + f"_{idx}"
                    augment_demos.append(dialog_with_turn_info) # only need this original dialog_with_turn_info
                    augment_message = copy.deepcopy(message)
                    augment_goal = copy.deepcopy(orig_goal)

                    # modify augment_goal and augment_message
                    slot_texts = re.findall(r"\*(.+?)\*", message, re.I)
                    for domain, domain_bs in orig_goal.items():
                        
                        augment_domain = domain.split("[")[1].split("]")[0].strip() # [hotel] -> hotel
                        augment_domain_bs = copy.deepcopy(domain_bs)

                        augment_domain_bs, augment_message, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs, augment_message)
                        augment_goal[domain] = augment_domain_bs
                        total_slot_num += slot_num

                else:
                    """
                    not based on message, use bs_reform as prompt
                    type: [substitute, drop, combine]
                    """
                    # augment_type = random.choice(["combine", "combine", "substitute", "substitute", "drop"])

                    if augment_type == "substitute":
                        
                        # the info to be recorded
                        aug_dial_id = "subs_" + dial_id + f"_{idx}"
                        augment_goal = {}
                        augment_message = "" # don't need
                        augment_demos.append(dialog_with_turn_info) # only need this original dialog_with_turn_info

                        for domain, domain_bs in orig_goal.items():
                            
                            augment_domain = domain.split("[")[1].split("]")[0].strip() # [hotel] -> hotel
                            augment_domain_bs = copy.deepcopy(domain_bs)

                            augment_domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs)
                            augment_goal[domain] = augment_domain_bs
                            total_slot_num += slot_num

                    elif augment_type == "drop":
                        
                        # the info to be recorded
                        aug_dial_id = "drop_" + dial_id
                        augment_goal = {}
                        augment_message = "" # don't need
                        augment_demos.append(dialog_with_turn_info) # only need this original dialog_with_turn_info

                        orig_goal_domains = list(orig_goal.keys())
                        if "[general]" in orig_goal_domains:
                            orig_goal_domains.remove("[general]")

                        # if there is not a domain
                        if len(orig_goal_domains) < 1:
                            continue
                        
                        # random drop some domains
                        selected_domains = orig_goal_domains
                        
                        for domain in selected_domains:
                            domain_bs = orig_goal[domain]

                            augment_domain = domain.split("[")[1].split("]")[0].strip() # [hotel] -> hotel
                            augment_domain_bs = {}
                            
                            if domain_bs:
                                domain_bs_name = list(domain_bs.keys())
                                if len(domain_bs_name) > 0:
                                    # drop at most 2 slots
                                    preserved_slot_num = random.randint(max(len(domain_bs_name)-2, 1), len(domain_bs_name))
                                    random.shuffle(domain_bs_name)
                                    preserved_slots = domain_bs_name[:preserved_slot_num]
                                    for slot in preserved_slots:
                                        augment_domain_bs[slot] = copy.deepcopy(domain_bs[slot])
                            else:
                                augment_domain_bs = copy.deepcopy(domain_bs)

                            augment_domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs)
                            augment_goal[domain] = augment_domain_bs
                            total_slot_num += slot_num

                    elif augment_type == "combine": # random select another dialog to combine with the current one
                        
                        assert args.k_shot == 2
                        
                        augment_goal = {}
                        augment_message = "" # don't need
                        aug_dial_id = ""
                        augment_demos.append(dialog_with_turn_info) # need this original dialog_with_turn_info and another one
                        
                        orig_goal_domains = list(orig_goal.keys())
                        if "[general]" in orig_goal_domains:
                            orig_goal_domains.remove("[general]")
                        
                        """
                        select another dialog to combine with this one according to similarity
                        """
                        similarities_dict = calculate_dialog_similarity(ref_dialogs_with_turn_info, orig_goal, dial_id)
                        other_dial_ids = similarities_dict['dial_ids']
                        other_dial_similarities = similarities_dict['dial_similarities']
                        
                        # sample based on dissimilarity, prefer the different dialogs with the current dialog to increase diversity
                        other_dial_weights = softmax([1-sim for sim in other_dial_similarities], args.temperature)
                        
                        another_dial_id = random.choices(other_dial_ids, weights=other_dial_weights, k=1)[0]
                        another_dialog_with_turn_info = ref_dialogs_with_turn_info[another_dial_id]
                        another_goal = another_dialog_with_turn_info['goal']

                        another_goal_domains = []
                        for domain, domain_bs in another_goal.items():
                            if domain_bs:
                                another_goal_domains.append(domain)
                            else: # avoid containing this kind of domain
                                if domain in ["[police]", "[hospital]"]:
                                    another_goal_domains = []
                                    break
                        another_dial_domain_num = len(another_goal_domains)

                        while another_dial_domain_num < 1:
                            another_dial_id = random.choices(other_dial_ids, weights=other_dial_weights, k=1)[0]
                            another_dialog_with_turn_info = ref_dialogs_with_turn_info[another_dial_id]
                            another_goal = another_dialog_with_turn_info['goal']
                            
                            another_goal_domains = []
                            for domain, domain_bs in another_goal.items():
                                if domain_bs:
                                    another_goal_domains.append(domain)
                                else: # avoid containing this kind of domain
                                    if domain in ["[police]", "[hospital]"]:
                                        another_goal_domains = []
                                        break
                            another_dial_domain_num = len(another_goal_domains)

                        # update augment_dial_id and turn_info
                        # comb_id1_id2
                        aug_dial_id = f"comb_{dial_id}_{another_dial_id}"
                        augment_demos.append(another_dialog_with_turn_info)

                        """
                        random select some domains from the another dialog and combine it with the current dialog
                        """
                        new_goal_domains = another_goal_domains + orig_goal_domains
                        new_goal_domains = list(set(new_goal_domains))
                        domain_slot_nums = random.choice([[2,1,3], [3,1,2]])
                        domain_num, min_domain_slot_num, max_domain_slot_num = domain_slot_nums[0], domain_slot_nums[1], domain_slot_nums[2] 
                        preserved_domain_num = min(domain_num, len(new_goal_domains)) # keep at most 2-3 domains
                        # preserved_domain_num = min(3, len(new_goal_domains)) # keep at most 3 domains
                        random.shuffle(new_goal_domains)
                        preserved_domains = new_goal_domains[:preserved_domain_num]
                        
                        for domain in preserved_domains:
                            if domain in orig_goal and domain in another_goal:
                                ori_domain_bs = orig_goal[domain]
                                another_domain_bs = another_goal[domain]
                                # combine two goals, another goal over original goal
                                domain_bs = ori_domain_bs.copy()
                                domain_bs.update(another_domain_bs)
                            elif domain in orig_goal and domain not in another_goal:
                                ori_domain_bs = orig_goal[domain]
                                domain_bs = copy.deepcopy(ori_domain_bs)
                            elif domain not in orig_goal and domain in another_goal:
                                another_domain_bs = another_goal[domain]
                                domain_bs = copy.deepcopy(another_domain_bs)
                            else:
                                raise Exception("Wrong domain!!!")

                            augment_domain = domain.split("[")[1].split("]")[0].strip() # [hotel] -> hotel
                            augment_domain_bs = {}
                            
                            if domain_bs:
                                domain_bs_name = list(domain_bs.keys())
                                if len(domain_bs_name) > 0:
                                    # preserved_slot_num = random.randint(min(len(domain_bs_name), 2), min(len(domain_bs_name), 4))
                                    preserved_slot_num = random.randint(min(len(domain_bs_name), min_domain_slot_num), min(len(domain_bs_name), max_domain_slot_num))
                                    random.shuffle(domain_bs_name)
                                    preserved_slots = domain_bs_name[:preserved_slot_num]
                                    for slot in preserved_slots:
                                        augment_domain_bs[slot] = copy.deepcopy(domain_bs[slot])
                            else:
                                augment_domain_bs = copy.deepcopy(domain_bs)

                            augment_domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs)
                            augment_goal[domain] = augment_domain_bs
                            total_slot_num += slot_num

                    elif augment_type == "random": # random generate goals
                        
                        augment_message = "" # don't need
                        aug_dial_id = "rand" # according to the selected demo, not the orig_goal
                        augment_goal, slot_num = random_generate_goal(norm_schema, possible_slot_values)
                        total_slot_num += slot_num
                        
                        # select demo dialogs to cover the not_mention_goal
                        augment_goal, aug_dial_id, augment_demos = sample_demo_dialogs(ref_dialogs_with_turn_info, augment_goal, aug_dial_id, augment_demos, args.k_shot)
                    
                    elif augment_type == "ref_test":
                        assert augment_dialogs_with_turn_info is not None
                        augment_message = "" # don't need
                        aug_dial_id = f"ref_test_{dial_id}" # according to the selected demo, not the orig_goal
                        augment_goal = copy.deepcopy(orig_goal)
                        slot_num = len(paser_bs_from_dict_to_list(augment_goal))
                        total_slot_num += slot_num

                        # find the most similar demo dialogs
                        augment_goal, aug_dial_id, augment_demos = sample_demo_dialogs(ref_dialogs_with_turn_info, augment_goal, aug_dial_id, augment_demos, args.k_shot)

                    else:
                        raise Exception("Wrong augment type!")

                augment_dialog_with_turn_info["augment_demos"] = augment_demos
                augment_dialog_with_turn_info["augment_message"] = augment_message
                augment_dialog_with_turn_info["augment_goal"] = augment_goal

                """
                Generate prompt for this augmented dialog !!!
                """
                prompt = generate_prompt_for_e2e(augment_dialog_with_turn_info, simplified_prompt)
                
                # avoid too lengthy prompt, try to make the length of two examples and the generated one less than 2048
                tokenized_prompt_length = len(tokenizer(prompt).input_ids)
                if tokenized_prompt_length > args.length_limit:
                    continue

                augment_dialog_with_turn_info['prompt'] = prompt

                # save all information for this dialog
                augment_dialogs_with_turn_info[aug_dial_id] = augment_dialog_with_turn_info

                # exit if has augment one time
                if len(augment_dialogs_with_turn_info) >= len(base_dialogs_with_turn_info)*(idx+1):
                    break

        print(f"Not changed slots:{not_change_slots}")
        print(f"Total slots:{total_slot_num}")
        print(f"Total augmented dialogs:{len(augment_dialogs_with_turn_info)}")

    return augment_dialogs_with_turn_info


def sample_demo_dialogs(ref_dialogs_with_turn_info, augment_goal, aug_dial_id, augment_demos, k_shot):
    not_mention_goal = copy.deepcopy(augment_goal)
    for k in range(k_shot):

        # find the similar real dialogs as demo
        similarities_dict = calculate_dialog_similarity(ref_dialogs_with_turn_info, not_mention_goal, "")
        other_dial_ids = similarities_dict['dial_ids']
        other_dial_similarities = similarities_dict['dial_similarities']

        # select the one with highest similarity
        another_dial_id = other_dial_ids[other_dial_similarities.index(max(other_dial_similarities))]
        another_dialog_with_turn_info = ref_dialogs_with_turn_info[another_dial_id]
        another_goal = another_dialog_with_turn_info['goal']

        # record the ref dialogs
        aug_dial_id += f"_{another_dial_id}"
        augment_demos.append(another_dialog_with_turn_info)


        # check which domain and slot hasn't been mentioned
        new_not_mention_goal = copy.deepcopy(not_mention_goal)
        not_mention_domain = []
        for domain, domain_bs in not_mention_goal.items():
            new_domain_bs = new_not_mention_goal[domain]
            if domain in another_goal:
                another_domain_bs = another_goal[domain]
                for slot in another_domain_bs:
                    if slot in domain_bs:
                        new_domain_bs.pop(slot) # don't need a slot_value
                if not new_domain_bs:
                    new_not_mention_goal.pop(domain)
            else:
                if domain != "[general]":
                    not_mention_domain.append(domain)
        
        del not_mention_goal
        not_mention_goal = copy.deepcopy(new_not_mention_goal)
        del new_not_mention_goal
        
        if not not_mention_goal:
            break
    
    return augment_goal, aug_dial_id, augment_demos


def calculate_dialog_similarity(dialogs_with_turn_info, query_goal, query_dial_id, metric="jaccard"):
    
    def jaccard(list1, list2):
        intersection = list(set(list1) & set(list2))
        unionset = list(set(list1).union(set(list2)))
        if unionset:
            return float(len(intersection) / len(unionset))
        else:
            return 0.0

    def cover(list1, list2):
        intersection = list(set(list1) & set(list2))
        if intersection == list1:
            return 1.0
        else:
            return 0.0

    dial_id1 = query_dial_id
    goal1 = query_goal
    goal_list1 = paser_bs_from_dict_to_list(goal1)
    goal_domain1 = list(goal1.keys())
    if "[general]" in goal_domain1:
        goal_domain1.remove("[general]")

    other_dial_ids = []
    other_dial_similarities = []
    other_dial_cover_similarities = []

    for dial_id2, dialog_with_turn_info2 in dialogs_with_turn_info.items():
        if dial_id2 == dial_id1:
            continue

        goal2 = dialog_with_turn_info2['goal']
        goal_list2 = paser_bs_from_dict_to_list(goal2)
        goal_domain2 = list(goal2.keys())
        if "[general]" in goal_domain2:
            goal_domain2.remove("[general]")

        if_cover = cover(goal_domain1, goal_domain2)
        domain_similarity = jaccard(goal_domain1, goal_domain2)
        bs_similarity = jaccard(goal_list1, goal_list2)
        similarity = domain_similarity * bs_similarity 
        cover_similarity = if_cover * domain_similarity * bs_similarity 

        other_dial_ids.append(dial_id2)
        other_dial_similarities.append(similarity)
        other_dial_cover_similarities.append(cover_similarity)

    if np.sum(other_dial_cover_similarities) <= 0:
        if np.sum(other_dial_similarities) <= 0:
            print(f"This goal doesn't has a similar one: {goal1}")
        else:
            other_dial_cover_similarities = other_dial_similarities

    similarity_dict = {"dial_ids": other_dial_ids, "dial_similarities": other_dial_cover_similarities}

    return similarity_dict



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

    if args.augment_type == 'substitute':
        augment_type = 'substitute'
    elif args.augment_type == 'drop':
        augment_type = 'drop'
    elif args.augment_type == 'combine':
        augment_type = 'combine'
    elif args.augment_type == 'random':
        augment_type = 'random'
    else:
        raise Exception('Wrong Augmentation Type!!!')

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

    """
    Start constructing the augmentation prompt for E2E
    """
    # these two are the same
    base_dialogs_with_turn_info = copy.deepcopy(dialogs_with_turn_info) # based on these dialogs we generate the augmentation prompt
    ref_dialogs_with_turn_info = copy.deepcopy(dialogs_with_turn_info) # based on these dialogs we select demos

    print("Start augmenting dialogs' goal and message......")
    # augment dialogue data
    augment_dialogs_with_turn_info = construct_augment_dialog(ref_dialogs_with_turn_info, base_dialogs_with_turn_info, possible_slot_values, args)

    print("Start saving augmented dialogs......")
    # save the augmented dialog turn info
    assert save_output_path is not None
    if augment_type == 'combine':
        save_augment_dialog_turn_info_path = os.path.join(save_output_path, f"{augment_type}{args.temperature}_{args.k_shot}_shot_augment_dialog_turn_info_" + one_dev_str + ".json")
    else:
        save_augment_dialog_turn_info_path = os.path.join(save_output_path, f"{augment_type}_{args.k_shot}_shot_augment_dialog_turn_info_" + one_dev_str + ".json")
    f = open(save_augment_dialog_turn_info_path, "w")
    json.dump(augment_dialogs_with_turn_info, f)
    f.close()


