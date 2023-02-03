import json
import random
import re
import copy
import os
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

num2word = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}

UBAR_SLOT_VALUE_DICT = {
"hotel-internet": ['yes'] ,
"hotel-type": ['hotel', 'guesthouse'],
"hotel-parking": ['yes'] ,
"hotel-pricerange": ['moderate', 'cheap', 'expensive'] ,
"hotel-day": ["march 11th", "march 12th", "march 13th", "march 14th", "march 15th", "march 16th", "march 17th", 
                   "march 18th", "march 19th", "march 20th"],
"hotel-people": ["20","21","22","23","24","25","26","27","28","29"],
"hotel-stay": ["20","21","22","23","24","25","26","27","28","29"],
"hotel-area": ['south', 'north', 'west', 'east', 'centre'],
"hotel-stars": ['0', '1', '2', '3', '4', '5'] ,
"hotel-name":["moody moon", "four seasons hotel", "knights inn", "travelodge", "jack summer inn", "paradise point resort"],
"restaurant-area": ['south', 'north', 'west', 'east', 'centre'],
"restaurant-food": ['asian fusion', 'burger', 'pasta', 'ramen', 'taiwanese'],
"restaurant-pricerange": ['moderate', 'cheap', 'expensive'] ,
"restaurant-name": ["buddha bowls","pizza my heart","pho bistro","sushiya express","rockfire grill","itsuki restaurant"],
"restaurant-day": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
"restaurant-people": ["20","21","22","23","24","25","26","27","28","29"],
"restaurant-time":["19:01","18:06","17:11","19:16","18:21","17:26","19:31","18:36","17:41","19:46","18:51","17:56",
                        "7:00 pm","6:07 pm","5:12 pm","7:17 pm","6:17 pm","5:27 pm","7:32 pm","6:37 pm","5:42 pm","7:47 pm","6:52 pm","5:57 pm",
                        "11:00 am","11:05 am","11:10 am","11:15 am","11:20 am","11:25 am","11:30 am","11:35 am","11:40 am","11:45 am","11:50 am",
                        "11:55 am"],
"taxi-arrive": [ "17:26","19:31","18:36","17:41","19:46","18:51","17:56",
                    "7:00 pm","6:07 pm","5:12 pm","7:17 pm","6:17 pm","5:27 pm",
                    "11:30 am","11:35 am","11:40 am","11:45 am","11:50 am","11:55 am"],
"taxi-leave":  [ "19:01","18:06","17:11","19:16","18:21",
                    "7:32 pm","6:37 pm","5:42 pm","7:47 pm","6:52 pm","5:57 pm",
                    "11:00 am","11:05 am","11:10 am","11:15 am","11:20 am","11:25 am"],
"taxi-departure":   ["moody moon", "four seasons hotel", "knights inn", "travelodge", "jack summer inn", "paradise point resort"],
"taxi-destination": ["buddha bowls","pizza my heart","pho bistro","sushiya express","rockfire grill","itsuki restaurant"],
"train-arrive": [ "17:26","19:31","18:36","17:41","19:46","18:51","17:56",
                    "7:00 pm","6:07 pm","5:12 pm","7:17 pm","6:17 pm","5:27 pm",
                    "11:30 am","11:35 am","11:40 am","11:45 am","11:50 am","11:55 am"],
"train-leave":   [ "19:01","18:06","17:11","19:16","18:21",
                    "7:32 pm","6:37 pm","5:42 pm","7:47 pm","6:52 pm","5:57 pm",
                    "11:00 am","11:05 am","11:10 am","11:15 am","11:20 am","11:25 am"],
"train-departure": ["gilroy","san martin","morgan hill","blossom hill","college park","santa clara","lawrence","sunnyvale"],
"train-destination":["mountain view","san antonio","palo alto","menlo park","hayward park","san mateo","broadway","san bruno"],
"train-day":       ["march 11th", "march 12th", "march 13th", "march 14th", "march 15th", "march 16th", "march 17th", 
                   "march 18th", "march 19th", "march 20th"],

"train-people":["20","21","22","23","24","25","26","27","28","29"],
"attraction-area": ['south', 'north', 'west', 'east', 'centre'],
"attraction-name": ["grand canyon","golden gate bridge","niagara falls","kennedy space center","pike place market","las vegas strip"],
"attraction-type": ['historical landmark', 'aquaria', 'beach', 'castle','art gallery'],
"hospital-department":
[
    "antenatal",
    "childrens surgical and medicine",
    "haematology and haematological oncology",
    "infectious diseases",
    "medical decisions unit",
    "teenage cancer trust unit",
    "transitional care",
    "john farman intensive care unit",
    "urology",
    "intermediate dependancy area",
    "respiratory medicine",
    "neonatal unit",
    "inpatient occupational therapy",
    "gynaecology",
    "medicine for the elderly",
    "neurology",
    "trauma high dependency unit",
    "cardiology",
    "cardiology and coronary care unit",
    "hepatobillary and gastrointestinal surgery regional referral centre",
    "acute medical assessment unit",
    "gastroenterology",
    "oral and maxillofacial surgery and ent",
    "hepatology",
    "acute medicine for the elderly",
    "transplant high dependency unit",
    "plastic and vascular surgery plastics",
    "clinical research facility",
    "infusion services",
    "cambridge eye unit",
    "dontcare",
    "coronary care unit",
    "neurology neurosurgery",
    "paediatric clinic",
    "oncology",
    "diabetes and endocrinology",
    "neurosciences",
    "clinical decisions unit",
    "surgery",
    "psychiatry",
    "paediatric intensive care unit",
    "emergency department",
    "haematology",
    "childrens oncology and haematology",
    "neurosciences critical care unit",
    "paediatric day unit",
    "trauma and orthopaedics",
    "haematology day unit"
  ]
}

all_domain = [
    "[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]"
]

requestable_slots = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}
all_reqslot = ["car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]
# count: 17

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
all_infslot = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
                     "leave", "destination", "departure", "arrive", "department", "food", "time"]
# count: 17

all_slots = all_reqslot + all_infslot + ["stay", "day", "people", "name", "destination", "departure", "department"]
all_slots = set(all_slots)

all_acts = ['[inform]', '[request]', '[nooffer]', '[recommend]', '[select]', '[offerbook]', '[offerbooked]', '[nobook]', '[bye]', '[greet]', '[reqmore]', '[welcome]']



GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports",
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall",
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east",
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre",
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
        # day
        "next friday":"friday", "monda": "monday",
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        # others
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",
        }

def normalize_domain_slot(schema):
    normalized_schema = []
    for service in schema:
        if service['service_name'] == 'bus':
            service['service_name'] = 'taxi'

        slots = service['slots']
        normalized_slots = []

        for slot in slots: # split domain-slots to domains and slots
            domain_slot = slot['name']
            domain, slot_name = domain_slot.split('-')
            if domain == 'bus':
                domain = 'taxi'
            if slot_name == 'bookstay':
                slot_name = 'stay'
            if slot_name == 'bookday':
                slot_name = 'day'
            if slot_name == 'bookpeople':
                slot_name = 'people'
            if slot_name == 'booktime':
                slot_name = 'time'
            if slot_name == 'arriveby':
                slot_name = 'arrive'
            if slot_name == 'leaveat':
                slot_name = 'leave'
            domain_slot = "-".join([domain, slot_name])
            slot['name'] = domain_slot
            normalized_slots.append(slot)

        service['slots'] = normalized_slots
        normalized_schema.append(service)

    return normalized_schema
         
def paser_bs_to_list(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    sent = sent.split()
    belief_state = []
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        sub_span = sent[d_idx+1:next_d_idx]
        sub_s_idx = [idx for idx,token in enumerate(sub_span) if token in all_slots]
        for j,s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j+1]
            slot = sub_span[s_idx]
            value = ' '.join(sub_span[s_idx+1:next_s_idx])
            bs = " ".join([domain,slot,value])
            belief_state.append(bs)
    return list(set(belief_state))

def paser_bs_to_dict(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    sent = sent.split()
    belief_state = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in belief_state:
            domain_bs = belief_state[domain]
        else:
            domain_bs = {}
        sub_span = sent[d_idx+1:next_d_idx]
        sub_s_idx = [idx for idx,token in enumerate(sub_span) if token in all_slots]
        for j,s_idx in enumerate(sub_s_idx):
            next_s_idx = len(sub_span) if j+1 == len(sub_s_idx) else sub_s_idx[j+1]
            slot = sub_span[s_idx]
            value = " ".join(sub_span[s_idx+1:next_s_idx])
            bs = " ".join([domain,slot,value])
            domain_bs[slot] = value
        belief_state[domain] = domain_bs
    return belief_state

def paser_bs_reform_to_dict(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
    all_domain = ["[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]","[general]"]
    sent = sent.split()
    belief_state = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain] 
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in belief_state:
            domain_bs = belief_state[domain]
        else:
            domain_bs = {}
        sub_span = " ".join(sent[d_idx+1:next_d_idx])
        for bs in sub_span.split(","):
            if bs and len(bs.split(" is ")) == 2:
                slot_name, slot_value = bs.split(" is ")
                slot_name = slot_name.strip()
                slot_value = slot_value.strip()
                if slot_name and slot_value:
                    domain_bs[slot_name] = slot_value
        belief_state[domain] = domain_bs
    return belief_state

def paser_bs_from_dict_to_list(bs):
        """
        Convert compacted bs span to triple list
        Ex:  
        """
        belief_state = []
        for domain, domain_bs in bs.items():
            if domain_bs:
                for slot_name, slot_value in domain_bs.items():
                    belief_state.append(" ".join([domain, slot_name]))
        return list(set(belief_state))

def paser_dict_to_bs(goal, ignore_none_bs=True):
    bs_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bs_text.append(domain)
        if bs:
            if ignore_none_bs:
                bs_text.append(domain)
            for slot_name, slot_value in bs.items():
                bs_text.append(slot_name)
                bs_text.append(slot_value)

    bs_text = " ".join(bs_text)
    return bs_text

def print_paser_dict(goal):
    for domain, slot in goal.items():
        print("{:<12}>> ".format(domain.replace("[","").replace("]","")) + ", ".join([f"{s} is {v}" for s, v in slot.items()]))
        
def paser_dict_to_bs_reform(goal, ignore_none_bs=True):
    bs_reform_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bs_reform_text.append(domain)
        if bs:
            if ignore_none_bs:
                bs_reform_text.append(domain)
            domain_text = []
            for slot_name, slot_value in bs.items():
                domain_text.append(f"{slot_name} is {slot_value}")
            domain_text = " , ".join(domain_text)
            bs_reform_text.append(domain_text)

    bs_reform_text = " ".join(bs_reform_text)
    return bs_reform_text

def paser_dict_to_bsdx(goal, ignore_none_bs=True):
    bsdx_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bsdx_text.append(domain)
        if bs:
            if ignore_none_bs:
                bsdx_text.append(domain)
            for slot_name, slot_value in bs.items():
                bsdx_text.append(slot_name)

    bsdx_text = " ".join(bsdx_text)
    return bsdx_text

def paser_dict_to_bsdx_reform(goal, ignore_none_bs=True):
    bsdx_reform_text = []            
    for domain, bs in goal.items(): # reverse the dict to align with the original pptod fotmat 
        if not ignore_none_bs:
            bsdx_reform_text.append(domain)
        if bs:
            if ignore_none_bs:
                bsdx_reform_text.append(domain)
            bsdx_domain_text = []
            for slot_name, slot_value in bs.items():
                bsdx_domain_text.append(slot_name)
            bsdx_domain_text = " , ".join(bsdx_domain_text)
            bsdx_reform_text.append(bsdx_domain_text)

    bsdx_reform_text = " ".join(bsdx_reform_text)
    return bsdx_reform_text

def paser_aspn_to_dict(sent):
    sent = sent.split()
    dialog_act = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain+["[general]"]]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in dialog_act:
            domain_da = dialog_act[domain]
        else:
            domain_da = {}
        sub_span = sent[d_idx+1:next_d_idx]
        sub_a_idx = [idx for idx,token in enumerate(sub_span) if token in all_acts]
        for j,a_idx in enumerate(sub_a_idx):
            next_a_idx = len(sub_span) if j+1 == len(sub_a_idx) else sub_a_idx[j+1]
            act = sub_span[a_idx]
            act_slots = sub_span[a_idx+1:next_a_idx]
            domain_da[act] = act_slots
        dialog_act[domain] = domain_da
    return dialog_act
    
def paser_dict_to_list(goal):
    belief_state = []
    for domain, domain_bs in goal.items():
        for slot_name, slot_value in domain_bs.items():
            belief_state.append(" ".join([domain, slot_name, slot_value]))
    return list(set(belief_state))

def system_prefix(aspn=None, system=None):
    if not aspn:
        system_prefix = f"Assistant(["
    else:
        if system:
            system_prefix = f"Assistant({aspn}): {system}"
        else:
            system_prefix = f"Assistant({aspn}):"
    return system_prefix

def user_prefix(bs_reform=None, user=None):
    if not bs_reform:
        user_prefix = f"You require(["
    else:
        if user:
            user_prefix = f"You require({bs_reform}): {user}"
        else:
            user_prefix = f"You require({bs_reform}):"
    return user_prefix

def random_dict(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dict = {}
    for key in dict_key_ls:
        new_dict[key] = dicts.get(key)
    return new_dict

def reverse_dict(dicts):
    dict_key_ls = list(dicts.keys())
    new_dict = {}
    for key in dict_key_ls[::-1]:
        new_dict[key] = dicts.get(key)
    return new_dict

def rearrange_dict(dicts):
    new_dict = {}
    dict_keys = ['dial_id', 'turn_num', 'user', 'usdx', 'turn_domain', 'dspn', 'bspn_gpt3', 'bspn_verifier', 'over_generate', 'de_generate', 
    'bspn', 'bsdx', 'bspn_reform', 'bsdx_reform', 'db', 'pointer', 'aspn_verifier', 'aspn', 'aspn_reform', 'resp', 'resp_verifier', 'resp_gen']
    for k in dict_keys:
        if k in dicts:
            if k == 'resp_gen':
                new_dict['resp_verifier'] = dicts[k]
            else:
                new_dict[k] = dicts[k]
    return new_dict

def compare_dict(old_dict, new_dict):
    differ = {}
    for domain, slot in new_dict.items():
        if domain not in old_dict:
            differ[domain] = slot
        else:
            old_slot = old_dict[domain]
            for slot_name, slot_value in slot.items():
                if slot_name not in old_slot:
                    if domain not in differ:
                        differ[domain] = {}
                    differ[domain][slot_name] = slot_value
                elif old_slot[slot_name] != slot_value:
                    if domain not in differ:
                        differ[domain] = {}
                    differ[domain][slot_name] = slot_value
    return differ

def substitute_domain_slot_value(norm_schema, possible_slot_values, augment_domain, augment_domain_bs, augment_message=None):
    
    augment_domain = augment_domain.replace("[", "").replace("]", "")
    augment_message = copy.deepcopy(augment_message)

    for service in norm_schema:
        if service['service_name'] == augment_domain:
            slot_schema = service['slots']
            break

    slot_num = 0
    for slot_name, slot_value in augment_domain_bs.items():
    
        slot_num += 1

        augment_slot_value = ""
        orig_slot_value = slot_value
        for slot in slot_schema:
            if slot['name'].split("-")[1].strip() == slot_name:
                
                if slot['is_categorical']:
                    possible_values = slot['possible_values']
                    candidate_values = copy.deepcopy(possible_values)
                    if slot_value in candidate_values: candidate_values.remove(slot_value)
                    if candidate_values: augment_slot_value = random.choice(candidate_values)
                elif slot_name in ["arrive", "leave", "time"]:
                    augment_slot_value = "{}:{:0>2d}".format(random.randint(10, 23), random.randint(0, 59))
                else:    
                    norm_slot_text = f"{augment_domain}-{slot_name}"
                    print(f"{norm_slot_text} using possible values in training set!")
                    norm_domain = f"[{augment_domain}]"
                    candidates = copy.deepcopy(possible_slot_values["train"][norm_domain][slot_name])              
                    if slot_value in candidates:
                        candidates.remove(slot_value)
                    augment_slot_value = random.choice(candidates)

        if augment_message:
            slot_texts = re.findall(r"\*(.+?)\*", augment_message, re.I)
            if augment_slot_value: 
                if slot_texts:
                    appear_time = 0
                    # modify the message
                    for slot_text in slot_texts:
                        if (len(slot_value)==1 and slot_value in slot_text.split()) or (len(slot_value)>1 and slot_value in slot_text):
                            appear_time += 1
                    if appear_time == 1: # change the slot value if it appears one and only one time
                        # replace this value with the target value
                        for slot_text in slot_texts:
                            if (len(slot_value)==1 and slot_value in slot_text.split()) or (len(slot_value)>1 and slot_value in slot_text):
                                augment_slot_text = slot_text.replace(slot_value, augment_slot_value)
                                augment_message = augment_message.replace(slot_text, augment_slot_text)
                                augment_domain_bs[slot_name] = augment_slot_value
                                break
                else:
                    appear_time = augment_message.count(orig_slot_value)
                    if appear_time == 1:
                        augment_message = augment_message.replace(orig_slot_value, augment_slot_value)
                        augment_domain_bs[slot_name] = augment_slot_value
            else:
                norm_slot_text = f"{augment_domain}-{slot_name}"
                print(f"{norm_slot_text} doesn't have possible values!")
        else:
            if augment_slot_value: 
                augment_domain_bs[slot_name] = augment_slot_value
            else:
                norm_slot_text = f"{augment_domain}-{slot_name}"
                print(f"{norm_slot_text} doesn't have possible values!")
    
    return augment_domain_bs, augment_message, slot_num


def random_generate_goal(norm_schema, possible_slot_values, selected_domains=None, excluded_domains=['police', 'hospital']):
    from ontology import informable_slots

    # if not given domains, random select some
    if not selected_domains: 
        possible_domains = []
        for domain in informable_slots:
            if domain not in excluded_domains:
                possible_domains.append(domain)
        assert len(possible_domains) >= 3
        # num_domains, min_num_slot, max_num_slot = random.choices([[1, 4, 6], [2, 3, 5], [3, 2, 5], [4, 2, 4]], [0.1, 0.5, 0.3, 0.1], k=1)[0]
        num_domains, min_num_slot, max_num_slot = random.choices([[1, 4, 6], [2, 3, 5], [3, 2, 5], [4, 2, 4]], [0.3, 0.6, 0.1, 0.0], k=1)[0]
        selected_domains = random.sample(possible_domains, k=num_domains)
    else: # if given domains, ususally just one turn
        min_num_slot, max_num_slot = 2, 4

    # generate the goal    
    goal = {}
    
    goal_slot_num = 0
    for domain in selected_domains:
        domain = domain.replace("[", "").replace("]", "") 
        domain_bs = {}
        possible_slots = informable_slots[domain]
        # random select slots
        num_slots = random.randint(min(min_num_slot, len(possible_slots)), min(max_num_slot, len(possible_slots)))
        selected_slots = random.sample(possible_slots, k=num_slots)
        for slot in selected_slots:
            domain_bs[slot] = "" # don't need to fill in now
        domain_bs, _, slot_num = substitute_domain_slot_value(norm_schema, possible_slot_values, domain, domain_bs)
        domain = f"[{domain}]"
        goal[domain] = domain_bs
        goal_slot_num += slot_num

    return goal, goal_slot_num


def detect_error_turn(user, bs):

    if not user:
        return True

    error_turn = False

    for domain, domain_bs in bs.items():
        for slot, slot_value in domain_bs.items():
            # check if the slot appears in user's utterance of this turn
            mentioned_in_this_turn = False
            if slot_value in ['yes', 'no', '0', 'dont care', "don't care", "do nt care", "doesn't care", "dontcare"]: 
                mentioned_in_this_turn = True
            else:
                if slot_value in user:
                    mentioned_in_this_turn = True
                elif slot_value in num2word:
                    if num2word[slot_value] in user:
                        mentioned_in_this_turn = True
                # else:
                #     for norm_slot_value, typo in GENERAL_TYPO.items():
                #         if slot_value == typo:
                #             if ((len(norm_slot_value)==1 and (norm_slot_value in user.split() or norm_slot_value+"." in user or norm_slot_value+"," in user or norm_slot_value+"?" in user)) or (len(norm_slot_value)>1 and norm_slot_value in user)):
                #                 mentioned_in_this_turn = True
                #                 break
                #         if slot_value == norm_slot_value:
                #             if ((len(typo)==1 and (typo in user.split() or typo+"." in user or typo+"," in user or typo+"?" in user)) or (len(typo)>1 and typo in user)):
                #                 mentioned_in_this_turn = True
                #                 break
            if not mentioned_in_this_turn:
                print(f"Slot value {slot_value} not in {user}")
                error_turn = True
                return error_turn

    return error_turn


def detect_valid_bs(goal):
    has_valid_bs = False
    has_dontcare = False
    for domain in goal:
        if goal[domain]:
            has_valid_bs = True
            for slot_value in goal[domain].values():
                if slot_value == 'dontcare':
                    has_dontcare = True
                    return has_valid_bs, has_dontcare
    return has_valid_bs, has_dontcare


if __name__ == '__main__':
    sent = "[hotel] [request] stay people "
    sent = paser_aspn_to_dict(sent)
    print(sent)
    sent1 = "[hotel] people 2 stay 3"
    sent2 = "[hotel] people 1 [restaurant] people 2"
    differ_dict = compare_dict(paser_bs_to_dict(sent1),  paser_bs_to_dict(sent2))
    print(differ_dict)
    sent = "[hotel] neighbor north star 1"
    sent = paser_bs_to_dict(sent)
    print(sent)


