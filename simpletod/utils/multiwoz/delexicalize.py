import re
import os
import simplejson as json

from .nlp import normalize, normalize_mine

import ipdb

digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")

# FORMAT
# domain_value
# restaurant_postcode
# restaurant_address
# taxi_car8
# taxi_number
# train_id etc..

# PATH = './utils/multiwoz'
# ROOT = '/home/ehosseiniasl/github_repos/ehsan-scratchpad/dialogue'
PATH = './resources/multi-woz/MULTIWOZ2.1'

def prepareSlotValuesIndependent():
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    # read databases
    for domain in domains:
        try:
            # fin = file(os.path.join(PATH, 'db/' + domain + '_db.json'))
            fin = open(os.path.join(PATH, domain + '_db.json'), 'r')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append((normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append((normalize(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append((normalize(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    elif key == 'food':
                        dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    elif key == 'pricerange':
                        dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    else:
                        pass
                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.append((normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append((normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    # fin = open(os.path.join(PATH, 'db/' + 'train' + '_db.json'))
    fin = open(os.path.join(PATH, 'train' + '_db.json'))
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == 'departure' or key == 'destination':
                dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # add specific values:
    for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def prepareSlotValuesIndependent_mine():
    '''just delex entity names, not price, food or area'''
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    normalize_func = normalize_mine
    # read databases
    for domain in domains:
        try:
            # fin = file(os.path.join(PATH, 'db/' + domain + '_db.json'))
            fin = open(os.path.join(PATH, domain + '_db.json'), 'r')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    # elif key == 'area':
                    #     dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    # elif key == 'food':
                    #     dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    # elif key == 'pricerange':
                    #     dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    # else:
                    #     pass
                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.append((normalize_func('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize_func('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize_func('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize_func('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append((normalize_func('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize_func('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize_func('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    # fin = open(os.path.join(PATH, 'db/' + 'train' + '_db.json'))
    # db_json = json.load(fin)
    # fin.close()
    #
    # for ent in db_json:
    #     for key, val in ent.items():
    #         if key == 'departure' or key == 'destination':
    #             dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))
    #
    # # add specific values:
    # for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
    #     dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key == domain or key == 'value':
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?
    return utt

if __name__ == '__main__':
    prepareSlotValuesIndependent()
