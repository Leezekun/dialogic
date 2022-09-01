
# # preprocess multiwoz with delexicalized responses
# python preprocess_multiwoz.py delex

# # preprocess multiwoz with lexicalized responses
# python preprocess_multiwoz.py lexical

# # create dataset for language modeling with SimpleTOD
# python prepare_simpletod_data.py


# preprocess multiwoz with delexicalized responses
python preprocess_multiwoz_aug.py delex

# preprocess multiwoz with lexicalized responses
python preprocess_multiwoz_aug.py lexical

# create dataset for language modeling with SimpleTOD
python prepare_simpletod_data_aug.py