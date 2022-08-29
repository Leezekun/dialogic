cd ./ubar-preprocessing/data
unzip multi-woz.zip
cd multi-woz
unzip data.json.zip
cd ..
unzip MultiWOZ_2.1.zip
cd ..
python data_analysis21.py
python preprocess21.py
cd ..
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.1
cp -r ./ubar-preprocessing/data ./
cp -r ./ubar-preprocessing/db ./data/
cd ./utlis
python postprocessing_dataset21.py
cd ..
cp special_token_list.txt ./data/multi-woz-2.1-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-2.1-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-2.1-fine-processed/possible_slot_values.json