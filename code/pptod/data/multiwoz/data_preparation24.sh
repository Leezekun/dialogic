cd ./ubar-preprocessing/data
unzip multi-woz.zip
cd multi-woz
unzip data.json.zip
cd ..
unzip MultiWOZ_2.4.zip
cd ..
python data_analysis24.py
python preprocess24.py
cd ..
rm -rf ./ubar-preprocessing/data/MultiWOZ_2.4
cp -r ./ubar-preprocessing/data ./
cp -r ./ubar-preprocessing/db ./data/
cd ./utlis
python postprocessing_dataset24.py
cd ..
cp special_token_list.txt ./data/multi-woz-2.4-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-2.4-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-2.4-fine-processed/possible_slot_values.json