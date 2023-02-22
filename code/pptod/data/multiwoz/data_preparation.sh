python -m spacy download en_core_web_sm
cd ./ubar-preprocessing/data
unzip multi-woz.zip
cd multi-woz
unzip data.json.zip
cd ..
cd ..
python data_analysis.py
python preprocess.py 
cd ..
cp -r ./ubar-preprocessing/data ./
cp -r ./ubar-preprocessing/db ./data/
cd ./utlis
python postprocessing_dataset.py
cd ..
cp special_token_list.txt ./data/multi-woz-fine-processed/special_token_list.txt
cp schema.json ./data/multi-woz-fine-processed/schema.json
cp possible_slot_values.json ./data/multi-woz-fine-processed/possible_slot_values.json