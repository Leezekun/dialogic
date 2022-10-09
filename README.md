# Dialogic: Controllable Dialogue Simulation with In-Context Learning
This is the pytorch implementation of **Controllable Dialogue Simulation with In-Context Learning**, which got accepted by Findings of EMNLP2022.

<p align="center"><img width="100%" src="./imgs/demo_both.gif" /></p>

## Introduction
Starting from a small seed dataset, our method dialogic can generate good-quality annotated dialogues without human labor, parameter update, or engineering efforts, which is a much more time-saving and cost-efficient alternative to crowdsourcing in dataset creation. 

We show a [demo](#demo) of how a dialogue is simulated above. You can type into your user goal or use the automatically generated one. 
We also provide simulated dialogues in the `./simulated_dialogues` directory. The description of data format can be found [here](#format-of-simulated-dialogues).

<!-- Taking the [MultiWOZ](https://github.com/budzianowski/multiwoz) for example, given any user goal, such as booking a hotel (area is center, stay is 1, people is 2, bookday is Monday), and a restaurant (area is north, pricerange is moderate), DS-ICL can generate the corresponding dialogue along with annotations. An illustration of the generation process is presented above. -->

## Table of Contents
- [Dialogic: Controllable Dialogue Simulation with In-Context Learning](#dialogic-controllable-dialogue-simulation-with-in-context-learning)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Preparation](#preparation)
    - [Environment setup](#environment-setup)
    - [Data preparation](#data-preparation)
    - [Verification model preparation](#verification-model-preparation)
  - [Simulation](#simulation)
    - [Dialogue simulation](#dialogue-simulation)
    - [Demo](#demo)
    - [Format of simulated dialogues](#format-of-simulated-dialogues)
    - [Turn-level simulation](#turn-level-simulation)
  - [Training on simulated dialogues](#training-on-simulated-dialogues)
    - [PPTOD](#pptod)
    - [SimpleTOD](#simpletod)
    - [MinTL](#mintl)

## Preparation
The code is placed in the `./code` directory. As we use PPTOD as the auxiliary model for verification in this repo, most important files `dialogic_*.py` are in the `./code/pptod/E2E_TOD` directory.

### Environment setup
Set up the environment for PPTOD and SimpleTOD. To set up the environment for MinTL, please refer to `./code/MinTL/README.md`.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Data preparation
We use [MultiWOZ_2.3](https://github.com/lexmen318/MultiWOZ-coref) dataset by default. [MultiWOZ_2.0](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.0.zip), [MultiWOZ_2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip), and [MultiWOZ_2.4](https://github.com/smartyfh/MultiWOZ2.4) datasets are also supported.
You can use the following script to prepare the data.
```bash
cd ./code/pptod/data/multiwoz
# MultiWOZ_2.3 dataset
chmod +x ./data_preparation23.sh 
./data_preparation23.sh 
# MultiWOZ_2.0 dataset
# chmod +x ./data_preparation.sh 
# ./data_preparation.sh 
# MultiWOZ_2.1 dataset
# chmod +x ./data_preparation21.sh 
# ./data_preparation21.sh 
# MultiWOZ_2.4 dataset
# chmod +x ./data_preparation24.sh 
# ./data_preparation24.sh 
```

### Verification model preparation
We use [PPTOD](https://github.com/awslabs/pptod) as the auxiliary model for verification in this codebase. To use it, you should download the initial checkpoint you want and unzip it in the `./code/pptod/checkpoints` directory. We use PPTOD-small by default.
```bash
cd ./code/pptod/checkpoints
# Downloading Initial PPTOD-small Checkpoint:
chmod +x ./download_pptod_small.sh
./download_pptod_small.sh
# Downloading Initial PPTOD-base Checkpoint:
chmod +x ./download_pptod_base.sh
./download_pptod_base.sh
#Downloading Initial PPTOD-large Checkpoint:
chmod +x ./download_pptod_large.sh
./download_pptod_large.sh
```

Then you can use the script to train the verification model on the given seed dataset (1% few-shot setting by default):
```bash
cd ./code/pptod/E2E_TOD/sh_folder/small/training
chmod +x pptod_small_training_few_shot_0.01.sh
./pptod_small_training_few_shot_0.01.sh
```
Some important options include:
  - `--train_data_ratio`: the ratio of training data we use, i.e., the few-shot setting (1% by default).
  - `--ckpt_save_path`: the path where the trained verification model is saved.
<!-- The trained verifier is saved in `./pptod/E2E_TOD/ckpt23/small/few_shot_0.01/` directory. You can try other few-shot settings by changing `0.01` to any number in (0, 1]. -->
> We provide the checkpoints of verification models trained on 1%/5%/10% of the training data, which are placed at `./code/pptod/E2E_TOD/ckpt23/small/`.

## Simulation
Put your OpenAI API key in `./code/pptod/E2E_TOD/dialogic_utils.py` to use GPT-3!

### Dialogue simulation
First, we extract the turn-level annotations. 
```bash
cd ./code/pptod/E2E_TOD/
# process the dialogues to get turn-level annotations
python dialogic_pre_process.py\
 --train_data_ratio 0.01
```

Then we generate the user goals, select in-context examples and construct the prompts.
```bash
cd ./code/pptod/E2E_TOD/
python dialogic_aug_e2e.py\
  --train_data_ratio 0.01\
  --augment_type combine\
  --augment_time 1\
  --k_shot 2\
  --temperature 0.2
```
Some important options include:
  - `--train_data_ratio`: the ratio of training data we use, the few-shot setting.
  - `--augment_type`: how to generate the user goals, options: [combine substitution, random].
  - `--augment_time`: how many times of the seed dataset we are going to augment.
  - `--k_shot`: how many in-context examples used in the prompt.
  - `--temperature`: the temperature when using the combine method. Lower temperature results in less random example selection.

Finally, you can use the following script to start simulating the dialogues:
```bash
cd ./code/pptod/E2E_TOD/sh_folder/small/simulation/
chmod +x ./pptod_small_few_shot_0.01_simulation.sh
./pptod_small_few_shot_0.01_simulation.sh
```
Some important options include:
  - `--train_data_ratio`: the ratio of training data we use, the few-shot setting.
  - `--pretrained_path`: the path of trained auxiliary verification model.
  - `--output_save_path`: the path to store the simulated dialogues.
  - `--augment_dialog_path`: the path where the constructed prompts store. 
  - `--max_turn_num`: the maximum number of turns in each simulated dialog.
  - `--max_dialog_num`: the maximum number of dialogs simulated using gpt-3.
  - `--max_aug_num`: the size of simulated data w.r.t. the original seed data.
  - `--verify_bs`: whether to use the auxiliary verification model to correct the generated belief state annotations.
  - `--verify_da`: whether to use the auxiliary verification model to correct the generated dialog act annotations.
  - `--debug`: whether to print out the simulation process.
  - `--save`: whether to save the simulated dialogues.
  
You will see the dialogue simulation process as:
```console
--------------------------------------------- Generation process of turn 0 ---------------------------------------------
GPT-3 generated user turn   >> You require([hotel] pricerange is cheap , day is tuesday , people is 5): hi there , i ' m looking for a hotel that is cheap and can accommodate 5 people on tuesday .
Verifier generated bs       >> [hotel] pricerange cheap and accommodate 5 day tuesday
Corrected bs                >> [hotel] day tuesday pricerange cheap people 5
Corrected user turn         >> You require([hotel] day is tuesday , pricerange is cheap , people is 5): hi there , i ' m looking for a hotel that is cheap and can accommodate 5 people on tuesday .
DB query result             >> [db_3]
Verifier generated da       >> [hotel] [inform] choice [request] area
Corrected system turn       >> Assistant([hotel] [inform] choice [request] area): i have [value_choice] options that are in the [value_area] .
------------------------------------------------ Conversation of turn 0 ------------------------------------------------
User                        >> hi there , i ' m looking for a hotel that is cheap and can accommodate 5 people on tuesday .
System                      >> i have [value_choice] options that are in the [value_area] .
------------------------------------------------------------------------------------------------------------------------

--------------------------------------------- Generation process of turn 1 ---------------------------------------------
GPT-3 generated user turn   >> You require([hotel] area is east): perfect , can you give me more information about the hotels in the east ?
Verifier generated bs       >> [hotel] pricerange cheap and accommodate 5 day tuesday
Corrected bs                >> [hotel] day tuesday pricerange cheap people 5 area east
Corrected user turn         >> You require([hotel] area is east): perfect , can you give me more information about the hotels in the east ?
DB query result             >> [db_2]
Verifier generated da       >> [hotel] [inform] area name internet parking
Corrected system turn       >> Assistant([hotel] [inform] area name internet parking): there is the [value_name] , which has free wifi and parking .
------------------------------------------------ Conversation of turn 1 ------------------------------------------------
User                        >> perfect , can you give me more information about the hotels in the east ?
System                      >> there is the [value_name] , which has free wifi and parking .
------------------------------------------------------------------------------------------------------------------------
......
```

### Demo
We also provide a demo to demonstrate how to simulate a dialogue turn by turn given a user goal. You can type into any user goal or use an automatically generated one to see how the corresponding dialogue is generated.
```bash
cd ./code/pptod/E2E_TOD/sh_folder/small/demo
chmod +x ./pptod_small_few_shot_0.01_demo.sh
./pptod_small_few_shot_0.01_demo.sh
```
An illustration of the demo example can be seen [here](#dialogic-controllable-dialogue-simulation-with-in-context-learning).


### Format of simulated dialogues
The simulated dialogues are saved in json format. For each dialogue, we save the following information:
  - **dial_id**: the id of the simulated dialogue, which consists of the ids of the used example dialogues. For example, `comb_pmul3021_sng0548` is simulated with the examples of `pmul3021` and `sng0548`.
  - **goal**: the user goal of this dialogue. 
  - **turns**: a list of turns in this dialogue, where each turn is represented as a dictionary that contains the following fields:
    - **dial_id** - the unique ID for the dialogue session instance.
    - **turn_num** - this argument indicates the turn position in the dialogue session.
    - **user** - the user's utterance.
    - **resp** - the delexicalized reference system response.
    - **bspn** - the belief state.
    - **aspn** - the system action.
    - **db** - The database query result.
  - **prompt**: the prompt used to instruct GPT-3 to simulate the dialogue.

> We provide the simulated dialogues in `./simulated_dialogues/` (w/o prompt for simplicity) and `./code/pptod/E2E_TOD/simulation_result23/small/` (w/ prompt) directory.


### Turn-level simulation
You can use the following script to start simulating dialogue turns for DST augmentation.
```bash
cd ./pptod/E2E_TOD/
python dialogic_aug_dst.py\
  --train_data_ratio 0.01\
  --augment_time 2\
  --k_shot 2\
  --temperature 0.2
```

## Training on simulated dialogues 
Convert the format of simulated dialogues for E2E training.
```bash
cd ./code/pptod/E2E_TOD/
python dialogic_post_process.py\
  --data_type E2E\
  --raw_data_path ./simulation_result23/small/few_shot_0.01/combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.01_simulation_result.json
```
Convert the format of simulated dialogue turns for DST training.
```bash
cd ./code/pptod/E2E_TOD/
python dialogic_post_process.py\
  --data_type DST\
  --raw_data_path ../data/multiwoz/data/multi-woz-2.3-dialogic-processed/2_shot_augment_x2_dst_turn_info_train_ratio_0.01.json
```

### PPTOD
You can use the following scripts to train PPTOD:
```bash
# E2E
cd ./code/pptod/E2E_TOD/sh_folder/small/training/
chmod +x ./pptod_small_train_few_shot_0.01_augx1.sh
./pptod_small_train_few_shot_0.01_augx1.sh
# DST
cd ./code/pptod/DST/sh_folder/small/training/
chmod +x ./pptod_small_train_few_shot_0.01.sh
./pptod_small_train_few_shot_0.01.sh
```

### SimpleTOD
Convert the format of simulated dialogues to fit SimpleTOD.
```bash
# E2E
cd ./code/pptod/E2E_TOD/
python dialogic_export_dialog_e2e.py\
  --train_data_ratio 0.01\
  --aug_train_data_file multi-woz-fine-processed-train-combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.01_simulation_result.json\
  --save_data_path_prefix ../../simpletod/resources_e2e_2.3_0.01_augx1/multi-woz
# DST
cd ./code/pptod/DST/
python dialogic_export_dialog_dst.py\
  --train_data_ratio 0.01\
  --aug_train_data_file multi-woz-fine-processed-train-2_shot_augment_x2_dst_turn_info_train_ratio_0.01.json\
  --save_data_path_prefix ../../simpletod/resources_DST_2.3_0.01_augx2/multi-woz
```

Then you can use the simulated dialogue to train SimpleTOD:
```bash
cd ./code/simpletod/
# create data
chmod +x create_dataset.sh
./create_dataset.sh
# E2E training
./train_end2end.sh 7 gpt2 gpt2 1
# DST training
./train_dst.sh 7 gpt2 gpt2 1
```
Use the following command for inference on test set:
```bash
CUDA_VISIBLE_DEVICES=$GPU python generate_dialogue_aug.py $CHECKPOINT $DECODING
```
Use the following command for evaluation:
```bash
python evaluate_multiwoz_aug.py $MODEL_OUTPUT $DATA_DIR
```

### MinTL
You should use another environment for experiments on MinTL.
```bash
cd ./code/MinTL
pip install -r requirements.txt
```
Convert the format of simulated dialogues to fit MinTL.
```bash
# E2E
cd ./code/pptod/E2E_TOD/
python dialogic_export_dialog_e2e.py\
  --train_data_ratio 0.01\
  --aug_train_data_file multi-woz-fine-processed-train-combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.01_simulation_result.json\
  --save_data_path_prefix ../../MinTL/generated_data/e2e_2.3_0.01_augx1/
# DST
cd ./code/pptod/DST/
python dialogic_export_dialog_dst.py\
  --train_data_ratio 0.01\
  --aug_train_data_file multi-woz-fine-processed-train-2_shot_augment_x2_dst_turn_info_train_ratio_0.01.json\
  --save_data_path_prefix ../../MinTL/generated_data/dst_2.3_0.01_augx2/
```

Then you can use the simulated dialogue to train MinTL:
```bash
export PYTHONPATH='$PROJECT_PATH/code/MinTL/damd_multiwoz'
# E2E training
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --context_window 2 --pretrained_checkpoint t5-small --cfg seed=557 batch_size=32 --use_db True --generated_data_file e2e_2.3_0.01_augx1
# DST training
CUDA_VISIBLE_DEVICES=1 python DST.py --mode train --context_window 3 --cfg seed=557 batch_size=32 --generated_data_file dst_2.3_0.01_augx2
```














