# BERTSel

##### Table of Contents  
[Introduction](#introduction)  
[Installation](#installation)  
[Usage](#usage) 

## Introduction
This repository contains reference implementation for [BERTSel: Answer Selection with Pre-trained Models](https://arxiv.org/abs/1905.07588) using [Transformers](https://github.com/huggingface/transformers) from Hugging Face. 

## Installation
### Step 1: Clone the repository
```
git clone https://github.com/BPYap/BERTSel
cd BERTSel
```
### Step 2: Install dependencies
```
python3 -m virtualenv env
source env/bin/activate

pip install -r requirements.txt
```

## Usage
### Train
```
python script/run_dataset.py --task_name BERTSel --do_train --do_eval --do_lower_case \
 --model_type bert --model_name_or_path bert-base-uncased --max_seq_length 512 \
 [--learning_rate LEARNING_RATE] [--num_train_epochs NUM_TRAIN_EPOCHS] \ 
 [--data_dir DATA_DIR] \
 [--negative_samples NEGATIVE_SAMPLES] \ 
 [--output_dir OUTPUT_DIR]

Arguments to note:
  DATA_DIR - Path to data directory. The directory should contain the following files:
  `train_questions.txt`, `train_answers.txt`, `dev_questions.txt` and `dev_answers.txt`
  
  NEGATIVE_SAMPLES - Number of negative sample pairs
  
  OUTPUT_DIR - Path to model directory
```

### Inference
```
python script/run_inference.py --task_name BERTSel --do_lower_case --batch_size 8 \ 
 --max_seq_length 512 --model_type bert \ 
 [--model_name_or_path MODEL_DIR] \
 [--tests TESTS] [--answers_pool ANSWERS_POOL] \ 
 [--output_path OUTPUT_PATH]
 
Arguments to note:
  MODEL_DIR - Path to model directory
  
  TESTS - Path to text file containing test questions. Each question is separated by newline.
  
  ANSWERS_POOL - Path to text file containing list of answers to be compared against each test question. 
  Each answer is separated by newline.
  
  OUTPUT_PATH - Path to output file in json format. Entries in the json object corresponds to rank results 
  (highest to lowest) of each question.
```

## References
- Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
- Li, Dongfang, et al. "BERTSel: Answer Selection with Pre-trained Models." arXiv preprint arXiv:1905.07588 (2019).
