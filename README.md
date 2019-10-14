# BERTSel

##### Table of Contents  
[Introduction](#introduction)  
[Installation](#installation)  
[Basic Usage](#basic-usage) 

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

## Basic Usage
### Train
```
python script/run_dataset.py --task_name BERTSel --do_train --do_lower_case \
 --model_type bert --model_name_or_path bert-base-uncased --max_seq_length 512 \
 [--learning_rate LEARNING_RATE] [--num_train_epochs NUM_TRAIN_EPOCHS] \ 
 [--train_tsv TRAIN_TSV] \ 
 [--output_dir OUTPUT_DIR]

Arguments to note:
  TRAIN_TSV - Patb to training data in .tsv format. Each line should have three items (question, answer, label) separated by tab.
  
  OUTPUT_DIR - Path to model directory.
```

### Inference
```
python script/run_inference.py --task_name BERTSel --do_lower_case --batch_size 8 \ 
 --max_seq_length 512 --model_type bert \ 
 [--model_name_or_path MODEL_DIR] \
 [--test_tsv TEST_TSV] [--answer_pool ANSWER_POOL] \ 
 [--output_path OUTPUT_PATH]
 
Arguments to note:
  MODEL_DIR - Path to model directory.
  
  TEST_TSV - Filename of the testing data in .tsv format. Each line should have two items (question, indices) separated by tab.
  "indices" are list of indices (comma-separated) of the possible answers in the answer_pool.
  
  ANSWER_POOL - Path to the .txt file containing list of answer candidates separated by newline.
  
  OUTPUT_PATH - Path to output file in json format. Entries in the json object corresponds to rank results 
  (highest to lowest) of each question.
```

### Generate Training Examples
```
python script/generate_training.py [--input_tsv INPUT_TSV] [--num_negatives NUM_NEGATIVES] [--output_tsv OUTPUT_TSV]
 
Arguments:
  INPUT_TSV - Path to the training data in .tsv format. Each line should have three items: (question, answer, label) separated by tab.
  
  NUM_NEGATIVES - Number of training pairs to generate for each positive example.
  
  OUTPUT_PATH - Path to the output data in .tsv format. Each line in the output .tsv contains three items: (question, positive_answer, negative_answer) separated by tab.
```

## References
- Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
- Li, Dongfang, et al. "BERTSel: Answer Selection with Pre-trained Models." arXiv preprint arXiv:1905.07588 (2019).
