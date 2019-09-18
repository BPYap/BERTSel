# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging

import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm

from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from utils_dataset import (convert_examples_to_features,
                           output_modes, processors, InputExample)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
                 ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


def inference(args, model, tokenizer, prefix=""):
    inf_task = args.task_name
    inf_dataset = load_example(args, inf_task, tokenizer)
    inf_sampler = SequentialSampler(inf_dataset)
    inf_dataloader = DataLoader(inf_dataset, sampler=inf_sampler, batch_size=1)

    # Inference!
    logger.info("***** Running inference {} *****".format(prefix))

    with open(args.answers_file, 'r', encoding='utf-8') as f:
        answers = [line for line in f]
    possible_answer_indices = []
    for i, batch in enumerate(tqdm(inf_dataloader, desc="Inferencing")):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            inf_loss, logits = outputs[:2]

            pred_arr = logits.detach().cpu().numpy()

            # logger.info("pred_arr: %s", pred_arr)
            pred_prob = np.squeeze(softmax(pred_arr, axis=1))
            # logger.info("[0]: %s, [1]: %s", pred_prob[0], pred_prob[1])

            if args.output_mode == "classification":
                pred = np.argmax(pred_arr, axis=1)
            elif args.output_mode == "regression":
                pred = np.squeeze(pred_arr)

            # if pred == 0:
            #     logger.info("Text is negative with confidence: %d ", pred_prob[0] * 100)
            # else:
            #     logger.info("Text is positive with confidence: %d ", pred_prob[1] * 100)

            if pred == 1:
                possible_answer_indices.append((pred_prob[1] * 100, i))

    print(f"\n{len(possible_answer_indices)} possible answers for '{args.text}':")
    for prob, i in sorted(possible_answer_indices, key=lambda x: x[0], reverse=True):
        print(f"[{round(prob, 2)}%] {answers[i]}")


def load_example(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]

    logger.info("Creating features from input")
    label_list = processor.get_labels()
    examples = []
    with open(args.answers_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            examples.append(InputExample(guid=i, text_a=args.text, text_b=line, label="1"))

    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                            cls_token_at_end=bool(args.model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                            pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--text", default="None", type=str, required=True,
                        help="text to analyze")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--answers_file", default=None, type=str, required=True,
                        help="Path to text file with answers separated by newline")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, ", args.device)

    # Prepare task
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.to(args.device)

    logger.info("Inference parameters %s", args)

    # Inference
    inference(args, model, tokenizer)


if __name__ == "__main__":
    main()
