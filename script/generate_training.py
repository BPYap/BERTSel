import argparse
import csv
import random
from collections import defaultdict

random.seed(42)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_tsv", type=str)
    arg_parser.add_argument("--num_negatives", type=int)
    arg_parser.add_argument("--output_tsv", type=str)
    args = arg_parser.parse_args()

    input_file = open(args.input_tsv, 'r', encoding='utf-8', newline='')
    output_file = open(args.output_tsv, 'w', encoding='utf-8', newline='')

    reader = csv.reader(input_file, delimiter='\t')
    writer = csv.writer(output_file, delimiter='\t')

    positives = []
    negatives = defaultdict(list)
    answers = set()
    for question, answer, label in reader:
        if label == "1":
            positives.append((question, answer))
        else:
            negatives[question].append(answer)
        answers.add(answer)

    answers = list(answers)
    for question, positive in positives:
        counter = 0
        while counter < args.num_negatives:
            if len(negatives[question]) > 0:
                negative = negatives[question].pop()
            else:
                negative = random.choice(answers)

            if negative != positive:
                writer.writerow([question, positive, negative])
                counter += 1

    input_file.close()
    output_file.close()
