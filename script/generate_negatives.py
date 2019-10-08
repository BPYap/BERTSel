import argparse
import csv
import random

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

    question_answers = dict()
    for question, answer, label in reader:
        if label == "1":
            question_answers[question] = answer
        else:
            writer.writerow([question, answer, label])

    answers = list(question_answers.values())
    for question, answer in question_answers.items():
        counter = 0
        while counter < args.num_negatives:
            negative = random.choice(answers)
            if negative != answer:
                writer.writerow([question, answer, 1])
                writer.writerow([question, negative, 0])
                counter += 1

    input_file.close()
    output_file.close()
