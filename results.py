import pickle
import numpy as np
import torch
import argparse
import os
import logging
import sys


def load_data(data_dir):
    with open(data_dir, 'rb') as f:
        results = pickle.load(f)
    return results


class TextResult(object):

    def __init__(self,
                 context,
                 real_question,
                 posterior_question,
                 prior_question,
                 real_answer,
                 posterior_answer,
                 prior_answer):
        self.context = context
        self.real_question = real_question
        self.posterior_question = posterior_question
        self.prior_question = prior_question
        self.real_answer = real_answer
        self.posterior_answer = posterior_answer
        self.prior_answer = prior_answer


def main():
    args = setup()
    results = load_data(args.results_file)

    for index in args.indexes:
        result = results[index]
        print("Context :")
        print(result.context + "\n")

        print("real Question :")
        print(result.real_question + "\n")
        print("real Answer :")
        print(result.real_answer + "\n")

        print("posterior Question :")
        print(result.posterior_question + "\n")
        print("posterior Answer :")
        print(result.posterior_answer + "\n")

        print("prior Question :")
        print(result.prior_question + "\n")
        print("prior Answer :\n")
        print(result.prior_answer + "\n")


def setup():
    parser = argparse.ArgumentParser(
        description='Train QA model.'
    )
    # system

    parser.add_argument('--results_file', default="../ex/vae_reverse_true_feed/results.pkl",
                        help='path to saved results.')
    parser.add_argument('--indexes', default=[22],
                        help='path to saved results.')
    parser.add_argument('--what_to_gen', default='qa',
                        help='qa')

    # model
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
