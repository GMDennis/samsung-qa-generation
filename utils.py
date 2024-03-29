import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from squad_utils import read_squad_examples, convert_examples_to_features_answer_id


def get_squad_data_loader(tokenizer, file, shuffle, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    features = convert_examples_to_features_answer_id(examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_c_len,
                                                      max_query_length=args.max_q_len,
                                                      doc_stride=128,
                                                      is_training=True)

    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_a_ids = (all_tag_ids != 0).long()
    all_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)

    all_data = TensorDataset(all_c_ids, all_q_ids, all_a_ids, all_start_positions, all_end_positions)
    data_loader = DataLoader(all_data, args.batch_size, shuffle=shuffle)

    return data_loader, examples, features

def get_harv_data_loader(tokenizer, file, shuffle, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    num_examples = len(examples)
    random.shuffle(examples)
    examples = examples[:5000]

    features = convert_examples_to_features_answer_id(examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_c_len,
                                                      max_query_length=args.max_q_len,
                                                      doc_stride=128,
                                                      is_training=True)

    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_data = TensorDataset(all_c_ids)
    data_loader = DataLoader(all_data, args.batch_size, shuffle=shuffle)

    return data_loader

def batch_to_device(batch, device):
    batch = (b.to(device) for b in batch)
    c_ids, q_ids, a_ids, start_positions, end_positions = batch

    c_len = torch.sum(torch.sign(c_ids), 1)
    max_c_len = torch.max(c_len)
    c_ids = c_ids[:, :max_c_len]
    a_ids = a_ids[:, :max_c_len]

    q_len = torch.sum(torch.sign(q_ids), 1)
    max_q_len = torch.max(q_len)
    q_ids = q_ids[:, :max_q_len]

    return c_ids, q_ids, a_ids, start_positions, end_positions
