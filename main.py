import argparse
import os
import random
import socket
import numpy as np
import torch
from transformers import BertTokenizer
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from eval import eval_vae
from trainer import VAETrainer
from utils import get_squad_data_loader, get_harv_data_loader, \
                  batch_to_device

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_loader, _, _ = get_squad_data_loader(tokenizer, args.train_dir,
                                         shuffle=True, args=args)
    eval_data = get_squad_data_loader(tokenizer, args.dev_dir,
                                      shuffle=False, args=args)

    args.device = torch.cuda.current_device()

    trainer = VAETrainer(args)

    log_dir = os.path.join(args.model_dir, socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    loss_log = tqdm(total=0, bar_format='{desc}', position=2)
    eval_log = tqdm(total=0, bar_format='{desc}', position=4)
    best_eval_log = tqdm(total=0, bar_format='{desc}', position=5)

    print("MODEL DIR: " + args.model_dir)

    stack = 0
    niter = 0
    best_avg_qa_loss, best_bleu, best_em, best_f1 = 1000.0, 0.0, 0.0, 0.0
    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        #train_iterator = train_loader
        for batch in tqdm(train_loader, desc="Train iter", leave=False, position=1):
            c_ids, q_ids, a_ids, start_positions, end_positions \
            = batch_to_device(batch, args.device)
            trainer.train(c_ids, q_ids, a_ids, start_positions, end_positions)
            niter += 1
            writer.add_scalars('data/loss_group',
                               {'loss_q_rec': trainer.loss_q_rec,
                                'loss_a_rec': trainer.loss_a_rec,
                                'loss_kl': trainer.loss_kl,
                                'loss_info': trainer.loss_info},
                               niter)
            str = 'Q REC : {:06.4f} A REC : {:06.4f} KL : {:06.4f} INFO : {:06.4f}'
            str = str.format(float(trainer.loss_q_rec), float(trainer.loss_a_rec),
                             float(trainer.loss_kl), float(trainer.loss_info))
            loss_log.set_description_str(str)

        metric_dict, bleu, all_results \
        = eval_vae(epoch, args, trainer, eval_data)
        f1 = metric_dict["f1"]
        em = metric_dict["exact_match"]
        bleu = bleu * 100
        str = '{}-th Epochs BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
        str = str.format(epoch, bleu, em, f1)
        eval_log.set_description_str(str)
        writer.add_scalars('data/performance',
                           {'bleu': bleu, 'em': em, 'f1': f1}, epoch)
        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
            trainer.save(os.path.join(args.model_dir, "best_f1_model.pt"))
        if bleu > best_bleu:
            best_bleu = bleu
            trainer.save(os.path.join(args.model_dir, "best_bleu_model.pt"))

        str = 'BEST BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
        str = str.format(best_bleu, best_em, best_f1)
        best_eval_log.set_description_str(str)

        mat = []
        metadata = []
        for j in range(len(all_results)):
            mat.append(all_results[j].posterior_z_prob.view(-1))
            str = "[{}] [Pos] Real Q: {} Real A: {} Pos Q: {} Pos A: {}"
            str = str.format(j, all_results[j].real_question,
                                all_results[j].real_answer,
                                all_results[j].posterior_question,
                                all_results[j].posterior_answer)
            metadata.append(str)

            mat.append(all_results[j].prior_z_prob.view(-1))
            str = "[{}] [Pri] Pri Q: {} Pri A: {}"
            str = str.format(j, all_results[j].prior_question,
                                all_results[j].prior_answer)
            metadata.append(str)
        mat = torch.stack(mat, dim=0)
        writer.add_embedding(mat=mat, metadata=metadata, global_step=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_dir', default='../data/KorQuAD_v1.0_train.json')
    parser.add_argument('--dev_dir', default='../data/KorQuAD_v1.0_dev.json')
    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")

    parser.add_argument("--model_dir", default="../ex3/korean-vae", type=str)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--batch_size", default=16, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--bert_model", default='bert-base-multilingual-cased', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=600)
    parser.add_argument('--enc_nlayers', type=int, default=2)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.2)
    parser.add_argument('--dec_q_nhidden', type=int, default=600)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nz', type=int, default=20)
    parser.add_argument('--nzdim', type=int, default=10)

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"
    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
