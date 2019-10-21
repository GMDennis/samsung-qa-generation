import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import DiscreteVAE, return_mask_lengths


class VAETrainer(object):

    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args).to(self.device)
        self.params = filter(lambda p: p.requires_grad, self.vae.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr)

        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_kl = 0
        self.loss_info = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae = self.vae.train()

        # Forward
        loss, loss_q_rec, loss_a_rec, \
        loss_kl, loss_info = \
        self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Step
        clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()

        self.loss_q_rec = loss_q_rec.item()
        self.loss_a_rec = loss_a_rec.item()
        self.loss_kl = loss_kl.item()
        self.loss_info = loss_info.item()

    def generate_posterior(self, c_ids, q_ids, a_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            posterior_z_prob, posterior_z = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            q_ids, start_positions, end_positions = self.vae.generate(posterior_z, c_ids)
        return q_ids, start_positions, end_positions, posterior_z_prob

    def generate_prior(self, c_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            prior_z_prob, prior_z = self.vae.prior_encoder(c_ids)
            q_ids, start_positions, end_positions = self.vae.generate(prior_z, c_ids)
        return q_ids, start_positions, end_positions, prior_z_prob

    def save(self, filename):
        params = {
            'state_dict': self.vae.state_dict(),
            'args': self.args
        }
        torch.save(params, filename)

    def reduce_lr(self):
        self.optimizer.param_groups[0]['lr'] *= 0.5

    @staticmethod
    def post_process(q_ids, start_positions, end_positions, c_ids):
        batch_size = q_ids.size(0)
        # exclude CLS token in c_ids
        c_ids = c_ids[:, 1:]
        start_positions = start_positions - 1
        end_positions = end_positions - 1

        q_mask, q_lengths = return_mask_lengths(q_ids)
        c_mask, c_lengths = return_mask_lengths(c_ids)

        total_max_len = torch.max(q_lengths + c_lengths)

        all_input_ids = []
        all_seg_ids = []
        for i in range(batch_size):
            q_length = q_lengths[i]
            c_length = c_lengths[i]
            q = q_ids[i, :q_length]  # exclude pad tokens
            c = c_ids[i, :c_length]  # exclude pad tokens

            # input ids
            pads = torch.zeros((total_max_len - q_length - c_length), device=q_ids.device, dtype=torch.long)
            input_ids = torch.cat([q, c, pads], dim=0)
            all_input_ids.append(input_ids)

            # segment ids
            zeros = torch.zeros_like(q)
            ones = torch.ones_like(c)
            seg_ids = torch.cat([zeros, ones, pads], dim=0)
            all_seg_ids.append(seg_ids)

            start_positions[i] = start_positions[i] + q_length
            end_positions[i] = end_positions[i] + q_length

        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_seg_ids = torch.stack(all_seg_ids, dim=0)
        all_input_mask = (all_input_ids != 0).byte()

        return all_input_ids, all_seg_ids, all_input_mask, start_positions, end_positions

    @staticmethod
    def get_loss(start_logits, end_logits, start_positions, end_positions):
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction="none")
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) * 0.5
        return total_loss
