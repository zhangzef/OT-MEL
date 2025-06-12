import json
import os
import time
import math
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from codes.model.modeling_ot import Encoder, Matcher
from datetime import datetime
import shutil
from utils.functions import triplet_loss
import pdb


class LightningForOT(pl.LightningModule):
    def __init__(self, args):
        super(LightningForOT, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs = []

        self.path = (
            f"/home/zhangzefeng/Code/MIMIC-master/mel_/{datetime.now()}_{self.args.run}"
        )
        self.record_name = f"{datetime.now()}_{self.args.run}.json"

        self.encoder = Encoder(args)
        self.matcher = Matcher(args)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.error_record_list = []

    def training_step(self, batch):
        ent_batch = {}
        ent_triplet_batch = {}
        mention_batch = {}
        self.encoder.train()
        self.matcher.train()
        for k, v in batch.items():
            if k.startswith("ent_triplet_"):
                ent_triplet_batch[k.replace("ent_triplet_", "")] = v
            elif k.startswith("ent_"):
                ent_batch[k.replace("ent_", "")] = v
            else:
                mention_batch[k] = v
        entity_empty_image_flag = ent_batch.pop("empty_img_flag")  # not use

        # [bs, dim]
        (
            mention_text_seq_tokens,
            mention_image_patch_tokens,
            mention_text_cls,
            mention_image_cls,
        ) = self.encoder(**mention_batch)
        (
            entity_text_seq_tokens,
            entity_image_patch_tokens,
            entity_text_cls,
            entity_image_cls,
        ) = self.encoder(**ent_batch)
        (
            logits,
            (text_matching_score, image_matching_score, multimodal_matching_score),
        ) = self.matcher(
            entity_text_seq_tokens,
            mention_text_seq_tokens,
            entity_image_patch_tokens,
            mention_image_patch_tokens,
            entity_text_cls,
            mention_text_cls,
            entity_image_cls,
            mention_image_cls,
        )
        labels = (
            torch.arange(len(mention_text_seq_tokens))
            .long()
            .to(mention_text_seq_tokens.device)
        )

        text_loss = self.loss_fct(text_matching_score, labels) + triplet_loss(
            text_matching_score
        )
        image_loss = self.loss_fct(image_matching_score, labels) + triplet_loss(
            image_matching_score
        )
        multimodal_loss = self.loss_fct(
            multimodal_matching_score, labels
        ) + triplet_loss(multimodal_matching_score)
        overall_loss = self.loss_fct(logits, labels) + triplet_loss(logits)

        loss = overall_loss + text_loss + image_loss + multimodal_loss
        self.log("Train/loss", loss.detach().cpu().item(), on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        answer = batch.pop("answer")
        answer_qid = batch.pop("answer_qid")
        id_list = batch.pop("id")
        self.matcher.eval()
        batch_size = len(answer)
        (
            mention_text_seq_tokens,
            mention_image_patch_tokens,
            mention_text_cls,
            mention_image_cls,
        ) = self.encoder(
            **batch
        )  # [bs, dim]

        # We use chuck/mini-batch to alleviate GRAM usage
        scores = []
        with torch.no_grad():
            for idx in range(
                math.ceil(
                    self.args.data.num_entity / self.args.data.embed_update_batch_size
                )
            ):
                ent_inputs = torch.load(os.path.join(self.path, f"{idx}.pth"))
                chunk_entity_text_seq_tokens = ent_inputs["entity_text_seq_tokens"].to(
                    mention_text_seq_tokens.device
                )
                chunk_entity_image_patch_tokens = ent_inputs[
                    "entity_image_patch_tokens"
                ].to(mention_text_seq_tokens.device)
                chunk_entity_text_cls = ent_inputs["entity_text_cls"].to(
                    mention_text_seq_tokens.device
                )
                chunk_entity_image_cls = ent_inputs["entity_image_cls"].to(
                    mention_text_seq_tokens.device
                )

                chunk_score, _ = self.matcher(
                    chunk_entity_text_seq_tokens,
                    mention_text_seq_tokens,
                    chunk_entity_image_patch_tokens,
                    mention_image_patch_tokens,
                    chunk_entity_text_cls,
                    mention_text_cls,
                    chunk_entity_image_cls,
                    mention_image_cls,
                )
                scores.append(chunk_score)

        scores = torch.concat(scores, dim=-1)
        rank = (
            torch.argsort(
                torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False
            )
            + 1
        )
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()

        self.validation_step_outputs.append(
            {"rank": tgt_rank, "all_rank": rank.detach().cpu().numpy()}
        )

    def on_validation_start(self):
        # Update entity embedding before validation starts
        # Note that we use entity_dataloader defined in our datamodule (please see codes/utils/dataset)
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        os.makedirs(self.path)
        self.matcher.eval()

        with torch.no_grad():
            for idx, batch in enumerate(
                tqdm(
                    entity_dataloader, desc="UpdateEmbed", total=len(entity_dataloader)
                )
            ):
                batch = pl.utilities.move_data_to_device(batch, self.device)
                (
                    entity_text_seq_tokens,
                    entity_image_patch_tokens,
                    entity_text_cls,
                    entity_image_cls,
                ) = self.encoder(**batch)

                ent2save = {
                    "entity_text_seq_tokens": entity_text_seq_tokens,
                    "entity_image_patch_tokens": entity_image_patch_tokens,
                    "entity_text_cls": entity_text_cls,
                    "entity_image_cls": entity_image_cls,
                }
                torch.save(ent2save, os.path.join(self.path, f"{idx}.pth"))

    def on_validation_epoch_end(self):
        shutil.rmtree(self.path)
        torch.cuda.empty_cache()

        ranks = np.concatenate(
            [output["rank"] for output in self.validation_step_outputs]
        )
        self.validation_step_outputs = []
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Val/hits20", hits20)
        self.log("Val/hits10", hits10)
        self.log("Val/hits5", hits5)
        self.log("Val/hits3", hits3)
        self.log("Val/hits1", hits1)
        self.log("Val/mr", ranks.mean())
        self.log("Val/mrr", (1.0 / ranks).mean())

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        answer = batch.pop("answer")
        answer_qid = batch.pop("answer_qid")
        id_list = batch.pop("id")
        self.matcher.eval()
        batch_size = len(answer)
        (
            mention_text_seq_tokens,
            mention_image_patch_tokens,
            mention_text_cls,
            mention_image_cls,
        ) = self.encoder(
            **batch
        )  # [bs, dim]

        # We use chuck/mini-batch to alleviate GRAM usage
        scores = []
        with torch.no_grad():
            for idx in range(
                math.ceil(
                    self.args.data.num_entity / self.args.data.embed_update_batch_size
                )
            ):
                ent_inputs = torch.load(os.path.join(self.path, f"{idx}.pth"))
                chunk_entity_text_seq_tokens = ent_inputs["entity_text_seq_tokens"].to(
                    mention_text_seq_tokens.device
                )
                chunk_entity_image_patch_tokens = ent_inputs[
                    "entity_image_patch_tokens"
                ].to(mention_text_seq_tokens.device)
                chunk_entity_text_cls = ent_inputs["entity_text_cls"].to(
                    mention_text_seq_tokens.device
                )
                chunk_entity_image_cls = ent_inputs["entity_image_cls"].to(
                    mention_text_seq_tokens.device
                )

                chunk_score, _ = self.matcher(
                    chunk_entity_text_seq_tokens,
                    mention_text_seq_tokens,
                    chunk_entity_image_patch_tokens,
                    mention_image_patch_tokens,
                    chunk_entity_text_cls,
                    mention_text_cls,
                    chunk_entity_image_cls,
                    mention_image_cls,
                )
                scores.append(chunk_score)

        scores = torch.concat(scores, dim=-1)
        rank = (
            torch.argsort(
                torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False
            )
            + 1
        )
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()

        rank2record = rank.tolist()
        tgt_rank2record = tgt_rank.tolist()
        for idx in range(len(tgt_rank2record)):
            if tgt_rank2record[idx] > 1:
                self.error_record_list.append(
                    {
                        "id": id_list[idx],
                        "answer_qid": answer_qid[idx],
                        "rank": tgt_rank2record[idx],
                        "rank_result": rank2record[idx][:20],
                    }
                )

        self.test_step_outputs.append(
            {
                "rank": tgt_rank,
                "all_rank": rank.detach().cpu().numpy(),
                "scores": scores.detach().cpu().numpy(),
            }
        )
        # return dict(rank=tgt_rank, all_rank=rank.detach().cpu().numpy(), scores=scores.detach().cpu().numpy())

    def on_test_start(self):
        # Update entity embedding before test starts
        # Note that we use entity_dataloader defined in our datamodule (please see codes/utils/dataset)
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        os.mkdir(self.path)
        self.matcher.eval()

        with torch.no_grad():
            for idx, batch in enumerate(
                tqdm(
                    entity_dataloader, desc="UpdateEmbed", total=len(entity_dataloader)
                )
            ):
                batch = pl.utilities.move_data_to_device(batch, self.device)
                (
                    entity_text_seq_tokens,
                    entity_image_patch_tokens,
                    entity_text_cls,
                    entity_image_cls,
                ) = self.encoder(**batch)

                ent2save = {
                    "entity_text_seq_tokens": entity_text_seq_tokens,
                    "entity_image_patch_tokens": entity_image_patch_tokens,
                    "entity_text_cls": entity_text_cls,
                    "entity_image_cls": entity_image_cls,
                }
                torch.save(ent2save, os.path.join(self.path, f"{idx}.pth"))

    def on_test_epoch_end(self):
        shutil.rmtree(self.path)
        torch.cuda.empty_cache()

        if not os.path.exists(f"./error_record/{self.args.run_name}"):
            os.makedirs(f"./error_record/{self.args.run_name}")
        with open(f"./error_record/{self.args.run_name}/{self.record_name}", "w") as f:
            f.write(json.dumps(self.error_record_list, indent=4, ensure_ascii=False))

        ranks = np.concatenate([output["rank"] for output in self.test_step_outputs])
        self.test_step_outputs = []
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Test/hits20", hits20)
        self.log("Test/hits10", hits10)
        self.log("Test/hits5", hits5)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mr", ranks.mean())
        self.log("Test/mrr", (1.0 / ranks).mean())

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0001,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-4
        )
        return [optimizer]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Assuming batch is a dictionary
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        return batch
