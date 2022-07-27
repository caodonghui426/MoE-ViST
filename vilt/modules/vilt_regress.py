from typing import OrderedDict
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils



class regressViLTransformerSS(pl.LightningModule):

    def __init__(self, config,sensor_class_n):
        super().__init__()
        self.save_hyperparameters()
        self.sensor_linear = nn.Linear(sensor_class_n,config["hidden_size"]) 

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # if self.hparams.config["load_path"] == "":
        self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
       

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)
        self.classifier = nn.Linear(config["hidden_size"],1)
        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            if isinstance(ckpt,OrderedDict):

                state_dict = ckpt
            else:
                state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
 
        sensor_embeds = self.sensor_linear(batch['sensor']) # input[1,1,12]  output[1,1,768]
        if image_embeds is None and image_masks is None:
            img = batch["image"][0]
            (
                image_embeds, # torch.Size([1, 217, 768])
                image_masks, # torch.Size([1, 217])
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        # 用embedding对数据输入预处理，降低维度
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )
        sensor_masks = batch['sensor_masks'] # 序列数量

        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x) # torch.Size([1, 240, 768])
        sensor_feats, image_feats = ( # torch.Size([1, 23, 768]),torch.Size([1, 217, 768])
            x[:, : sensor_embeds.shape[1]], # 后面字数输出23维
            x[:, sensor_embeds.shape[1] :], # 前面图片输出217维
        )
        cls_feats = self.pooler(x) # torch.Size([1, 768])
        cls_output = self.classifier(cls_feats)
        
        ret = {
           "sensor_feats":sensor_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats, # class features
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
           
            "patch_index": patch_index,

            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        ret.update(self.infer(batch))
        print("items:",ret.items())
        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k]) # ?
        self.log('train_loss', total_loss, on_epoch=True)
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        """
        定义优化器
        """
        return vilt_utils.set_schedule(self)

