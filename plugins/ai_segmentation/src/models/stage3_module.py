import torch
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall,
)
from src.models.components.loss_stage3 import Stage3Loss
from src.models.components.unet3d import U_Net

class Stage3SegmentationModule(pl.LightningModule):
    def __init__(
			self,
			in_ch: int = 4,
			out_ch: int = 1,
			init_features: int = 64,
			lr: float = 0.0003,
            dice_weight: float = 0.4,
            bce_weight: float = 0.4,
            skeleton_weight: float = 0.2,
            spine_error_weight: float = 2.0,
            neck_error_weight: float = 3.0,
	):
        super().__init__()
        self.save_hyperparameters()

        self.net = U_Net(in_ch=self.hparams.in_ch, out_ch=self.hparams.out_ch, init_features=self.hparams.init_features)

        self.loss_fn = Stage3Loss(dice_weight, bce_weight, skeleton_weight,
                                 spine_error_weight, neck_error_weight)

        self.val_dice = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.val_precision = BinaryPrecision(threshold=0.5)
        self.val_recall = BinaryRecall(threshold=0.5)

    def forward(self, x):
        return self.net(x)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Slice coordinates are Python metadata; reading CUDA scalars would synchronize.
        tensors = {key: value for key, value in batch.items() if key != 'borders'}
        transferred = super().transfer_batch_to_device(tensors, device, dataloader_idx)
        if 'borders' in batch:
            transferred['borders'] = batch['borders']
        return transferred

    def calculate_loss(self, logits, batch, collect_predictions=True):
        losses, ps, ts, present = [], [], [], []
        parts = {key: [] for key in ('dice', 'bce', 'skeleton')}
        for i in range(logits.shape[0]):
            slices = tuple(slice(int(a), int(b)) for a, b in batch['borders'][i])
            region = (slice(i, i + 1), slice(None), *slices)
            target = batch['mask'][region]
            valid = batch['valid_mask'][region] > 0
            present.append(valid.any().float())
            loss, components = self.loss_fn(
                logits[region], target, batch['spine_roi'][region],
                batch['neck_roi'][region], batch['local_skeleton'][region], valid,
                allow_empty=True)
            losses.append(loss)
            for key in parts:
                parts[key].append(components[key])
            if collect_predictions:
                ps.append(torch.sigmoid(logits[region].detach().float())[valid])
                ts.append(target[valid])
        present = torch.stack(present)
        count = present.sum()
        if count.is_cuda:
            torch._assert_async(count > 0)
        elif not count:
            raise ValueError('Stage3 batch contains no valid annotated voxels')
        def mean_valid(values):
            return (torch.stack(values) * present).sum() / count.clamp_min(1)
        return (mean_valid(losses), torch.cat(ps) if collect_predictions else None,
                torch.cat(ts) if collect_predictions else None,
                {k: mean_valid(v) for k, v in parts.items()})

    def _shared_step(self, batch, phase):
        logits = self(batch['image'])
        loss, p_valid, t_valid, parts = self.calculate_loss(
            logits, batch, collect_predictions=phase == 'val')
        self.log(f'{phase}/loss', loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=logits.shape[0])
        for key, value in parts.items():
            self.log(f'{phase}/loss_{key}', value, on_step=False, on_epoch=True,
                     batch_size=logits.shape[0])
        if phase == 'val':
            self.val_dice.update(p_valid, t_valid.int())
            self.val_iou.update(p_valid, t_valid.int())
            self.val_precision.update(p_valid, t_valid.int())
            self.val_recall.update(p_valid, t_valid.int())
            self.log('val/dice', self.val_dice, on_step=False, on_epoch=True)
            self.log('val/iou', self.val_iou, on_step=False, on_epoch=True)
            self.log('val/precision', self.val_precision, on_step=False, on_epoch=True)
            self.log('val/recall', self.val_recall, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):

        x, coords, valid_shape = batch["image"], batch["coords"],  batch["valid_shape"]

		# Инференс
        logits = self(x)
        preds = torch.sigmoid(logits)

		# Возвращаем предсказание и координаты, чтобы собрать их обратно
        return {"preds": preds, "coords": coords, "valid_shape": valid_shape}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,        # Ждать 3 валидационных эпох без улучшений перед снижением
            min_lr=1e-6,
            verbose=True
        )

        # 3. Возвращаем словарь, понятный для PyTorch Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1
            }
        }
