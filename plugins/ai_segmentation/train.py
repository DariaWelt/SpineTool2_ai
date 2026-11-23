# import os
# отключаем строгий режим загрузки весов PyTorch 2.6+, чтобы можно было продолжить обучение загруженных весов
# os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage1_lora")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    # 1. Init DataModule
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # 2. Init Model
    model = hydra.utils.instantiate(cfg.model)

    is_stage3 = cfg.model._target_ == "src.models.stage3_module.Stage3SegmentationModule"
    checkpoint = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir, 
        monitor="val/loss", 
        mode="min", 
        save_top_k=2,#сколько лучших эпох сохранить
        save_last=False,
        save_weights_only=True,
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        auto_insert_metric_name=False
    )
    
    callbacks = [checkpoint]
    # Stage3 keeps independent rankings: loss for historical comparisons,
    # Dice for selecting segmentation quality on validation data.
    if is_stage3:
        callbacks.append(ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            monitor="val/dice",
            mode="max",
            save_top_k=2,
            save_last=False,
            save_weights_only=True,
            filename="epoch={epoch:02d}-val_dice={val/dice:.4f}",
            auto_insert_metric_name=False,
        ))
    # All models keep one full checkpoint for resuming training.
    # A separate callback is needed: save_weights_only also applies to last.ckpt.
    callbacks.append(ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        save_top_k=0,
        save_last=True,
        save_weights_only=False,
    ))
    logger = TensorBoardLogger(cfg.paths.log_dir, 
        name="folds",
        version=f"fold_{cfg.paths.fold}")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # 3. Init Trainer
    trainer = pl.Trainer(
        logger=logger, callbacks=[*callbacks, lr_monitor],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.get('devices', 'auto'),
        accumulate_grad_batches=cfg.trainer.get('accumulate_grad_batches', 1),
        precision=cfg.trainer.get('precision', '32-true'),
        gradient_clip_val=cfg.trainer.get('gradient_clip_val', None),

    )
    # 4. Train
    trainer.fit(model=model, datamodule=datamodule)
    #если нужно продолжить обучение с какой то эпохи
    # trainer.fit(model=model, datamodule=datamodule, ckpt_path="C:/Users/Student/datasets/CVstage4/checkpoints/f_s2_08s/folds2/last-v1.ckpt")

if __name__ == "__main__":
    main()
