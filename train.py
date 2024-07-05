import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

# NOTE: Need to import tensorflow before pytorch lightning else protocol buffer runtime library miscongruencies
import tensorflow as tf  
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
import gc
import pdb

from src.dataset.dataset import SubCellDatset, DatasetType
from src.model.full_model import SubCellProtModel


def run_train(
    collection_name,
    batch_size=32,
    num_workers=8,
    max_epochs=100,
    train_batch_limit=3000,
    eval_batch_limit=500,
    debug=False,
    devices=None,
    use_old_hpa_client=True
):
    seed_everything(42, workers=True)
    model = SubCellProtModel()
    train_dataset, val_dataset, test_dataset = (
        SubCellDatset(
            split=DatasetType.TRAIN,
            collection_name=collection_name,
            if_alphabetical=use_old_hpa_client,
        ),
        SubCellDatset(
            split=DatasetType.EVAL,
            collection_name=collection_name,
            if_alphabetical=use_old_hpa_client,
        ),
        SubCellDatset(
            split=DatasetType.TEST,
            collection_name=collection_name,
            if_alphabetical=use_old_hpa_client,
        ),
    )

    def ignore_bad_collate(batch):
        batch = [b for b in batch if b is not None]
        batch = [b for b in batch if b[0].sum() != 0]
        return torch.utils.data.default_collate(batch)

    train_loader, val_loader, test_loader = (
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=ignore_bad_collate,
        ),
        torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=ignore_bad_collate,
        ),
        torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=ignore_bad_collate,
        ),
    )
    trainer_args = {
        "limit_train_batches": train_batch_limit,
        "limit_val_batches": eval_batch_limit,
        "max_epochs": max_epochs,
        "default_root_dir": "checkpoints",  
        "log_every_n_steps": 1,
        "callbacks": [
            ModelCheckpoint(
                monitor="val_combined_loss",
                save_top_k=-1,
                filename=collection_name + "-{epoch:02d}-{val_combined_loss:.2f}",
            ),
        ],
        "deterministic": True,
    }
    if debug:
        assert (
            devices is not None and len(devices) == 1
        ), "For debugging purposes (aka to use the python debugger), run it on just one gpu"
        trainer_args["devices"] = devices

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    train_dataset.__del__()
    val_dataset.__del__()
    test_dataset.__del__()


def main():
    torch.cuda.empty_cache()
    run_train(
        collection_name="splice_isoform_dataset_cell_line_and_gene_split_full",
        batch_size=32,
        num_workers=8,
        max_epochs=100,
        debug=False,
        devices=None,
        if_alphabetical=True,
    )

if __name__ == "__main__":
    main()
    n = gc.collect()
    print("garbage collected ", n, " items")
