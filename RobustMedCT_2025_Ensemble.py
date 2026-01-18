import os
import random
import numpy as np
import pandas as pd
import warnings
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm
from tqdm import tqdm


# =======
# CONFIG
# =======


class Config:
    # Most hyperparams are set here because they will be same for all configurations
    def __init__(
        self,
        dataset_root="dataset",
        num_classes=11,
        image_size=224,
        batch_size=16,
        num_workers=4,
        epochs=None,
        lr=3e-4,
        weight_decay=1e-4,
        model_names=None,
        seed=None,
        device=None,
    ):
        self.dataset_root = dataset_root

        # Paths
        self.train_images_dir = os.path.join(dataset_root, "train", "images_train")
        self.train_labels_path = os.path.join(dataset_root, "train", "labels_train.csv")

        self.val_images_dir = os.path.join(dataset_root, "val", "images_val")
        self.val_labels_path = os.path.join(dataset_root, "val", "labels_val.csv")

        self.test_images_dir = os.path.join(dataset_root, "test", "images")
        self.test_manifest_path = os.path.join(dataset_root, "test", "manifest_public.csv")

        # Hyperparams
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        # Models
        self.model_names = model_names 

        # Reproducibility
        self.seed = seed

        # Device
        if torch.cuda.is_available():
            self.device = device or "cuda"
            print(f"using device {self.device}")
        else:
            self.device = "cpu"
            warnings.warn("No CUDA detected. Training on CPU will be extremely slow.")


# ======
# UTILS
# ======

def set_seed(seed) -> None:
    if seed is None:
        raise ValueError("Config error: 'seed' cannot be None.")
    print(f"Setting seed to: {seed}")

    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def read_table(path: str) -> pd.DataFrame:
    print(f"Reading filename: {os.path.basename(path)}")
    return pd.read_csv(path)


# ========
# DATASET
# ========


class OrganDataset(Dataset):
    """Custom dataset

    - For training/validation:
      pass in a labels file (CSV/Excel) with columns: 'file' and 'label'.

    - For testing:
      set manifest=True and use a manifest CSV with columns: 'index' and 'file'
      (no labels, just filenames and their indices for the submission).
    """

    def __init__(self, images_dir: str, labels_path: str | None = None,
                 manifest: bool = False, transform=None) -> None:
        self.images_dir = images_dir
        self.transform = transform
        self.manifest = manifest

        if labels_path is not None:
            df = read_table(labels_path)
            if manifest:
                # test set case: 'index' + 'file'
                self.indices = df["index"].tolist()
                self.filenames = df["file"].tolist()
                self.labels = None
            else:
                # train/val case: 'file' + 'label'
                self.filenames = df["file"].tolist()
                self.labels = df["label"].astype(int).tolist()
                self.indices = None
        else:
            # We don't want to end up here as we cannot proceed without label or manifest file in any circumstance.
            raise ValueError(
                f"labels_path is None. This dataset requires a label file or manifest file. "
                f"Received labels_path=None for images_dir='{images_dir}'."
            )


    # magic method python calls when len is used on this object i.e "len(dataset)"
    def __len__(self) -> int:
        return len(self.filenames)

    # magic method python calls when this object is indexed... i.e "dataset[idx]"
    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert("L")  # greyscale
        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            idx_manifest = self.indices[idx] if self.indices is not None else idx # Indices in manifest are 0,1,2,3,4,5.... so this is redundant here but good practice. 
            return img, idx_manifest

# ================
# MODEL UTILITIES
# ================

def create_model(model_name: str) -> nn.Module:

    print(f"Creating model: {model_name}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=cfg.num_classes,
        in_chans=3,
    )
    return model


def train_single_model(model_name: str) -> None:

    if cfg.epochs == None:
        raise ValueError("Config error: 'epochs' cannot be None.")

    set_seed(cfg.seed)

    # Datasets & loaders
    train_dataset = OrganDataset(
        cfg.train_images_dir,
        cfg.train_labels_path,
        manifest=False,
        transform=train_tfms,
    )
    val_dataset = OrganDataset(
        cfg.val_images_dir,
        cfg.val_labels_path,
        manifest=False,
        transform=val_tfms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = create_model(model_name).to(cfg.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Cosine annealing i.e LR starts high and smoothly curves down to a small value following a cosine wave shape.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    checkpoint_path = f"best_{model_name}.pth"

    for epoch in range(cfg.epochs):

        # ---- TRAIN ----
        # Main training loop: forward, backward, optimize, log metrics
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{cfg.epochs} [train]") # progress bar 
        for images, labels in pbar:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ---- VALIDATION ----
        # Run validation pass: disable grads, measure loss/accuracy, update LR
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"{model_name} Epoch {epoch+1}/{cfg.epochs} [val]  ")
            for images, labels in pbar_val:
                images = images.to(cfg.device, non_blocking=True)
                labels = labels.to(cfg.device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        print(
            f"{model_name} Epoch {epoch+1}/{cfg.epochs} "
            f"TrainLoss: {train_loss:.4f} "
            f"TrainAcc: {train_acc:.4f} "
            f"ValLoss: {val_loss:.4f} "
            f"ValAcc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved to {checkpoint_path}")

    print(f"{model_name} training finished. Best val acc: {best_val_acc:.4f}")


def train_all_models() -> None:
    """Train all models defined in cfg.model_names sequentially."""
    for name in cfg.model_names:
        # Skip training if checkpoint already exists
        checkpoint_path = f"best_{name}.pth"
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} exists. Skipping training for {name}.")
            continue
        train_single_model(name)


# ==========
# INFERENCE 
# ==========


def predict_ensemble_and_create_submission() -> None:
    """Ensemble inference for the test set, outputting submission.csv.

    Loads each model checkpoint, averages their logits over the batch, picks
    the highest-scoring class, and saves the predictions to submission.csv.
    """

    # Load manifest and sort indices to guarantee correct ordering
    df_manifest = read_table(cfg.test_manifest_path)
    df_manifest = df_manifest.sort_values("index").reset_index(drop=True)

    # Prepare test dataset and loader
    test_dataset = OrganDataset(
        cfg.test_images_dir,
        labels_path=cfg.test_manifest_path,
        manifest=True,
        transform=val_tfms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Load each model checkpoint
    models: list[nn.Module] = []
    for name in cfg.model_names:
        checkpoint_path = f"best_{name}.pth"
        if not os.path.exists(checkpoint_path):
            err = f"Checkpoint for {name} not found at {checkpoint_path}"
            raise FileNotFoundError(err)
        
        model = create_model(name)
        model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
        model.to(cfg.device)
        model.eval()
        models.append(model)
        print(f"Loaded {name} from {checkpoint_path}")

    all_indices: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Ensemble inference on test set") # progress bar
        for images, idx_manifest in pbar:
            images = images.to(cfg.device, non_blocking=True)
            # Sum logits across models
            logits_sum: torch.Tensor | None = None
            for model in models:
                outputs = model(images)
                if logits_sum is None:
                    logits_sum = outputs
                else:
                    logits_sum += outputs
            # Average logits (sum = average with argmax) 
            preds = logits_sum.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_indices.extend(idx_manifest.cpu().numpy().tolist())

    # Build DataFrame from indices + preds
    df_sub = pd.DataFrame({"index": all_indices, "id": all_preds})
    df_sub = df_sub.sort_values("index").reset_index(drop=True)
    df_sub.to_csv("submission.csv", index=False)
    print("submission.csv generated using ensemble")


# =====
# MAIN
# =====

def main():
    global cfg, train_tfms, val_tfms

    cfg = Config(seed=3407, epochs=20, model_names= [
            "convnext_small",
            "tf_efficientnetv2_s",
            "swin_tiny_patch4_window7_224",
            "davit_small.msft_in1k",
            "pit_s_distilled_224"])

    # ===========
    # TRANSFORMS
    # ===========

    train_tfms = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.Grayscale(num_output_channels=3),
        T.RandomResizedCrop(cfg.image_size, scale=(0.9, 1.0)),
        T.RandomRotation(15),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Train each model if no checkpoint exists
    train_all_models()
    # Run ensemble inference and generate submission
    predict_ensemble_and_create_submission()


if __name__ == "__main__":
    main()




