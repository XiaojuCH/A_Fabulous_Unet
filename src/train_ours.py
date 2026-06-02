import os
import argparse
import datetime
import time
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from model import ST_SAM

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset_Ours/dataset")

CONFIG = {
    "batch_size": 2,
    "num_workers": 4,
    "lr": 1e-4,
    "epochs": 50,
    "img_size": 1024,
    "model_name": "ST_SAM",
}


class OursDataset(Dataset):
    def __init__(self, split, img_size=1024):
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
        stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(img_dir))]
        self.items = [(os.path.join(img_dir, s + ".jpg"), os.path.join(lbl_dir, s + ".png")) for s in stems]
        self.img_size = img_size
        self.is_train = (split == "train")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lbl_path = self.items[idx]
        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        label = Image.open(lbl_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        image_tensor = TF.to_tensor(image)
        label_np = (np.array(label) > 127).astype(np.uint8)
        label_tensor = torch.from_numpy(label_np).float().unsqueeze(0)

        ys, xs = np.where(label_np > 0)
        if len(ys) == 0:
            box = [0, 0, self.img_size, self.img_size]
        else:
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        if self.is_train:
            import random
            n = 20
            s = self.img_size
            box = [max(0, box[0] - random.randint(0, n)), max(0, box[1] - random.randint(0, n)),
                   min(s, box[2] + random.randint(0, n)), min(s, box[3] + random.randint(0, n))]

        return {"image": image_tensor, "label": label_tensor, "box": torch.tensor(box, dtype=torch.float32)}


class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        sig = torch.sigmoid(inputs)
        flat_i, flat_t = sig.view(-1), targets.view(-1)
        dice = 1 - (2 * (flat_i * flat_t).sum() + smooth) / (flat_i.sum() + flat_t.sum() + smooth)
        return 0.5 * bce + 0.5 * dice


def dice_score(preds, labels):
    preds_bin = (torch.sigmoid(preds) > 0.5).float()
    inter = (preds_bin * labels).sum()
    return (2 * inter / (preds_bin.sum() + labels.sum() + 1e-6)).item()


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["box"].to(device)
            preds = model(images, boxes)
            total += dice_score(preds, labels)
    return total / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(OursDataset("train", CONFIG["img_size"]), batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(OursDataset("val", CONFIG["img_size"]), batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = ST_SAM().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    criterion = DiceBCELoss().to(device)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    best_dice, best_epoch = 0.0, 0
    start = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            boxes = batch["box"].to(device)
            optimizer.zero_grad()
            preds = model(images, boxes)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = train_loss / len(train_loader)
        val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  -> New best: {best_dice:.4f} (epoch {best_epoch})")

    duration = str(datetime.timedelta(seconds=int(time.time() - start)))
    print(f"\nDone. Best Val Dice: {best_dice:.4f} (epoch {best_epoch}) | Time: {duration}")

    if args.test:
        test_loader = DataLoader(OursDataset("test", CONFIG["img_size"]), batch_size=CONFIG["batch_size"],
                                 shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
        test_dice = evaluate(model, test_loader, device)
        print(f"Test Dice: {test_dice:.4f}")

    stats = {**CONFIG, "best_dice": f"{best_dice:.4f}", "best_epoch": best_epoch, "duration": duration,
             "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../checkpoints_ours")
    parser.add_argument("--test", action="store_true", help="Run test set after training")
    main(parser.parse_args())
