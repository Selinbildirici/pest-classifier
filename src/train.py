"""
Training loop with mixed precision, differential LR, and checkpointing.

Mirrors the training logic used across all 8 runs in the project notebook
(notebooks/pest_classifier.ipynb), organized as reusable functions.

"""

import os
import json
import time
import torch
import torch.nn as nn


def freeze_backbone(model):
    """Freeze all parameters except the classification head.

    Used in Run 1 (frozen ViT baseline). Only the 'head' layer remains
    trainable.
    """
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    return model


def unfreeze_all(model):
    """Set all parameters trainable. Used in Runs 2-8."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def get_differential_optimizer(model, head_lr=3e-4, backbone_lr=3e-5, weight_decay=0.05):
    """AdamW with different learning rates for head vs. backbone.

    The classification head trains at head_lr (typically 3e-4) because it
    starts random; the pretrained backbone trains at backbone_lr (typically
    3e-5) to avoid disrupting useful pretrained features.

    Default values match Run 3 (our reference configuration).
    For HP tuning runs, backbone_lr was set to 1e-5 (Run 7) or 1e-4 (Run 8).
    """
    head_params = [p for n, p in model.named_parameters() if 'head' in n]
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': head_lr},
        {'params': backbone_params, 'lr': backbone_lr},
    ], weight_decay=weight_decay)
    return optimizer


def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=500):
    """Linear warmup + cosine annealing scheduler.

    For the first warmup_steps steps, learning rate ramps linearly from
    0 to its target. After that, it follows half a cosine wave down to 0.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device,
                    log_every=200):
    """Run one epoch of training with mixed precision.

    Returns (avg_train_loss, train_accuracy).
    """
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)
        if batch_idx % log_every == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}  "
                  f"loss={loss.item():.3f}  running_acc={train_correct/train_total:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
    return train_loss / train_total, train_correct / train_total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Compute validation loss and top-1 accuracy.

    For Run 4 (class-weighted training), pass an unweighted criterion here
    to keep validation comparable across runs.
    """
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.amp.autocast('cuda'):
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += images.size(0)
    return val_loss / val_total, val_correct / val_total


def train_run(model, train_loader, val_loader, optimizer, scheduler, scaler,
              criterion, device, num_epochs, run_id, checkpoint_dir, results_dir,
              patience=3, val_criterion=None):
    """Full training loop with early stopping and checkpoint saving.

    Used for all 8 runs. Saves the best checkpoint to checkpoint_dir and
    the training history JSON to results_dir.

    Args:
        model: PyTorch model
        train_loader: training DataLoader
        val_loader: validation DataLoader
        optimizer: PyTorch optimizer
        scheduler: LR scheduler
        scaler: torch.amp.GradScaler for mixed precision
        criterion: training loss function (may be class-weighted)
        device: torch.device
        num_epochs: max epochs to train
        run_id: string identifier for saving (e.g., 'run3_aug_vit')
        checkpoint_dir: directory to save .pt checkpoints
        results_dir: directory to save history JSON
        patience: early stopping patience (default 3)
        val_criterion: optional separate criterion for validation
                       (used in Run 4 to keep val unweighted for comparability)

    Returns:
        (best_val_acc, history_dict)
    """
    if val_criterion is None:
        val_criterion = criterion

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{num_epochs}  |  "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  |  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}  |  "
              f"time={epoch_time:.0f}s\n")

        # Save epoch checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f'{run_id}_epoch{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, ckpt_path)

        # Save best checkpoint and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'{run_id}_best.pt'))
            print(f"  New best val_acc={val_acc:.3f} — saved as best")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break

    # Save training history
    with open(os.path.join(results_dir, f'{run_id}_history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\nRun {run_id} complete. Best val_acc: {best_val_acc:.3f}")
    return best_val_acc, history
