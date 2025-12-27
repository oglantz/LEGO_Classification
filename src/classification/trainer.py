"""Training pipeline for LEGO classifier."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Tuple
import time
from tqdm import tqdm
import numpy as np

from .classifier import LEGOClassifier
from ..utils.device_utils import get_device, set_seed


class Trainer:
    """
    Trainer for LEGO classifier with mixed precision, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: LEGOClassifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        mixed_precision: bool = True,
        grad_accum_steps: int = 1,
        checkpoint_dir: str = "models/classification",
        log_dir: str = "logs",
        early_stopping_patience: int = 10,
    ):
        """
        Initialize trainer.
        
        Args:
            model: LEGOClassifier model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: PyTorch device
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            mixed_precision: Whether to use mixed precision training
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            early_stopping_patience: Patience for early stopping
            grad_accum_steps: Number of micro-batches to accumulate before optimizer step
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else get_device()
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.grad_accum_steps = max(1, grad_accum_steps)

        # Enable TF32 on Ampere+ for faster matmul/convs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        
        # Enable cuDNN autotuner for faster convolutions on fixed-size inputs
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        
        # Move model to device
        self.model = self.model.to(self.device)
        # Use channels_last for better tensor cores memory access
        self.model = self.model.to(memory_format=torch.channels_last)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated based on epochs
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision dtype and scaler
        self.autocast_dtype = torch.float32
        self.scaler = None
        if self.mixed_precision:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
                print("Using mixed precision training (BF16)")
            else:
                self.autocast_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
                print("Using mixed precision training (FP16)")
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                scaled_loss = loss / self.grad_accum_steps
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Optimizer step every grad_accum_steps or on last batch
                if ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader)):
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                (loss / self.grad_accum_steps).backward()

                if ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
            self.global_step += 1
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        # Compute top-5 accuracy
        top5_correct = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                if self.mixed_precision:
                    with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                _, top5_preds = torch.topk(outputs, 5, dim=1)
                top5_correct += sum([labels[i] in top5_preds[i] for i in range(len(labels))])
        
        top5_acc = 100.0 * top5_correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'top5_accuracy': top5_acc,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_classes': self.model.num_classes,
            'best_val_acc': self.best_val_acc,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint (val_acc: {self.best_val_acc:.2f}%)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        # Update scheduler T_max
        self.scheduler.T_max = num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.writer.add_scalar('Train/EpochLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/EpochAccuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            if val_metrics:
                self.writer.add_scalar('Val/EpochLoss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/EpochAccuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/Top5Accuracy', val_metrics.get('top5_accuracy', 0), epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            if val_metrics:
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                if 'top5_accuracy' in val_metrics:
                    print(f"Val Top-5 Acc: {val_metrics['top5_accuracy']:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Checkpointing
            is_best = False
            if val_metrics and val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        self.writer.close()

