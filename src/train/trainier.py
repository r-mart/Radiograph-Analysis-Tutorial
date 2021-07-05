import time
from datetime import datetime

import torch
import torch.nn as nn

from .utils import AverageMeter


class ClassificationTrainer():
    def __init__(self, model, cfg) -> None:
        self.cfg = cfg
        self.epoch = 0

        self.log_base = cfg.log_path
        if not self.log_base.exists():
            self.log_base.mkdir(parents=True)

        self.log_path = self.log_base / 'log.txt'
        self.best_val_loss = 10**5
        self.best_val_acc = 0.0

        self.model = model
        self.device = cfg.device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min')
        self.log(f'Trainer prepared. Device is {self.device}')
        self.log(f'Fold num is {cfg.fold_num}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.cfg.n_epochs):
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            train_loss = self.training_step(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, train_loss: {train_loss:.5f}, time: {(time.time() - t):.5f}')
            self.save(self.log_base / 'last-checkpoint.pt')

            t = time.time()
            val_loss, val_acc = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}, time: {(time.time() - t):.5f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.model.eval()
                self.save(self.log_base /
                          f'best-checkpoint-{str(self.epoch).zfill(3)}epoch.pt')
                for path in sorted(self.log_base.glob('best-checkpoint-*epoch.pt'))[:-3]:
                    path.unlink()

            if self.cfg.use_scheduler:
                self.scheduler.step(metrics=val_loss)

            self.epoch += 1

    def training_step(self, train_loader):
        self.model.train()

        for images, targets, image_ids in train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()

            self.optimizer.step()

        return loss

    def validation(self, val_loader):
        self.model.eval()

        score = AverageMeter()
        for images, targets, image_ids in val_loader:

            with torch.no_grad():
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                _, preds = torch.max(logits, 1)
                n_correct = (preds == targets).sum().item()

                score.update(n_correct, self.cfg.batch_size)

        return loss, score.avg

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
