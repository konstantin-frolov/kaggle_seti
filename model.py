from utils import *
from pathlib import Path
from datetime import datetime
from time import time
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings


warnings.filterwarnings("ignore")


###################################
# Config
###################################
class TrainGlobalConfig:
    num_workers = 10
    batch_size = 24
    n_epochs = 40
    lr = 0.0003
    folds = 5

    folder = 'models'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    #     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    #     scheduler_params = dict(
    #         max_lr=0.001,
    #         epochs=n_epochs,
    #         steps_per_epoch=int(len(train_dataset) / batch_size),
    #         pct_start=0.1,
    #         anneal_strategy='cos',
    #         final_div_factor=10**5
    #     )

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )


###################################
# Dataset
###################################
class SetiDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = Path(img_dir)

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['id']
        label = self.df.iloc[idx]['target'].astype(float)
        image = np.load(self.img_dir.joinpath(image_name[0], image_name + '.npy'))
        image -= image.min()
        image = ((image / np.max(image)) * 255).astype(np.uint8)
        return image, label

    def __change_label__(self, idx, label):
        self.df.at[idx, 'target'] = label


class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        image, label = self.dataset.__getitem__(idx)
        if self.transform:
            image = self.transform(image=image)['image']
        image = image / 255.
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)


############################################################
# Losses
############################################################
def compute_class_weights(df):
    w_n = df['target'].sum() / df['target'].__len__()
    w_p = 1 - w_n
    return w_p, w_n


def weighted_bce(df):
    w_p, _ = compute_class_weights(df)
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w_p, dtype=torch.float))


class MetricsMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.max = -1e6
        self.min = -1e6
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e6
        self.min = -1e6

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        if self.max < val:
            self.max = val
        if self.min > val:
            self.min = val

    def __str__(self):
        answer = f'mean value - {np.mean(self.vals)}, ' \
                 f'max value - {self.max}, ' \
                 f'min value - {self.min}'
        return answer


######################################
# Models
######################################
class DoubleNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model1 = backbone
        self.model2 = backbone
        last_layer = list(self.model1.named_modules())[-1]
        self.fc = torch.nn.Linear(2 * last_layer[1].in_features, 1)
        self.model1.add_module(last_layer[0], torch.nn.Identity())
        self.model2.add_module(last_layer[0], torch.nn.Identity())
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.model1(x[:, :3, ...])
        x2 = self.model2(x[:, 3:, ...])
        x = torch.cat((x1, x2), 1)
        x = self.fc(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sigmoid(x)
        return x


######################################
# Fitter
######################################
class Fitter:

    def __init__(self, net, path2weights, now_device, config, loss):
        self.config = config
        self.epoch = 0
        self.now_fold = 0
        self.now_device = now_device
        self.summary_scores = []

        self.base_dir = Path(__file__).parent.joinpath(config.folder)
        self.best_models = self.base_dir.joinpath('best_models')
        self.all_models = self.base_dir.joinpath('all_models')
        self.log_dir = self.base_dir.joinpath('logs')

        check_and_create_folder(self.base_dir)
        check_and_create_folder(self.best_models)
        check_and_create_folder(self.all_models)
        check_and_create_folder(self.log_dir)

        self.log_path = self.log_dir.joinpath('log.txt')
        self.best_summary_loss = 1e5
        self.best_val_auc = 1e-5
        self.best_train_auc = 1e-5
        self.crosval_best_aucs = []
        self.criterion = loss.to(self.now_device)
        self.model = net.to(self.now_device)
        self.weights = path2weights
        self.optimizer = None
        param_optimizer = list(self.model.named_parameters())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = self.config.SchedulerClass(self.optimizer, **self.config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.now_device}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __clear__(self):
        self.summary_scores = []
        self.crosval_best_aucs = []
        self.best_summary_loss = 1e5
        self.best_val_auc = 1e-5
        self.best_train_auc = 1e-5

    def prepare_model(self):
        self.model.load_state_dict(torch.load(self.weights))
        self.model = self.model.to(self.now_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = self.config.SchedulerClass(self.optimizer, **self.config.scheduler_params)

    def fit_crossval(self, dataset, transform):
        self.__clear__()
        crosval_metrics = MetricsMeter()
        kfold = KFold(n_splits=self.config.folds, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            self.now_fold = fold
            tqdm.write('START NEW FOLD')
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(
                ApplyTransform(dataset, transform),
                batch_size=self.config.batch_size,
                sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                ApplyTransform(dataset),
                batch_size=self.config.batch_size,
                sampler=test_subsampler)
            now_best_auc = self.fit(trainloader, testloader, return_vals=True)
            crosval_metrics.update(now_best_auc)
        tqdm.write(f'[ALL RESULTS]: AUC ROC \n' +
                   crosval_metrics.__str__())

    def fit(self, train_loader, validation_loader, return_vals=False):
        self.prepare_model()

        for e in range(self.config.n_epochs):  # by epochs
            self.epoch = e
            t = time()
            summary_loss, summary_score = self.train_one_epoch(train_loader)

            tqdm.write(
                f'[RESULT]: Train. Epoch: {self.epoch}, '
                f'summary_loss: {summary_loss.avg:.5f}, '
                f'AUC_ROC: {summary_score:.3f} '
                f'time: {(time() - t):.5f}')
            self.writer.add_scalar('Loss/train', summary_loss.avg, self.epoch)
            self.writer.add_scalar('AUC_ROC/train', summary_score, self.epoch)

            self.save(self.base_dir.joinpath('last-checkpoint.bin'))

            summary_loss, summary_score = self.validation(validation_loader)
            self.writer.add_scalar('Loss/val', summary_loss.avg, self.epoch)
            self.writer.add_scalar('AUC_ROC/val', summary_score, self.epoch)
            t = time()
            tqdm.write(
                f'[RESULT]: Val. Epoch: {self.epoch}, '
                f'summary_loss: {summary_loss.avg:.5f}, '
                f'AUC ROC: {summary_score:.3f} '
                f'time: {(time() - t):.5f}')

            self.save(self.all_models.joinpath(
                f'checkpoint_fold_{str(self.now_fold)}_{str(self.epoch).zfill(3)}_epoch_val_aucroc_{str(np.round(summary_score, 3))}.bin'))
            if summary_score > self.best_val_auc:
                self.best_val_auc = summary_score
                self.model.eval()
                self.save(self.best_models.joinpath(
                    f'best_checkpoint_fold_{str(self.now_fold)}_val_aucroc_{str(np.round(summary_score, 3))}_{str(self.epoch).zfill(3)}_epoch.bin'))

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

        if return_vals:
            return self.best_val_auc

    def validation(self, val_loader):
        self.model.eval()
        summary_scores = []
        summary_loss = MetricsMeter()
        all_outputs = np.zeros(val_loader.batch_size * val_loader.__len__())
        all_targets = np.zeros(val_loader.batch_size * val_loader.__len__())
        t = time()
        with tqdm(val_loader, unit="batch", ncols=400) as vepoch:
            vepoch.set_description(f'Validation time: {np.round(time() - t, 2)}')
            for step, (images, target) in enumerate(vepoch):
                with torch.no_grad():
                    images = images.to(self.now_device)
                    target = target.to(self.now_device)
                    batch_size = images.shape[0]
                    outputs = self.model(images)
                    loss = self.criterion(outputs, target.view(-1, 1))
                    summary_loss.update(loss.detach().item(), batch_size)
                    all_outputs[step * batch_size:(step + 1) * batch_size] = np.reshape(outputs.cpu().detach().numpy(), -1)
                    all_targets[step * batch_size:(step + 1) * batch_size] = np.reshape(target.cpu().detach().numpy(), -1)
                    vepoch.set_postfix(loss=np.round(summary_loss.avg, 5))
            summary_score = get_roc_score(all_targets, all_outputs)
            summary_scores.append(summary_score)
            vepoch.set_postfix(loss=np.round(summary_loss.avg, 5), auc_roc=np.round(summary_score, 3))
        self.summary_scores.append(summary_scores)
        return summary_loss, summary_score

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = MetricsMeter()
        all_outputs = np.zeros(train_loader.batch_size * train_loader.__len__())
        all_targets = np.zeros(train_loader.batch_size * train_loader.__len__())
        t = time()
        tqdm.write(f'Time: {datetime.utcnow().isoformat()} \n')
        with tqdm(train_loader, unit="batch", ncols=400) as tepoch:
            lr = self.optimizer.param_groups[0]['lr']
            tepoch.set_description(f'LR: {lr}, Time: {np.round(time() - t, 2)}')
            for step, (images, target) in enumerate(tepoch):
                images = images.to(self.now_device)
                target = target.to(self.now_device)
                batch_size = images.shape[0]
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, target.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                summary_loss.update(loss.detach().item(), batch_size)
                all_outputs[step * batch_size:(step + 1) * batch_size] = np.reshape(outputs.cpu().detach().numpy(), -1)
                all_targets[step * batch_size:(step + 1) * batch_size] = np.reshape(target.cpu().detach().numpy(), -1)
                tepoch.set_postfix(loss=np.round(summary_loss.avg, 5))
            summary_score = get_roc_score(all_targets, all_outputs)
            tepoch.set_postfix(loss=np.round(summary_loss.avg, 5), auc_roc=np.round(summary_score, 3))
        if self.config.step_scheduler:
            self.scheduler.step()
        return summary_loss, summary_score

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
