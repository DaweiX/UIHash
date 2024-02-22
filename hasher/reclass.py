"""Reidentify visible views in a UI"""

import argparse
import sys
from typing import Tuple

import torch.optim as optim
import cv2
import torchvision.models
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import progressbar as bar
from sys import stdout
import torch
from time import perf_counter
from os.path import exists, join
import torch.nn.functional as f
from os import walk, listdir, makedirs
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import os

curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
if rootpath not in sys.path:
    sys.path.append(rootpath)


class ImgDataSet(Dataset):
    """Build and load the view image dataset"""
    def __init__(self, root_path: str):
        npy_file = join(root_path, "imgdata.npy")
        self._class_names = listdir(root_path)
        self._class_names = [i for i in self._class_names if not i.endswith('.npy')]
        if not exists(npy_file):
            print("generate dataset...")
            data = []
            for i, _class in enumerate(self._class_names):
                print(_class)
                sub_class = f'{root_path}\\{_class}'
                img_files = listdir(sub_class)
                for img_file in img_files:
                    img = cv2.imread(join(sub_class, img_file))
                    try:
                        img = cv2.resize(img, (28, 28))
                    except:
                        continue
                    # we ignore the color of control images
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    data.append([img, i])
            if len(data) > 0:
                np.save(join(root_path, "imgdata.npy"),
                        data, allow_pickle=True)
                self.data = np.load(npy_file, allow_pickle=True)
                self._classnum = np.max(self.data[:, 1]) + 1
            else:
                self.data = []
                self._classnum = len(self._class_names)
        else:
            self.data = np.load(npy_file, allow_pickle=True)
            self._classnum = np.max(self.data[:, 1]) + 1
        # shuffle the data by shuffling the indices
        self.index = [i for i in range(self.__len__())]
        np.random.shuffle(self.index)
        print("dataset ready")

    def __getitem__(self, idx: int):
        idx = self.index[idx]
        i, label = self.data[idx]
        return i, label

    def __len__(self):
        return len(self.data)

    @property
    def class_num(self):
        return self._classnum

    @property
    def class_names(self):
        return self._class_names


class ImgNet(nn.Module):
    def __init__(self, class_num: int):
        """A convolutional neural network based on ResNet

        Args:
            class_num: neural number of the output fc layer
        """
        super(ImgNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet18(pretrained=False)
        fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fc_in_features, class_num)
        self.target = [i for i in range(class_num)]

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = f.softmax(x, dim=1)
        return x


class ImgClassifier:
    def __init__(self, dataset_path: str,
                 lr_init: float = 0.001,
                 lr_decay: Tuple[int, float] = (10, 0.1),
                 epoch: int = 5,
                 batch_size: int = 32,
                 retrain_model: bool = False,
                 model_name: str = "",
                 confidence_threshold: float = 0.95):
        """

        Args:
            dataset_path (str): Dataset path
            lr_init (float): Initial value for the learning rate
            lr_decay: (int, float) - The first int indicates the
              epoch interval to decrease the learning rate. The
              second float is the decay ratio
            epoch (int): Training epoch
            batch_size (int): Training batch size
            retrain_model (bool): Retrain and update the model if
              the model exists
            model_name (str): Name for output model. If not assign it,
              it will be `reclass_e{epoch}_{batch_size}.tar`
            confidence_threshold (float): The confidence to take the
              predicted label
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {self.device}")
        self.epoch = epoch
        self.batch_size = batch_size
        self.dataset = ImgDataSet(dataset_path)
        self._class_names = self.dataset.class_names
        self.net = ImgNet(self.dataset.class_num).to(self.device)

        self.lr = lr_init
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        s1 = int(0.8 * len(self.dataset))
        s2 = int(0.9 * len(self.dataset))  # 8:1:1
        index_list = list(range(len(self.dataset)))
        self.train_idx, self.valid_idx, self.test_idx = \
            index_list[:s1], index_list[s1:s2], index_list[s2:]

        self.tr_samp = sampler.SubsetRandomSampler(self.train_idx)
        self.val_samp = sampler.SubsetRandomSampler(self.valid_idx)
        _model_name = model_name if len(model_name) > 3 else \
            f"reclass_e{self.epoch}_{self.batch_size}.tar"
        model_root_path = join(os.path.abspath(os.path.dirname(__file__)),
                               "..", "models")
        if not exists(model_root_path):
            makedirs(model_root_path)
        self.model_path = join(model_root_path, _model_name)
        self.retrain_model = retrain_model
        self.confidence_threshold = confidence_threshold

    @property
    def class_names(self):
        return self._class_names

    def deal_data(self, data):
        i, label = data
        i = np.array([a.unsqueeze_(0).numpy() for a in i])
        i = torch.from_numpy(i).float()
        label = label.long()
        i, label = i.to(self.device), label.to(self.device)
        return i, label

    def train(self, trainloader, validloader, draw_history: bool = False):
        """training & validating

        Args:
            trainloader: Dataload for the training subset
            validloader: Dataload for the validating subset
            draw_history (bool): whether to show a history plot
        """
        loss_tr, loss_val = [], []
        for e in range(self.epoch):
            train_loss = []
            valid_loss = []
            self.net.train()
            pbar = bar.ProgressBar(
                widgets=[f"Training ({e + 1}/{self.epoch}) ",
                         bar.Percentage(), ' ', bar.Bar('=')],
                fd=stdout, maxval=len(trainloader))
            pbar.start()
            for batch_idx, data in enumerate(trainloader):
                pbar.update(batch_idx + 1)
                i, t = self.deal_data(data)
                self.optimizer.zero_grad()
                o = self.net(i)
                loss = f.nll_loss(torch.log(o), t)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            pbar.finish()
            self.net.eval()
            pbar2 = bar.ProgressBar(
                widgets=[f"Validating ({e + 1}/{self.epoch}) ",
                         bar.Percentage(), ' ', bar.Bar('=')],
                fd=stdout, maxval=len(validloader))
            pbar2.start()
            for batch_idx, data in enumerate(validloader):
                pbar2.update(batch_idx + 1)
                i, label = self.deal_data(data)
                o = self.net(i)
                loss = f.nll_loss(torch.log(o), label)
                valid_loss.append(loss.item())

            pbar2.finish()
            self.lr = self.lr_init * (self.lr_decay[1] ** int(e / self.lr_decay[0]))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            _loss_train = np.round(np.mean(train_loss), 2)
            _loss_val = np.round(np.mean(valid_loss), 2)

            loss_tr.append(_loss_train)
            loss_val.append(_loss_val)
            print("Train Loss", _loss_train, "Valid Loss", _loss_val)

        # draw loss plot
        if draw_history:
            from util.util_draw import draw_history
            draw_history(self.epoch - 1, loss_tr[1:], loss_val[1:], True)
        torch.save(self.net.state_dict(), self.model_path)

    def test(self, dataloader):
        """Test the model on a test dataset"""
        self.net.eval()
        sum_correct = 0
        sum_total = 0
        pbar = bar.ProgressBar(
            widgets=["Testing ", bar.Percentage(),
                     ' ', bar.Bar('=')],
            fd=stdout, maxval=len(dataloader))
        pbar.start()
        confuse_mat = np.zeros((self.dataset.class_num,
                                self.dataset.class_num))
        num_predict, num_predict_correct = 0, 0
        for batch_idx, data in enumerate(dataloader):
            u, t = self.deal_data(data)
            pbar.update(batch_idx + 1)
            o = self.net(u)
            output_vector = o.cpu().detach().numpy()
            output_labels = np.argmax(output_vector, axis=1)
            output_labels_confident = \
                [np.argmax(i) if max(i) > self.confidence_threshold
                 else -1 for i in output_vector]

            sum_total += t.size(0)
            tarray = t.cpu()
            num_predict += len([i for i in output_labels_confident if i > -1])
            num_predict_correct += \
                len([i for i in range(len(tarray)) if
                     output_labels_confident[i] == tarray[i] and
                     output_labels_confident[i] > -1])
            for o, t in zip(output_labels, tarray):
                confuse_mat[o, t] += 1

            a = len([i for i in range(len(tarray)) if
                     output_labels[i] == tarray[i]])
            sum_correct += a

        pbar.finish()
        print(f'acc for {sum_total} samples: '
              f'{(sum_correct / sum_total * 100):.2f}%')
        print(f'acc for {num_predict} samples (confidence={self.confidence_threshold}): '
              f'{(num_predict_correct / num_predict * 100):.2f}%')
        for i, name in enumerate(self.class_names):
            if max(confuse_mat[i]) == 0:
                continue
            acc = confuse_mat[i][i] / sum(confuse_mat[i])
            print(f"Acc of {name}: {acc}")

    def train_and_test(self):
        trainloader = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 sampler=self.tr_samp)
        testloader = DataLoader(torch.utils.data.Subset(
            self.dataset, self.test_idx), batch_size=self.batch_size)
        validloader = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 sampler=self.val_samp)

        # training
        start = perf_counter()
        if not self.retrain_model and exists(self.model_path):
            self.net.load_state_dict(
                torch.load(self.model_path, map_location=self.device))
            print("model loaded")
        else:
            self.train(trainloader, validloader)
            torch.save(self.net.state_dict(), self.model_path)

        # predicting
        mid = perf_counter()
        self.test(testloader)
        stop = perf_counter()

        print("---------------------------")
        print("Training time:", mid - start)
        print("Testing time:", stop - mid)

    def predict(self, root_opt_path: str, skip_existance: bool = True):
        """Predict views for a UI dataset

        Args:
            root_opt_path (str): Root path for the app UIs (starts with
              `opt_`)
            skip_existance (bool): If a view is already labeled, than
              skip it

        Returns:
            No return value. The results will be saved in the input folders
        """
        if not exists(self.model_path):
            self.train_and_test()
        print(f"reidentify views in {root_opt_path}...")
        self.net.load_state_dict(
            torch.load(self.model_path, map_location=self.device))
        dlist = []
        for root, dirs, _ in walk(root_opt_path):
            for d in dirs:
                d = join(root, d)
                _list = str(listdir(d))
                if ".xml\'" in _list or _list == "[]":
                    continue
                dlist.append(d)
        total = len(dlist)
        views_handled, views_identified = 0, 0
        k = 0
        print(f"reidentify views for {total} uis")
        self.net.eval()
        start = perf_counter()
        for d in dlist:
            output_file = join(d, "classify.txt")
            if exists(output_file) and skip_existance:
                continue
            imgs = listdir(d)
            imgs = [i for i in imgs if i.endswith(".jpg")]
            imgs = sorted(imgs)
            labels = dict()
            for i, img_f in enumerate(imgs):
                img = cv2.imread(join(d, img_f))
                try:
                    img = cv2.resize(img, (28, 28))
                except:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, 0)
                img = np.expand_dims(img, 0)
                img = torch.from_numpy(img).float().to(self.device)
                pre_vec = self.net(img)
                pre_vec_array = pre_vec.cpu().detach().numpy()
                max_value = np.max(pre_vec_array)
                if max_value > self.confidence_threshold:
                    max_index = torch.Tensor.argmax(pre_vec)
                    if self.device.type == "cpu":
                        pre_label = int(max_index.detach().numpy())
                    else:
                        pre_label = int(max_index.cpu().numpy())
                    views_identified += 1
                else:
                    # the "others" type
                    pre_label = -1
                labels[img_f[:-4]] = pre_label
                views_handled += 1

            with open(output_file, mode='w+') as fc:
                fc.write(str(labels))
                k += 1
                print(f'\t({k}/{total}) {d}')
        end = perf_counter()
        print("time span:", end - start)
        if views_handled > 0:
            print(f"reidentified views: {views_identified}/{views_handled} "
                  f"({(views_identified / float(views_handled) * 100):.2f}%)")


def parse_arg_reclass(input_args: list):
    parser = argparse.ArgumentParser(
        description="Reidenfity UI controls based on their image features")
    parser.add_argument("dataset_path",
                        help="the path for view image dataset. "
                             "I get the view type names according to it")
    parser.add_argument("input_path", help="input path")
    parser.add_argument("--lr", "-l", default=0.003, type=float,
                        help="training learning rate of the model")
    parser.add_argument("--decay", "-d", default='4,0.1', type=str,
                        help="training learning rate decay of the model, "
                             "format: decay_epoch,decay_rate")
    parser.add_argument("--batch_size", "-b", default=128, type=int,
                        help="training batch size of the model")
    parser.add_argument("--epoch", "-e", default=12, type=int,
                        help="training epoch of the model")
    parser.add_argument("--threshold", "-t", default=0.95, type=float,
                        help="prediction confidence of the model")
    parser.add_argument("--retrain", "-r", action="store_true", default=False,
                        help="retrain and overwrite the existing model")
    parser.add_argument("--notskip", "-s", action="store_false",
                        help="do not skip the reidentified items")
    _args = parser.parse_args(input_args)
    return _args


if __name__ == '__main__':
    args = parse_arg_reclass(sys.argv[1:])

    try:
        lr_decay_e, lr_decay_r = args.decay.split(',')[0], args.decay.split(',')[1]
        lr_decay_e, lr_decay_r = int(lr_decay_e), float(lr_decay_r)
        ic = ImgClassifier(dataset_path=args.dataset_path,
                           epoch=args.epoch, batch_size=args.batch_size,
                           lr_init=args.lr, lr_decay=(lr_decay_e, lr_decay_r),
                           retrain_model=args.retrain,
                           confidence_threshold=args.threshold)

        # output predictions for elements imgs
        ic.predict(args.input_path, skip_existance=args.notskip)

    except ValueError:
        print("invalid decay for learning rate. example: 4,0.1")
        exit(1)
