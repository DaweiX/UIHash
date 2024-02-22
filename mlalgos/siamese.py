"""Calculate similarity score via a Siamese network"""

import os.path
import time
from typing import Tuple

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from network import *
import progressbar as bar
from sys import stdout
import torch
from time import perf_counter
from dataset import LabelledDataSet
from os.path import exists
from os import makedirs
from shutil import copyfile
from os.path import join
from sklearn import metrics
import sys
import argparse

curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
if rootpath not in sys.path:
    sys.path.append(rootpath)

from util.util_draw import draw_roc, draw_history


class SiameseModel:

    def __init__(
            self, hash_size: Tuple[int, int, int] = (10, 5, 5),
            lr_init: float = 0.001,
            lr_decay: Tuple[int, float] = (10, 0.1),
            epoch: int = 5, batch_size: int = 32,
            binary_loss: bool = False,
            retrain_model: bool = False,
            load_labelled_dataset: bool = True):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu")
        print(f'DEVICE: {self.device}')
        self.epoch = epoch
        self.binary_loss = binary_loss
        self.batch_size = batch_size
        self.hash_size = hash_size
        if hash_size[1] == 10 and hash_size[2] == 10:
            cnn = NNParas(self.hash_size[0]).cnn10x10
            fc = NNParas(self.hash_size[0]).fc10x10
        elif hash_size[1] == 5 and hash_size[2] == 5:
            cnn = NNParas(self.hash_size[0]).cnn5x5
            fc = NNParas(self.hash_size[0]).fc5x5
        elif hash_size[1] == 4 and hash_size[2] == 3:
            cnn = NNParas(self.hash_size[0]).cnn4x3
            fc = NNParas(self.hash_size[0]).fc4x3
        elif hash_size[1] == 3 and hash_size[2] == 3:
            cnn = NNParas(self.hash_size[0]).cnn3x3
            fc = NNParas(self.hash_size[0]).fc3x3
        elif hash_size[1] == 2 and hash_size[2] == 2:
            cnn = NNParas(self.hash_size[0]).cnn2x2
            fc = NNParas(self.hash_size[0]).fc2x2
        elif hash_size[1] == 1 and hash_size[2] == 1:
            cnn = NNParas(self.hash_size[0]).cnn1x1
            fc = NNParas(self.hash_size[0]).fc1x1
        else:
            raise NotImplementedError(f"No CNN Model for Grid {hash_size[1]}x{hash_size[2]}")
        self.net = SiameseNet(cnn, fc,
                              binary_loss).to(self.device)
        self.criterion = torch.nn.BCELoss() if binary_loss \
            else CosLoss(pos_label=1, device=self.device)

        self.lr = lr_init
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        if load_labelled_dataset:
            self.set_repack = LabelledDataSet(reshape=True, hash_size=hash_size)
            s1 = int(0.8 * len(self.set_repack))
            s2 = int(0.9 * len(self.set_repack))
            index_list = list(range(len(self.set_repack)))
            self.train_idx, self.valid_idx, self.test_idx = \
                index_list[:s1], index_list[s1:s2], index_list[s2:]

            self.tr_samp = sampler.SubsetRandomSampler(self.train_idx)
            self.val_samp = sampler.SubsetRandomSampler(self.valid_idx)

        self.root_path = join(os.path.abspath(os.path.dirname(__file__)), "..")
        self.model_path = join(self.root_path, "models",
                               f"siamese_e{self.epoch}_{self.batch_size}_"
                               f"{hash_size[1]}x{hash_size[2]}.tar")
        self.retrain_model = retrain_model
        self.hash_size = hash_size
        self.model_ready = False
        model_root_path = join(self.root_path, "models")
        if not exists(model_root_path):
            makedirs(model_root_path)
        if not self.retrain_model and exists(self.model_path):
            self.net.load_state_dict(torch.load(self.model_path,
                                                map_location=self.device))
            print("model loaded")
            self.model_ready = True

    def deal_data(self, data):
        i1, i2, label = data
        i1, i2, label = i1.float(), i2.float(), label.float()
        i1, i2, label = i1.to(self.device), i2.to(self.device), label.to(self.device)
        return i1, i2, label

    def _forward(self, _i1, _i2):
        _i = torch.stack((_i1, _i2), 0)
        _o = self.net(_i)
        r = int(_o.shape[0] / 2)
        _o1, _o2 = _o[:r, :], _o[r:, :]
        return _o1, _o2

    def train(self, trainloader, validloader,
              drawhistory: bool = False):
        # training & validating
        loss_tr, loss_val = [], []
        start = time.perf_counter()
        for e in range(self.epoch):
            train_loss, valid_loss = [], []
            self.net.train()
            pbar = bar.ProgressBar(
                widgets=[
                    f'Training ({e + 1}/{self.epoch}) ',
                    bar.Percentage(),
                    ' ', bar.Bar('=')],
                fd=stdout,
                maxval=len(trainloader))
            pbar.start()
            for batch_idx, data in enumerate(trainloader):
                pbar.update(batch_idx + 1)
                i1, i2, label = self.deal_data(data)
                self.optimizer.zero_grad()
                if self.binary_loss:
                    i = np.vstack((i1, i2))
                    output_prob, o1, o2 = self.net(i)
                    loss = self.criterion(o1, o2, label)
                else:
                    output1, output2 = self._forward(i1, i2)
                    loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            pbar.finish()
            self.net.eval()
            pbar2 = bar.ProgressBar(
                widgets=[
                    f'Validating ({e + 1}/{self.epoch}) ',
                    bar.Percentage(),
                    ' ',
                    bar.Bar('=')],
                fd=stdout,
                maxval=len(validloader))
            pbar2.start()
            for batch_idx, data in enumerate(validloader):
                pbar2.update(batch_idx + 1)
                i1, i2, label = self.deal_data(data)
                if self.binary_loss:
                    output_prob, o1, o2 = self.net(i1, i2)
                    loss = self.criterion(o1, o2, label)
                else:
                    output1, output2 = self._forward(i1, i2)
                    loss = self.criterion(output1, output2, label)
                valid_loss.append(loss.item())

            pbar2.finish()
            self.lr = self.lr_init * (self.lr_decay[1] ** int(e / self.lr_decay[0]))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            _loss_train = np.round(np.mean(train_loss), 3)
            _loss_val = np.round(np.mean(valid_loss), 3)

            loss_tr.append(_loss_train)
            loss_val.append(_loss_val)
            print('Train Loss', _loss_train, 'Valid Loss', _loss_val, 'lr', self.lr)

        end = time.perf_counter()
        print("training time cost:", end - start)
        
        # draw loss plot
        if drawhistory:
            draw_history(self.epoch, loss_tr, loss_val, True)

    def test(self, dataloader, threshold: float, showbar: bool = False):
        pset, tset = [], []
        fplist, fnlist = [], []
        tplist = []
        pbar = None
        self.net.eval()
        sum_correct = 0
        sum_total = 0
        if showbar:
            pbar = bar.ProgressBar(
                widgets=[
                    'Testing ',
                    bar.Percentage(),
                    ' ',
                    bar.Bar('*')],
                fd=stdout,
                maxval=len(dataloader))
            pbar.start()
        for batch_idx, data in enumerate(dataloader):
            ui1, ui2, target = self.deal_data(data)
            batch_size = len(data[2])
            if showbar:
                pbar.update(batch_idx + 1)
            if self.binary_loss:
                output_prob, _, _ = self.net(ui1, ui2)
                darray = [i.item() for i in output_prob]
            else:
                ui = torch.stack((ui1, ui2), 0)
                o = self.net(ui)
                output1, output2 = o[:, 0, :], o[:, 1, :]
                distance = torch.cosine_similarity(output1, output2)
                darray = [i.item() for i in distance]

            _positivep = [0 if i < 0 else i for i in darray]
            pset.extend(_positivep)
            output_labels = [1 if i > threshold else 0 for i in darray]
            tset.extend([t.item() for t in target])

            sum_total += target.size(0)
            tarray = target.cpu().numpy().tolist()
            a = len([i for i in range(len(tarray)) if
                     output_labels[i] == tarray[i]])
            fn = [i + batch_size * batch_idx for i in range(len(tarray))
                  if (tarray[i] == 1 and output_labels[i] == 0)]
            fp = [i + batch_size * batch_idx for i in range(len(tarray))
                  if (tarray[i] == 0 and output_labels[i] == 1)]
            tp = [i + batch_size * batch_idx for i in range(len(tarray))
                  if (tarray[i] == 1 and output_labels[i] == 1)]

            fnlist.extend([i + self.test_idx[0] for i in fn])
            fplist.extend([i + self.test_idx[0] for i in fp])
            tplist.extend([i + self.test_idx[0] for i in tp])

            sum_correct += a

        print(
            f'acc for {sum_total} samples: {(sum_correct / sum_total * 100):.2f}%')

        if showbar:
            pbar.finish()

        return pset, tset, fnlist, fplist, tplist

    def train_and_test(self, threshold: float,
                       retrain: bool = False, drawroc: bool = False):

        trainloader = DataLoader(self.set_repack,
                                 batch_size=self.batch_size,
                                 sampler=self.tr_samp)
        validloader = DataLoader(self.set_repack,
                                 batch_size=self.batch_size,
                                 sampler=self.val_samp)
        testloader = DataLoader(torch.utils.data.Subset(
            self.set_repack, self.test_idx),
            batch_size=self.batch_size)
        if not self.model_ready:
            self.train(trainloader, validloader, drawhistory=drawroc)
            if not exists(self.model_path) or \
                    (exists(self.model_path) and retrain):
                torch.save(self.net.state_dict(), self.model_path)
                print("model saved")

        # predicting
        pset, tset = [], []
        sum_total = 0
        correct = 0
        out_labels = []
        fns, fps = [], []
        start = perf_counter()
        for batch_idx, data in enumerate(testloader):
            if batch_idx < len(testloader) - 1:
                sample_indices = self.test_idx[batch_idx * self.batch_size:
                                               (batch_idx + 1) * self.batch_size]
            else:
                sample_indices = self.test_idx[batch_idx * self.batch_size:]

            ui1, ui2, target = self.deal_data(data)
            ui = torch.stack((ui1, ui2), 0)

            if self.binary_loss:
                output_prob, _, _ = self.net(ui)
                darray = [i.item() for i in output_prob]
            else:
                o1, o2 = self._forward(ui1, ui2)
                o1 = torch.squeeze(o1, 0)
                o2 = torch.squeeze(o2, 0)
                distance = torch.cosine_similarity(o1, o2)
                darray = [i.item() for i in distance]

            pset.extend([0 if i < 0 else i for i in darray])

            output_lables = [1 if i > threshold else 0 for i in darray]

            tarray = [t.item() for t in target]
            tset.extend(tarray)
            out_labels.extend(output_lables)

            # false samples
            fn = [sample_indices[i] for i in range(len(tarray))
                  if (tarray[i] == 1 and output_lables[i] == 0)]
            fp = [sample_indices[i] for i in range(len(tarray))
                  if (tarray[i] == 0 and output_lables[i] == 1)]
            fns.extend(fn)
            fps.extend(fp)           
            sum_total += target.size(0)
            correct += len([i for i in range(len(target)) if
                            output_lables[i] == target[i]])
        stop = perf_counter()
        print("prediction time cost:", stop - start)
        fpr, tpr, _ = metrics.roc_curve(tset, pset, pos_label=1)
        if drawroc:
            draw_roc(pset, tset, threshold, print_prf=False)

        auc = metrics.auc(fpr, tpr)
        p = metrics.precision_score(tset, out_labels, pos_label=1)
        r = metrics.recall_score(tset, out_labels, pos_label=1)
        f1 = metrics.f1_score(tset, out_labels, pos_label=1)
        print(f"p: {p}\nr: {r}\nf1: {f1}\nauc: {auc}")

    def evaluate_on_labelled_dataset(self, npz_prefix: str, threshold: float,
                                     drawroc: bool = False):
        test_dataset = LabelledDataSet(reshape=True, npz_prefix=npz_prefix,
                                       hash_size=self.hash_size, shuffle_data=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        self.net.load_state_dict(torch.load(self.model_path,
                                            map_location=self.device))
        print("model loaded")

        # predicting
        pset, tset = [], []
        sum_total = 0
        out_labels = []
        for data in test_loader:
            print("predicting on", test_loader.batch_size, "samples")
            ui1, ui2, target = self.deal_data(data)
            o1, o2 = self._forward(ui1, ui2)
            o1 = torch.squeeze(o1, 0)
            o2 = torch.squeeze(o2, 0)
            distance = torch.cosine_similarity(o1, o2)
            darray = [i.item() for i in distance]
            _positivep = [0 if i < 0 else i for i in darray]
            pset.extend(_positivep)
            output_lables = [1 if i > threshold else 0 for i in darray]
            tarray = [t.item() for t in target]
            tset.extend(tarray)
            out_labels.extend(output_lables)
            sum_total += target.size(0)

            fpr, tpr, _ = metrics.roc_curve(tset, pset, pos_label=1)
            print(len(tset), tset)
            print(len(pset), pset)
            fns = [i for i in range(len(tset))
                   if out_labels[i] == 0 and tset[i] == 1]
            fps = [i for i in range(len(tset))
                   if out_labels[i] == 1 and tset[i] == 0]
            auc = metrics.auc(fpr, tpr)
            print("\t\tp\t\tr\t\tf1\t\tauc")
            p = metrics.precision_score(tset, out_labels, pos_label=1)
            r = metrics.recall_score(tset, out_labels, pos_label=1)
            f1 = metrics.f1_score(tset, out_labels, pos_label=1)
            print(f'\t\t{p:.5f}\t\t{r:.5f}\t\t{f1:.5f}\t\t{auc:.5f}')
            print("fns:", fns, "\nfps:", fps)
            if drawroc:
                draw_roc(pset, tset, threshold, print_prf=False)

    def detect_on_wild_dataset(self, dataset, threshold: float,
                               batch_size: int = 1024, save_score: bool = False):
        test_loader = DataLoader(dataset, batch_size=batch_size)
        self.net.load_state_dict(torch.load(self.model_path,
                                            map_location=self.device))
        print("model loaded")

        # network forwarding
        out_labels = []
        scores = []
        print("detecting on", len(dataset), "paris")
        t1 = perf_counter()
        for i, data in enumerate(test_loader):
            print(i)
            ui1, ui2, _ = self.deal_data(data)
            ui = torch.stack((ui1, ui2), 0)
            o = self.net(ui)
            row = int(o.shape[0] / 2)
            o1, o2 = o[:row, :], o[row:, :]
            o1 = torch.squeeze(o1, 0)
            o2 = torch.squeeze(o2, 0)
            distance = torch.cosine_similarity(o1, o2)
            if not save_score:
                output_lables = [1 if i.item() > threshold else 0 for i in distance]
                out_labels.extend(output_lables)
            else:
                scores.extend([i.item() for i in distance])

        t2 = perf_counter()
        print(f"done in {t2 - t1} s")
        if not save_score:
            np.save(join(self.root_path, "output", "dataset",
                         f"{dataset.dataset_name}_label.npy"),
                    out_labels, allow_pickle=True)
        else:
            np.save(join(self.root_path, "output", "dataset",
                         f"{dataset.dataset_name}_score.npy"),
                    scores, allow_pickle=True)


def mycopy(items, ui_path, out_path, sim_list):
    def _copy(pkg, xml, dst, _i, reverse, uipath):
        xml = xml.replace('.xml', '')
        src1 = f'{uipath}opt_original_apk/{pkg}/{xml}.jpg'
        src2 = f'{uipath}opt_repackage_apk/{pkg}/{xml}.jpg'
        src = src1 if exists(src1) else src2
        src_txt = src.replace('.jpg', '')
        if not exists(f'{dst}/{_i}'):
            makedirs(f'{dst}/{_i}')
            # print(_i)

        filename = f'{pkg}@{xml}' if reverse else f'{xml}@{pkg}'
        if filename in sim_list:
            return 1
        copyfile(src_txt + '/classify.txt', f'{dst}/{_i}/{filename}.txt')
        copyfile(src, f'{dst}/{_i}/{filename}.jpg')
        copyfile(src.replace('.jpg', '.xml'), f'{dst}/{_i}/{filename}.xml')
        return 0

    next_index = len(sim_list)
    i = 0
    for item in items:
        pkg1, pkg2, xml1, xml2, hash1, hash2 = item
        if f'{xml1}@{pkg1} {xml2}@{pkg2}' in sim_list:
            return 1
        else:
            _copy(pkg1, xml1, out_path, i + next_index, False, ui_path)
            _copy(pkg2, xml2, out_path, i + next_index, False, ui_path)
            with open(f'{out_path}/{i}/{i}.txt', mode='a+') as ftxt:
                ftxt.write(f'{pkg1}@{xml1}\n')
                ftxt.write(str(hash1))
                ftxt.write('\n')
                ftxt.write(f'{pkg2}@{xml2}\n')
                ftxt.write(str(hash2))
            i += 1
    print(i, 'sim pairs added')


def copyout_sim(npz: str, ui_path: str, out_path_sim: str):
    tps = np.load(npz, allow_pickle=True)['tps']
    sim_list = analysis_sim_folder(out_path_sim)
    mycopy(tps, ui_path, out_path_sim, sim_list)


def analysis_sim_folder(out_path_sim: str) -> np.array:
    db_file = f'{out_path_sim}/000_db.npy'
    if exists(db_file):
        return np.load(db_file, allow_pickle=True)
    xmls = []
    from os import listdir
    indecies = listdir(out_path_sim)
    if '000_db.npy' in indecies:
        indecies.remove('000_db.npy')
    for i in indecies:
        files = listdir(
            f'{out_path_sim}/{i}'.replace('/\\', '/').replace('//', '/'))
        files = [ff for ff in files if ff.endswith('.xml')]
        xmls.append('#'.join(files))
    np.save(db_file, xmls, allow_pickle=True)
    return xmls


def copyout_false(npz: str, ui_path: str,
                  out_path_false: str):
    fps = np.load(npz, allow_pickle=True)['fps']
    fns = np.load(npz, allow_pickle=True)['fns']
    fps_path = out_path_false + 'false positive'
    fns_path = out_path_false + 'false negative'
    mycopy(fps, ui_path, fps_path, [])
    mycopy(fns, ui_path, fns_path, [])


def parse_arg_siamese(input_args: list):
    parser = argparse.ArgumentParser(
        description="Run the siamese network on a dataset")
    parser.add_argument("--Repack", "-R", action="store_true",
                        help="use repack dataset")
    parser.add_argument("--dataset_name", "-dn", type=str, default="",
                        help="if not use repack dataset, then assign another "
                             "dataset name. make sure the hash files exist in output/hash")
    parser.add_argument("--lr", "-l", default=0.001, type=float,
                        help="training learning rate of the model")
    parser.add_argument("--decay", "-d", default='10,0.1', type=str,
                        help="training learning rate decay of the model, "
                             "format: decay_epoch,decay_rate")
    parser.add_argument("--batch_size", "-b", default=32, type=int,
                        help="training batch size of the model")
    parser.add_argument("--epoch", "-e", default=36, type=int,
                        help="training epoch of the model")
    parser.add_argument("--threshold", "-t", default=0.6, type=float,
                        help="the threshold to determine whether a pair is similar. "
                             "note that if detect on a wild dataset, than the threshold "
                             "also serves for filtering similar UIs in each app")
    parser.add_argument("--retrain", "-r", action="store_true", default=False,
                        help="retrain and overwrite the existing model")
    parser.add_argument("--notskip", "-s", action="store_false",
                        help="do not skip the reidentified items")
    parser.add_argument("--figure", "-f", action="store_true",
                        help="draw and show roc figure")
    parser.add_argument("--hash_size", "-hs", default='10,5,5', type=str,
                        help="shape of UI#. format: channel,tick_horizontal,tick_vertical")
    _args = parser.parse_args(input_args)
    return _args


if __name__ == '__main__':
    args = parse_arg_siamese(sys.argv[1:])
    hash_shape = None
    decay = None
    try:
        c, h, v = args.hash_size.split(',')
        c, h, v = int(c), int(h), int(v)
        hash_shape = (c, h, v)
    except ValueError:
        print("invalid hash size. example: 10,5,5")
        exit(1)

    try:
        d_e, d_r = args.decay.split(',')
        d_e, d_r = int(d_e), float(d_r)
        decay = (d_e, d_r)
    except ValueError:
        print("invalid decay. example: 10,0.1")
        exit(1)

    sm = SiameseModel(
        epoch=args.epoch,
        lr_decay=decay,
        batch_size=args.batch_size,
        lr_init=args.lr,
        retrain_model=args.retrain,
        hash_size=hash_shape,
        load_labelled_dataset=args.Repack
    )
    if args.Repack:
        sm.train_and_test(args.threshold, drawroc=args.figure)
    else:
        if len(args.dataset_name) == 0:
            print("please provide a dataset name when not using repack")
            exit(1)
        else:
            from dataset import WildDataSet
            wd = WildDataSet("rmv", hash_size=hash_shape, threshold=args.threshold,
                             siamese_model=sm, reshape=True)
            sm.detect_on_wild_dataset(wd, threshold=args.threshold,
                                      batch_size=args.batch_size,
                                      save_score=True)
