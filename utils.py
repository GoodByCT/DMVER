import os

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import h5py
import scipy.io as scio
import torch.utils.data as data_utl
import json
import matplotlib.pyplot as plt

### For DSVER ###
# _TXT_PATH = {
#     "train_rgb": "datas/train_rgb.txt",
#     "test_rgb": "datas/test_rgb.txt",
#     "val_rgb": "datas/val_rgb.txt",
#     "train_lstm": "datas/train_rgb.txt",
#     "test_lstm": "datas/test_rgb.txt",
#     "val_lstm": "datas/val_rgb.txt",
#     "train_flow": "datas/train_flow.txt",
#     "test_flow": "datas/test_flow.txt",
#     "val_flow": "datas/val_flow.txt",
# }
#
# _MAT_PATH = {
#     "train_cqcc": "datas/cache_train_cqcc.mat",
#     "val_cqcc": "datas/cache_val_cqcc.mat",
#     "test_cqcc": "datas/cache_test_cqcc.mat"
# }

### For Music-Video ###
# _TXT_PATH = {
#     "train_rgb": "../Music_Video/trainData/train_rgb.txt",
#     "test_rgb": "../Music_Video/testData/test_rgb.txt",
#     "val_rgb": "../Music_Video/valData/val_rgb.txt",
#      "train_lstm": "../Music_Video/trainData/train_rgb.txt",
#     "test_lstm": "../Music_Video/testData/test_rgb.txt",
#     "val_lstm": "../Music_Video/valData/val_rgb.txt",
#     "train_flow": "../Music_Video/trainData/train_flow.txt",
#     "test_flow": "../Music_Video/testData/test_flow.txt",
#     "val_flow": "../Music_Video/valData/val_flow.txt",
# }
#
# _MAT_PATH = {
#     "train_cqcc": "../Music_Video/trainData/cache_train_cqcc.mat",
#     "val_cqcc": "../Music_Video/valData/cache_val_cqcc.mat",
#     "test_cqcc": "../Music_Video/testData/cache_test_cqcc.mat"
# }
#
# _PRE_PATH = "../Music_Video/"

### For IEMOCAP ###
_TXT_PATH = {
    "train_rgb": "IEMOCAP/train_4/train_rgb.txt",
    "test_rgb": "IEMOCAP/test_4/test_rgb.txt",
    "val_rgb": "IEMOCAP/val_4/val_rgb.txt",
    "train_lstm": "IEMOCAP/train_4/train_rgb.txt",
    "test_lstm": "IEMOCAP/test_4/test_rgb.txt",
    "val_lstm": "IEMOCAP/val_4/val_rgb.txt",
    "train_flow": "IEMOCAP/train_4/train_flow.txt",
    "test_flow": "IEMOCAP/test_4/test_flow.txt",
    "val_flow": "IEMOCAP/val_4/val_flow.txt",
}

_MAT_PATH = {
    "train_cqcc": "IEMOCAP/train_4/cache_train_cqcc.mat",
    "val_cqcc": "IEMOCAP/val_4/cache_val_cqcc.mat",
    "test_cqcc": "IEMOCAP/test_4/cache_test_cqcc.mat"
}

_PRE_PATH = "IEMOCAP/"

# _TXT_PATH = {
#     "z_rgb": "datas/rgb_0.txt",
#     "o_rgb": "datas/rgb_1.txt",
#     "t_rgb": "datas/rgb_2.txt",
#     "z_flow": "datas/flow_0.txt",
#     "o_flow": "datas/flow_1.txt",
#     "t_flow": "datas/flow_2.txt",
# }


class ImageDataset(data_utl.Dataset):

    def __init__(self, mode, feature):
        self.mode = mode
        self.feature = feature
        self.data = self.make_dataset()

    def __getitem__(self, item):
        try:
            path, label = self.data[item]
            if self.feature == "rgb":
                data = np.load(path).reshape([3, 79, 224, 224])
            elif self.feature == "lstm":
                data = np.load(path).reshape([79, 3, 224, 224])
            elif self.feature == "flow":
                data = np.load(path).reshape([2, 79, 224, 224])
            else:
                image = np.load(path).reshape([3, 79, 224, 224])
                flow = np.load(path).reshape([2, 79, 224, 224])
                return torch.from_numpy(np.array(image).astype(np.float32)), torch.from_numpy(np.array(flow).astype(np.float32)), torch.from_numpy(np.array(label).astype(np.int64))

            return torch.from_numpy(np.array(data).astype(np.float32)), torch.from_numpy(np.array(label).astype(np.int64))
        except ValueError:
            return self.__getitem__(item+1)

    def __len__(self):
        return len(self.data)

    def make_dataset(self):
        dataset = []

        with open(_TXT_PATH[self.mode + '_' + self.feature], 'r') as f:
            line = f.readline()
            while line != "":
                # line = line.split(" ")
                # path = line[0]
                # label = int(line[1]) + 1
                line = line.split(" , ")
                path = line[0].split('\\')
                path = _PRE_PATH + path[0] + '/' + path[1] + '/' + path[2] + '/' + path[3]
                label = int(line[1])
                dataset.append((path, label))
                line = f.readline()
        return dataset


class JointDataset(data_utl.Dataset):

    def __init__(self, multinum):
        self.multinum = multinum
        self.data = self.make_dataset()

    def __getitem__(self, item):
        try:
            rgb_path, flow_path, label = self.data[item]
            if self.multinum == 2:
                image = np.load(rgb_path).reshape([3, 79, 224, 224])
                flow = np.load(flow_path).reshape([2, 79, 224, 224])
                return torch.from_numpy(np.array(image).astype(np.float32)), torch.from_numpy(np.array(flow).astype(np.float32)), torch.from_numpy(np.array(label).astype(np.int64))
            elif self.multinum == 3:
                filename, audio, label = self.data[item]
                # paths = filename.split('/')
                paths = filename.split('\\')
                # rgb_path = "datas/RGB/" + paths[1] + '/' + paths[2] + '/' + paths[3].split('.')[0] + "_rgb.npy"
                # flow_path = "datas/FLOW/" + paths[1] + '/' + paths[2] + '/' + paths[3].split('.')[0] + "_flow.npy"
                # rgb_path = _PRE_PATH + paths[0] + "/RGB/" + paths[2] + "/" + paths[3].split('.')[0] + "_rgb.npy"
                # flow_path = _PRE_PATH + paths[0] + "/Flow/" + paths[2] + "/" + paths[3].split('.')[0] + "_flow.npy"
                rgb_path = _PRE_PATH + paths[0] + "/RGB/" + paths[2] + "/" + paths[3].split('.')[0] + "_rgb.npy"
                flow_path = _PRE_PATH + paths[0] + "/Flow/" + paths[2] + "/" + paths[3].split('.')[0] + "_flow.npy"
                image = np.load(rgb_path).reshape([3, 79, 224, 224])
                flow = np.load(flow_path).reshape([2, 79, 224, 224])
                audio = audio[0].astype(np.float32)
                audio.resize(224, 224, refcheck=False)
                audio = audio.reshape([1, 224, 224])

                return torch.from_numpy(np.array(image).astype(np.float32)), torch.from_numpy(np.array(flow).astype(np.float32)), torch.from_numpy(audio), torch.from_numpy(np.array(label).astype(np.int64))

        except ValueError:
            return self.__getitem__(item+1)
        except FileNotFoundError:
            return self.__getitem__(item+1)

    def __len__(self):
        return len(self.data)

    def make_dataset(self):
        dataset = []

        if self.multinum == 2:
            with open(_TXT_PATH["test_flow"], 'r') as f:
                line = f.readline()
                while line != "":
                    line = line.split(" ")
                    flow_path = line[0]
                    paths = flow_path.split('/')
                    rgb_path = paths[0] + "/RGB/" + paths[2] + "/" + paths[3] + "/" + paths[4].split('_')[0] + "_rgb.npy"
                    label = int(line[1]) + 1
                    # line = line.split(" , ")
                    # flow_path = line[0]
                    # paths = flow_path.split('/')
                    # paths = flow_path.split('\\')
                    # rgb_path = paths[0] + "/RGB/" + paths[2] + "/" + paths[3].split('_flow')[0] + "_rgb.npy"
                    # flow_path = paths[0] + "/" + paths[1] + "/" + paths[2] + "/" + paths[3]
                    # rgb_path = _PRE_PATH + rgb_path
                    # flow_path = _PRE_PATH + flow_path
                    # label = int(line[1])
                    dataset.append((rgb_path, flow_path, label))
                    line = f.readline()
        elif self.multinum == 3:
            f = h5py.File(_MAT_PATH["test_cqcc"])
            for i in range(len(f['filename'][0])):
                data = np.array(f[f["data"][0][i]])
                name_ref = f[f["filename"][0][i]]
                id_ref = f[f["sys_id"][0][i]]
                filename = ''.join([chr(v[0]) for v in name_ref])
                # sys_id = int(''.join([chr(v[0]) for v in id_ref])) + 1
                sys_id = int([v for v in id_ref][0])
                dataset.append((filename, data, np.array(sys_id)))
            f.close()
            # f = scio.loadmat(_MAT_PATH["test_cqcc"])
            # for i in range(len(f['filename'])):
            #     dataset.append((f['filename'][i], f['data'][i], f['sys_id'][i]))

        return dataset


class MatDataset(data_utl.Dataset):

    def __init__(self, mode, feature):
        self.mode = mode
        self.feature = feature
        self.data = self.make_dataset()

    def __getitem__(self, item):
        audio, label = self.data[item]
        audio = audio[0].astype(np.float32)
        audio.resize(224, 224, refcheck=False)
        audio = audio.reshape([1, 224, 224])

        return torch.from_numpy(audio), torch.from_numpy(label.astype(np.int64))

    def __len__(self):
        return len(self.data)

    def make_dataset(self):
        dataset = []

        f = h5py.File(_MAT_PATH[self.mode + '_' + self.feature])
        for i in range(len(f['filename'][0])):
            data = np.array(f[f["data"][0][i]])
            ref = f[f["sys_id"][0][i]]
            # sys_id = int(''.join([chr(v[0]) for v in ref])) + 1
            sys_id = int([v for v in ref][0])
            dataset.append((data, np.array(sys_id)))
        f.close()
        # f = scio.loadmat(_MAT_PATH[self.mode + '_' + self.feature])
        # for i in range(len(f['filename'])):
        #     dataset.append((f['data'][i], f['sys_id'][i]))

        return dataset


def saveAndDraw(train_loss, train_acc, val_loss, val_acc, save_path='modelWeight/'):
    dict = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }

    with open(save_path+'/train.json', 'a') as f:
        f.write(json.dumps(dict, ensure_ascii=False, indent=2))

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(dict['train_loss'], color='g', label='train loss', linestyle='-')
    ax1.plot(dict['val_loss'], color='b', label='val loss', linestyle='-')

    ax2.plot(dict['train_acc'], color='y', label='train acc', linestyle='-')
    ax2.plot(dict['val_acc'], color='r', label='val acc', linestyle='-')

    ax1.legend(loc=(0.7, 0.9))  # 使用二元组(0.7,0.8)定义标签位置
    ax2.legend(loc=(0.7, 0.7))

    ax1.set_xlabel('steps')  # 设置X轴标签
    ax1.set_ylabel('loss')  # 设置Y1轴标签
    ax2.set_ylabel('accuracy')  # 设置Y2轴标签
    plt.savefig(save_path+'/train.jpg', dpi=1080)  # 将图像保存到out_fig_path路径中，分辨率为100


def trainModel(model, train_dataset, val_dataset, batchsize, epochs=1, optimizer=None, device='cuda', save_path='modelWeight/', save_name='trainmodel'):
    allT_loss = []
    allT_acc = []
    allV_loss = []
    allV_acc = []

    train_dataloader = data_utl.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=10, pin_memory=True)
    val_dataloader = data_utl.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=10, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20])
    # model = nn.DataParallel(model, device_ids=[0, 1, 2]).module   #Multi-GPU
    devices = device if torch.cuda.is_available() else 'cpu'
    devices = torch.device(devices)

    model.to(devices)

    for epoch in range(epochs):
        print('Epoch: {:.0f} / All {:.0f}'.format(epoch + 1, epochs))
        model.train()
        train_loss = 0.0
        train_corrects = 0
        for inputs, classes in train_dataloader:
            classes = classes.reshape([len(classes)])
            inputs = inputs.to(devices)
            classes = classes.to(devices)

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, classes)
            loss.backward()
            optimizer.step()
            lr_sched.step()
            _, preds = torch.max(outputs.data, 1)
            # statistics
            train_loss += loss.data.item()
            train_corrects += torch.sum(preds == classes.data)
            # print('Training: No. ', count, ' process ... total: ', train_dataloader.__len__())
        train_loss = train_loss / train_dataloader.__len__()
        train_acc = train_corrects.data.item() / train_dataset.__len__()
        allT_loss.append(train_loss)
        allT_acc.append(train_acc)
        print('Train: Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, classes in val_dataloader:
                classes = classes.reshape([len(classes)])
                inputs = inputs.to(devices)
                classes = classes.to(devices)

                outputs = model(inputs)
                loss = criterion(outputs, classes)
                _, preds = torch.max(outputs.data, 1)
                # statistics
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == classes.data)
                # print('Validating: No. ', i, ' process ... total: ', val_dataloader.__len__())
        val_loss = val_loss / val_dataloader.__len__()
        val_acc = val_corrects.data.item() / val_dataset.__len__()
        allV_loss.append(val_loss)
        allV_acc.append(val_acc)
        print('Val: Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), save_path + save_name + '_' + str(epoch + 1) + '.pt')

    return allT_loss, allT_acc, allV_loss, allV_acc

def testModel(model, test_dataset, batchsize, epochs=1, device='cuda'):
    Test_acc = []

    test_dataloader = data_utl.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=10, pin_memory=True)
    devices = device if torch.cuda.is_available() else 'cpu'
    devices = torch.device(devices)

    model.to(devices)

    for epoch in range(epochs):
        print('Epoch: {:.0f} / All {:.0f}'.format(epoch + 1, epochs))
        model.eval()
        test_corrects = 0
        with torch.no_grad():
            for inputs, classes in test_dataloader:
                classes = classes.reshape([len(classes)])
                inputs = inputs.to(devices)
                classes = classes.to(devices)

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # statistics
                test_corrects += torch.sum(preds == classes.data)
                # print('Validating: No. ', i, ' process ... total: ', val_dataloader.__len__())
        test_acc = test_corrects.data.item() / test_dataset.__len__()
        Test_acc.append(test_acc)
        print('Test: Acc: {:.4f}'.format(test_acc))



