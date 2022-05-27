import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data_utl
from I3D import InceptionI3d
import numpy as np
import utils
# from torchsummary import summary

class VGG(nn.Module):
    """
    VGG builder
    """
    def __init__(self, arch: object, num_classes=1000) -> object:
        super(VGG, self).__init__()
        self.in_channels = 1
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return F.softmax(self.fc3(out))

def VGG_11(num_classes):
    return VGG([1, 1, 2, 2, 2], num_classes=num_classes)

def VGG_13(num_classes):
    return VGG([1, 1, 2, 2, 2], num_classes=num_classes)

def VGG_16(num_classes):
    return VGG([2, 2, 3, 3, 3], num_classes=num_classes)

def VGG_19(num_classes):
    return VGG([2, 2, 4, 4, 4], num_classes=num_classes)


class JointDataset(data_utl.Dataset):

    def __init__(self):
        self.data = self.make_dataset()

    def __getitem__(self, item):
        try:
            rgb_path, flow_path, label = self.data[item]
            image = np.load(rgb_path).reshape([3, 79, 224, 224])
            flow = np.load(flow_path).reshape([2, 79, 224, 224])
            return torch.from_numpy(np.array(image).astype(np.float32)), torch.from_numpy(np.array(flow).astype(np.float32)), torch.from_numpy(np.array(label).astype(np.int64))
        except ValueError:
            print("error")
            return self.__getitem__(item+1)
        except FileNotFoundError:
            print("error")
            return self.__getitem__(item+1)

    def __len__(self):
        return len(self.data)

    def make_dataset(self):
        dataset = []

        with open("datas/flow_2.txt", 'r') as f:
            line = f.readline()
            while line != "":
                line = line.split(" ")
                flow_path = line[0]
                paths = flow_path.split('/')
                rgb_path = paths[0] + "/RGB/" + paths[2] + "/" + paths[3] + "/" + paths[4].split('_')[0] + "_rgb.npy"
                label = int(line[1]) + 1
                dataset.append((rgb_path, flow_path, label))
                line = f.readline()
        return dataset


def train(batchsize, savepath, device, epochs, labels=3, feature='cqcc'):
    train_dataset = utils.MatDataset("train", feature)
    val_dataset = utils.MatDataset("val", feature)

    VGG = VGG_11(labels)

    optimizer = optim.SGD(VGG.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)  # lr=0.0005, weight_decay=0.0000001
    # optimizer = optim.Adam(I3D.parameters(), lr=0.0001, weight_decay=0.0001)

    train_loss, train_acc, val_loss, val_acc = utils.trainModel(VGG, train_dataset, val_dataset, batchsize,
                                                                epochs=epochs, optimizer=optimizer, device=device,
                                                                save_path=savepath, save_name='video-' + feature)

    utils.saveAndDraw(train_loss, train_acc, val_loss, val_acc, savepath)

def test(batchsize, device, epochs=5, labels=3, feature='cqcc'):
    test_dataset = utils.MatDataset("test", feature)

    VGG = VGG_11(labels)
    VGG.load_state_dict(torch.load('modelWeight/IEMOCAP_cqcc/video-cqcc.pt'))

    utils.testModel(VGG, test_dataset, batchsize, epochs=epochs, device=device)

def test_joint(batchsize, device, epochs=5, labels=3):
    I3D_rgb = InceptionI3d(labels, in_channels=3)
    I3D_flow = InceptionI3d(labels, in_channels=2)
    VGG = VGG_11(labels)
    I3D_rgb.load_state_dict(torch.load('modelWeight/IEMOCAP_I3D_rgb_adam/I3D-rgb.pt'))
    I3D_flow.load_state_dict(torch.load('modelWeight/IEMOCAP_I3D_flow_adam/I3D-flow.pt'))
    VGG.load_state_dict(torch.load('modelWeight/IEMOCAP_cqcc/video-cqcc_38.pt'))

    Test_acc = []

    test_dataset = utils.JointDataset(multinum=3)
    test_dataloader = data_utl.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=10,
                                          pin_memory=True)
    devices = device if torch.cuda.is_available() else 'cpu'
    devices = torch.device(devices)

    I3D_rgb.to(devices)
    I3D_flow.to(devices)
    VGG.to(devices)

    for epoch in range(epochs):
        print('Epoch: {:.0f} / All {:.0f}'.format(epoch + 1, epochs))
        I3D_rgb.eval()
        I3D_flow.eval()
        VGG.eval()
        test_corrects = 0
        with torch.no_grad():
            for images, flows, videos, classes in test_dataloader:
                classes = classes.reshape([len(classes)])
                images = images.to(devices)
                flows = flows.to(devices)
                videos = videos.to(devices)
                classes = classes.to(devices)

                rgb_output = I3D_rgb(images)
                flow_output = I3D_flow(flows)
                video_output = VGG(videos)
                outputs = F.softmax(rgb_output + flow_output +video_output)
                _, preds = torch.max(outputs.data, 1)
                # statistics
                test_corrects += torch.sum(preds == classes.data)
                # print('Validating: No. ', i, ' process ... total: ', val_dataloader.__len__())
        test_acc = test_corrects.data.item() / test_dataset.__len__()
        Test_acc.append(test_acc)
        print('Test: Acc: {:.4f}'.format(test_acc))


def test_classes(batchsize, device, epochs=5, labels=3):
    I3D_rgb = InceptionI3d(labels, in_channels=3)
    I3D_flow = InceptionI3d(labels, in_channels=2)
    I3D_rgb.load_state_dict(torch.load('modelWeight/I3DTrain_rgb_adam/I3D-rgb.pt', map_location='cpu'))
    I3D_flow.load_state_dict(torch.load('modelWeight/I3DTrain_flow_adam/I3D-flow.pt', map_location='cpu'))

    test_dataset = JointDataset()
    test_dataloader = data_utl.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=10,
                                          pin_memory=True)
    devices = device if torch.cuda.is_available() else 'cpu'
    devices = torch.device(devices)

    I3D_rgb.to(devices)
    I3D_flow.to(devices)

    for epoch in range(epochs):
        I3D_rgb.eval()
        I3D_flow.eval()
        labels_z = 0
        labels_o = 0
        labels_t = 0
        with torch.no_grad():
            for images, flows, classes in test_dataloader:
                classes = classes.reshape([len(classes)])
                images = images.to(devices)
                flows = flows.to(devices)
                classes = classes.to(devices)

                rgb_output = I3D_rgb(images)
                flow_output = I3D_flow(flows)
                outputs = F.softmax((0.3 * rgb_output) + (0.7 * flow_output))
                _, preds = torch.max(outputs.data, 1)
                # statistics
                labels_z += torch.sum(preds == 0)
                labels_o += torch.sum(preds == 1)
                labels_t += torch.sum(preds == 2)
                # print('Validating: No. ', i, ' process ... total: ', val_dataloader.__len__())
        print(labels_z, ' , ', labels_o, ' , ', labels_t)

if __name__ == "__main__":
    batchsize = 15
    feature = "cqcc"
    savepath = 'modelWeight/IEMOCAP_' + feature + '/'
    device = 'cuda:0'
    epochs = 1
    labels = 3

    # train(batchsize, savepath, device, epochs, labels, feature)
    # test(batchsize, device, epochs, labels, feature)
    # test_joint(batchsize, device, epochs, labels)
    test_classes(batchsize, device, epochs, labels)
