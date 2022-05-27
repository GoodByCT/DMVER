import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import utils
from torchsummary import summary

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides, bias=False)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Bottleneck(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides, bias=False)
        self.conv3 = nn.Conv2d(num_channels, num_channels*expansion,
                               kernel_size=1, stride=1, bias=False)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels*expansion,
                                   kernel_size=1, stride=strides, bias=False)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels*expansion)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)

class ResNet(nn.Module):
    def __init__(self, blocks, res=False):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        if res:
            self.layer1 = self.make_reslayer(input_channels=64, num_channels=64, block=blocks[0], strides=1)
            self.layer2 = self.make_reslayer(input_channels=64, num_channels=128, block=blocks[1], strides=2)
            self.layer3 = self.make_reslayer(input_channels=128, num_channels=256, block=blocks[2], strides=2)
            self.layer4 = self.make_reslayer(input_channels=256, num_channels=512, block=blocks[3], strides=2)
        else:
            self.expansion = 4
            self.layer1 = self.make_layer(input_channels=64, num_channels=64, block=blocks[0], strides=1)
            self.layer2 = self.make_layer(input_channels=256, num_channels=128, block=blocks[1], strides=2)
            self.layer3 = self.make_layer(input_channels=512, num_channels=256, block=blocks[2], strides=2)
            self.layer4 = self.make_layer(input_channels=1024, num_channels=512, block=blocks[3], strides=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fl = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_reslayer(self, input_channels, num_channels, block, strides):
        layers = []

        layers.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=strides))
        for i in range(1, block):
            layers.append(Residual(num_channels, num_channels))

        return nn.Sequential(*layers)

    def make_layer(self, input_channels, num_channels, block, strides):
        layers = []

        layers.append(Bottleneck(input_channels, num_channels, use_1x1conv=True, strides=strides))
        for i in range(1, block):
            layers.append(Bottleneck(num_channels*self.expansion, num_channels))

        return nn.Sequential(*layers)

    def forward(self, X):
        outputs = []
        for i in X:
            i = self.conv1(i)

            i = self.layer1(i)
            i = self.layer2(i)
            i = self.layer3(i)
            i = self.layer4(i)

            i = self.avgpool(i)
            out = self.fl(i)
            outputs.append(out)
        outputs = torch.stack(outputs, 0)

        return outputs

#512
def ResNet18():
    return ResNet([2, 2, 2, 2], res=True)

def ResNet34():
    return ResNet([3, 4, 6, 3], res=True)

#2048
def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])

class ResLSTM(nn.Module):
    def __init__(self, classes=3, device='cpu'):
        super(ResLSTM, self).__init__()
        self.device = device
        self.resnet = ResNet18()
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        self.fc1 = nn.Linear(79*512, 512)
        self.fc2 = nn.Linear(512, classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def hinder_init(self, batchsize):
        h0 = torch.zeros(1, batchsize, 512).to(torch.device(self.device))
        c0 = torch.zeros(1, batchsize, 512).to(torch.device(self.device))
        return (h0, c0)

    def forward(self, X):
        inputs = self.resnet(X)
        hider = self.hinder_init(X.shape[0]) #X的batch
        X, _ = self.lstm(inputs, hider)
        X = X.reshape(X.size(0), -1)
        X = self.fc1(X)
        X = F.relu(X)
        X = F.dropout(X, p=0.3, training=self.training)
        return F.softmax(self.fc2(X))

    def replace_fc(self, classes):
        self.fc2 = nn.Linear(512, classes)


def train(batchsize, savepath, device, epochs, labels=3):
    train_dataset = utils.ImageDataset("train", "lstm")

    val_dataset = utils.ImageDataset("val", "lstm")

    reslstm = ResLSTM(classes=labels, device=device)

    # optimizer = optim.SGD(reslstm.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001)    #lr=0.0005, weight_decay=0.0000001
    optimizer = optim.Adam(reslstm.parameters(), lr=0.0001, weight_decay=0.0001)

    train_loss, train_acc, val_loss, val_acc = utils.trainModel(reslstm, train_dataset, val_dataset, batchsize,
                                                                epochs=epochs, optimizer=optimizer, device=device,
                                                                save_path=savepath, save_name='ResLSTM')

    utils.saveAndDraw(train_loss, train_acc, val_loss, val_acc, savepath)

def fixed(batchsize, savepath, device, epochs, labels=3):
    train_dataset = utils.ImageDataset("train", "lstm")
    val_dataset = utils.ImageDataset("val", "lstm")

    reslstm = ResLSTM(device=device)
    reslstm.load_state_dict(torch.load('modelWeight/ResLSTM/ResLSTM_40.pt'))

    for i in reslstm.parameters():
        i.requires_grad = False

    reslstm.replace_fc(labels)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, reslstm.parameters()), lr=0.0001, weight_decay=0.0001)

    train_loss, train_acc, val_loss, val_acc = utils.trainModel(reslstm, train_dataset, val_dataset, batchsize,
                                                                epochs=epochs, optimizer=optimizer, device=device,
                                                                save_path=savepath, save_name='ResLSTM')

    utils.saveAndDraw(train_loss, train_acc, val_loss, val_acc, savepath)
            
def test(batchsize, device, epochs=5, labels=3):
    test_dataset = utils.ImageDataset("test", "lstm")

    reslstm = ResLSTM(classes=labels, device=device)
    reslstm.load_state_dict(torch.load('modelWeight/IEMOCAP_ResLSTM/ResLSTM_200.pt'))

    utils.testModel(reslstm, test_dataset, batchsize, epochs, device)


if __name__ == "__main__":
    # net = ResLSTM()
    # # net.to(device=torch.device('cuda'))
    # X = torch.randn(79, 3, 224, 224)#.to(device=torch.device('cuda'))
    # Y = torch.randn(79, 3, 224, 224)#.to(device=torch.device('cuda'))
    # X = net((X, Y))
    # print(X)
    # summary(net, (79, 3, 224, 224))
    batchsize = 10
    savepath = 'modelWeight/IEMOCAP_ResLSTM/'
    device = 'cuda:0'
    labels = 4
    epochs = 5

    # train(batchsize, savepath, device, epochs, labels)
    test(batchsize, device, epochs, labels)
    # fixed(batchsize, savepath, device, epochs, labels)

