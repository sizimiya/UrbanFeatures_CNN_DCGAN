import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd


batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CsvDataset(Dataset):
    def __init__(self):
        super(CsvDataset, self).__init__()
        self.feature_path = 'data_trans.csv'
        self.label_path = 'data_label_5.csv'
        feature_df_ = pd.read_csv(self.feature_path)
        label_df_ = pd.read_csv(self.label_path)
        assert feature_df_.columns.tolist()[1:] == label_df_[label_df_.columns[0]].tolist(), \
            'feature name does not match label name'
        self.feature = [feature_df_[i].tolist() for i in feature_df_.columns[1:]]
        self.label = label_df_[label_df_.columns[1]]
        assert len(self.feature) == len(self.label)
        self.length = len(self.feature)

    def __getitem__(self, index):
        x = self.feature[index]
        x = torch.Tensor(x)
        x = x.reshape(1, 224, 224)

        y = self.label[index]

        return x, y

    def __len__(self):
        return self.length

trainset = CsvDataset()

trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

#_____________________测试集____________________________________________________________________________________________________

class CsvDataset_test(Dataset):
    def __init__(self):
        super(CsvDataset_test, self).__init__()
        self.feature_path = 'data_test.csv'
        self.label_path = 'data_label_5_test.csv'
        feature_df_ = pd.read_csv(self.feature_path)
        label_df_ = pd.read_csv(self.label_path)
        assert feature_df_.columns.tolist()[1:] == label_df_[label_df_.columns[0]].tolist(), \
            'feature name does not match label name'
        self.feature = [feature_df_[i].tolist() for i in feature_df_.columns[1:]]
        self.label = label_df_[label_df_.columns[1]]
        assert len(self.feature) == len(self.label)
        self.length = len(self.feature)

    def __getitem__(self, index):
        xt = self.feature[index]
        xt = torch.Tensor(xt)
        xt = xt.reshape(1, 224, 224)

        yt = self.label[index]

        return xt, yt

    def __len__(self):
        return self.length

testset = CsvDataset_test()

testloader = DataLoader(dataset=testset, batch_size=16, shuffle=True)

#------------------------------------------------------------------------------------------------

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()  # 参数初始化

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历各个层进行参数初始化
            if isinstance(m, nn.Conv2d):  # 如果是卷积层的话 进行下方初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # 正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 如果偏置不是0 将偏置置成0  相当于对偏置进行初始化
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                nn.init.xavier_uniform_(m.weight)  # 也进行正态分布初始化
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  # 将所有偏执置为0


def make_features(cfg: list):
    layers = []
    in_channels = 1                       #******************************************************************
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# cfgs = {
#     'vgg11': [64, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg13': [64, 64, 128, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg16': [64, 64, 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'vgg19': [64, 64, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


def vgg(model_name="vgg11", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model

#------------------------------------------------------------------------------------------------


loss2 = nn.CrossEntropyLoss()  # loss2=NLLLoss
D = vgg(num_classes=5, init_weights=True)
D = D.to(device)
d_optimizer = optim.Adam(D.parameters(), lr=0.0001)


count = 0
epoch = 200
iter_count = 0

D_loss_real_cls_all = []
d_real_train_accuracy_all = []

test_loss_all = []  # 存放测试集损失的数组
test_accur_all = []  # 存放测试集准确率的数组

iter_count_all = []

for i in range(epoch):
    train_bar = tqdm(trainloader)
    train_num = 0.0

    D_loss_real_cls_epoch = 0
    d_real_train_accuracy = 0.0

    for (x, y) in train_bar:
        D.train()
        img = x
        label = y

        labels_onehot = np.zeros((img.shape[0], 10))           #28行，10列
        labels_onehot[np.arange(img.shape[0]), label.numpy()] = 1     #根据or_data中label的数字，将对应数字序列的位置换成1

        img = img.to(device)
        img = Variable(img)

        d_optimizer.zero_grad()  # 判别器D的梯度归零

        # 类标签
        real_cls_label = Variable(torch.from_numpy(labels_onehot).float()).to(device)  # 真的类别label相应为1  100*15(10)

        # 真图片的损失
        real_cls_out = D(img)  # 真图片送入判别器D  得到真假输出 100*1 和分类输出100*15(10)

        # 真图片的分类损失
        input_a = real_cls_out.to(device)
        target_a = label.to(device)
        d_loss_real_cls = loss2(input_a, target_a)

        #真实图片准确度
        input_a_acc = torch.argmax(input_a, 1)
        d_real_accuracy = torch.sum(input_a_acc == target_a)  # outputs == target的 即使预测正确的，统计预测正确的个数,从而计算准确率
        d_real_train_accuracy = d_real_train_accuracy + d_real_accuracy  # 求训练集的准确率

        d_loss_real_cls.backward()
        d_optimizer.step()  # 更新判别器D参数

        train_num += img.size(0)
        iter_count += 1
        iter_count_all.append(iter_count)

        with torch.no_grad():                              #loss在每个epoch训练完后后重置
            D_loss_real_cls_epoch += d_loss_real_cls.cpu().item()

    print(f"epoch：{i}， "
          f"d_real_train_accuracy：{d_real_train_accuracy / train_num}, ")

    d_real_train_accuracy_all.append(d_real_train_accuracy.double().item() / train_num)  # 将训练的损失放到一个列表里 方便后续画图

    # 求平均损失
    # with torch.no_grad():
    train_num = train_num/batch_size
    D_loss_real_cls_epoch /= train_num
    D_loss_real_cls_all.append(D_loss_real_cls_epoch)

    print(f"epoch:{i},\n"
          f"D_loss_real_cls:{D_loss_real_cls_epoch},"
          )

    #测试集
    test_loss = 0  # 同上 测试损失
    test_accuracy = 0.0  # 测试准确率
    test_num = 0
    D.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        for (x, y) in testloader:
            t_img = x.to(device)
            t_label = y.to(device)

            t_cls_out = D(t_img)  # 真图片送入判别器D  得到真假输出 100*1 和分类输出100*15(10)
            t_input = t_cls_out
            t_target = t_label
            test_loss = loss2(t_input, t_target)
            outputs = torch.argmax(t_input, 1)
            test_loss = test_loss + abs(test_loss.item()) * t_img.size(0)
            t_accuracy = torch.sum(outputs == t_label)
            test_accuracy = test_accuracy + t_accuracy
            test_num += t_img.size(0)

    print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
    test_loss_all.append(test_loss.double().item() / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)


    # 模型保存（10 epoch）
    if (i % 10 == 0) and (i != 0):
        print(i)
        torch.save(D.state_dict(), r'./ACGAN_model_save/Discriminator_cuda_%d.pkl' % i)


data_loss = zip(iter_count_all,
                D_loss_real_cls_all,
                d_real_train_accuracy_all,
                test_loss_all,
                test_accur_all)
header_name = ['iter_count',
               'D_loss_real_cls_all',
               'd_real_train_accuracy_all',
               'test_loss_all',
               'test_accuracy_all']
loss_csv = pd.DataFrame(columns=header_name, data=data_loss)
loss_csv.to_csv('./loss/loss.csv', index=False)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('D_loss and G_loss', fontsize=10)
plt.plot(range(epoch), D_loss_real_cls_all,
         "ro-", label="D_loss_real_cls_all")
plt.plot(range(epoch), test_loss_all,
         "bs-", label="test_loss_all")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")


plt.subplot(1, 2, 2)
plt.title('Real and Fake loss', fontsize=10)
plt.plot(range(epoch), d_real_train_accuracy_all,
         "ro-", label="d_real_train_accuracy_all")
plt.plot(range(epoch), test_accur_all,
         "bs-", label="test_accur_all")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()