import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys
import shutil
import pandas as pd
from tqdm import tqdm
import torchvision
import torch.nn.functional as F

#dataset/dataloader————————————————————————————————————————————————————————————————————
class CsvDataset(Dataset):
    def __init__(self):
        super(CsvDataset, self).__init__()
        self.feature_path = 'data_trans.csv'
        # self.label_path = 'data_label_10.csv'
        feature_df_ = pd.read_csv(self.feature_path)
        # label_df_ = pd.read_csv(self.label_path)
        # assert feature_df_.columns.tolist()[1:] == label_df_[label_df_.columns[0]].tolist(), \
        #     'feature name does not match label name'
        self.feature = [feature_df_[i].tolist() for i in feature_df_.columns[1:]]
        # self.label = label_df_[label_df_.columns[1]]
        # assert len(self.feature) == len(self.label)
        self.length = len(self.feature)

    def __getitem__(self, index):
        x = self.feature[index]
        x = torch.Tensor(x)
        x = x.reshape(1, 32, 32)

        return x

    def __len__(self):
        return self.length


batch_size = 1936

dataset = CsvDataset()
dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256 * 8 * 8)
        self.bn1 = nn.BatchNorm1d(256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1)  # 生成（128，7，7）的二维图像
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)  # 生成（64，14，14）的二维图像
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)  # 生成（1,28,28）的二维图像

    def forward(self, x1):
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x = x1.view(-1, 256, 8, 8)
        x = F.relu(self.deconv1(x))
        x = self.bn3(x)
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.tanh(self.deconv3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)  # 64*15*15
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # 128*7*7
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x1):
        x = F.dropout2d(F.leaky_relu(self.conv1(x1)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))  # (batch,128,6,6)
        x = self.bn(x)
        x = x.view(-1, 128 * 7 * 7)  # 展平
        x = torch.sigmoid(self.fc(x))
        return x



# #设置随机种子，方便重复性实验——————————————————————————————————————————————
# manualSeed = 999
# print('Random Seed:', manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

#基本参数设置
image_size = 32        #or_data尺寸
nc = 3                 #or_data通道数
nz = 100               #噪声向量维度
ngf = 64               #生成器通道数
ndf = 64               #判别器通道数

criterion = nn.BCELoss()  #损失函数

real_label = 1.0       #真假标签
fake_label = 0.0

#是否使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#创建生成器与判别器
netG = Generator().to(device)
netD = Discriminator().to(device)

#G和D的优化器，使用Adam
# Adam学习率与动量参数
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=0.00001, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(beta1, 0.999))


#损失变量
D_loss_all = []  # 记录训练过程中判别器的损失
G_loss_all = []  # 记录训练过程中生成器的损失

D_real_loss_all = []
D_fake_loss_all = []

iter_count_all = []

Real_score_all = []
Fake1_score_all = []
Fake2_score_all = []
#迭代次数
iter_count = 0
#epochs
epoch = 30000
gepoch = 1
# #生成固定噪声，便于每个epoch比较
# fixed_noise = torch.randn(64, nz, 1, 1, device= device)
#开始遍历
for i in range(epoch):
    D_epoch_loss = 0
    G_epoch_loss = 0

    D_epoch_real_loss = 0
    D_epoch_fake_loss = 0

    real_score_epoch = 0
    fake1_score_epoch = 0
    fake2_score_epoch = 0

    train_num = 0.0

    for step, x in enumerate(dl):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 训练真实图片
        data = x
        netD.zero_grad()
        real_data = data.to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_data).view(-1)
        # 计算真实图片损失，梯度反向传播
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## 训练生成图片
        # 产生latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # 使用G生成图片
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        # 计算生成图片损失，梯度反向传播
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # 累加误差，参数更新
        errD = errD_real + errD_fake
        optimizerD.step()
        train_num += data.size(0)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 给生成图赋标签
        # 对生成图再进行一次判别
        output = netD(fake).view(-1)
        # 计算生成图片损失，梯度反向传播
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        print(f"epoch:{i}, "
              f"d_loss:{errD.item()}, "
              f"g_loss:{errG.item()},\n "
              f"d_loss_fake:{errD_fake.item()},"
              f"d_loss_real:{errD_real.item()},\n"
              f"real_score:{D_x},"
              f"fake1_score:{D_G_z1},"
              f"fake2_score:{D_G_z2},\n")

        netG.eval()
        with torch.no_grad():
            fixed_noise = torch.randn(1, nz, device=device)
            fake = netG(fixed_noise).detach().cpu()
            data_singel = fake.reshape(1, 1 * 32 * 32)
            data_np = data_singel.detach().numpy()
            df = pd.DataFrame(data_np)
            data_df = df
            data_df.to_csv(f'./pred/data{str(iter_count)}.csv', index=None, header=None)
        netG.train()

        iter_count += 1
        iter_count_all.append(iter_count)
        print('iter_count:', iter_count)  # 总计训练次数（固定值）=（样本数/batch_size)*epoch， 在每个epoch训练完后重置
        print('train_num:', train_num)  # 样本递增数(递增值）=样本数+=1， 在每个epoch训练完后重置

        with torch.no_grad():  # loss在每个epoch训练完后后重置
            D_epoch_loss += errD.cpu().item()
            G_epoch_loss += errG.cpu().item()
            D_epoch_real_loss += errD_real.cpu().item()
            D_epoch_fake_loss += errD_fake.cpu().item()

            real_score_epoch += D_x
            fake1_score_epoch += D_G_z1
            fake2_score_epoch += D_G_z2

        # 求平均损失
        # with torch.no_grad():
    train_num = train_num / batch_size
    D_epoch_loss /= train_num
    G_epoch_loss /= train_num
    D_epoch_real_loss /= train_num
    D_epoch_fake_loss /= train_num

    real_score_epoch /= train_num
    fake1_score_epoch /= train_num
    fake2_score_epoch /= train_num


    D_loss_all.append(D_epoch_loss)
    G_loss_all.append(G_epoch_loss / gepoch)
    D_real_loss_all.append(D_epoch_real_loss)
    D_fake_loss_all.append(D_epoch_fake_loss)

    Real_score_all.append(real_score_epoch)
    Fake1_score_all.append(fake1_score_epoch)
    Fake2_score_all.append(fake2_score_epoch)

    # 保存模型
    # state = gen.state_dict()
    # torch.save(state, f'./Generator/model_save/gen_model{str(i)}.pth')

data_loss = zip(iter_count_all, D_loss_all, G_loss_all, D_real_loss_all, D_fake_loss_all, Real_score_all, Fake1_score_all, Fake2_score_all)
header_name = ['iter_count', 'd_loss', 'g_loss', 'd_loss_real', 'd_loss_fake', 'Real_score', 'Fake1_score', 'Fake2_score']
loss_csv = pd.DataFrame(columns=header_name, data=data_loss)
loss_csv.to_csv('./loss/loss.csv', index=False)

plt.figure(figsize=(12, 4))
plt.subplot(2, 2, 1)
plt.plot(range(epoch), D_loss_all,
         "ro-", label="d_loss")
plt.plot(range(epoch), G_loss_all,
         "bs-", label="g_loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(range(epoch), D_real_loss_all,
         "ro-", label="d_loss_real")
plt.plot(range(epoch), D_fake_loss_all,
         "bs-", label="d_loss_fake")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(epoch), Real_score_all,
         "ro-", label="Real_score_all")
plt.plot(range(epoch), Fake1_score_all,
         "bs-", label="Fake1_score_all")
plt.plot(range(epoch), Fake2_score_all,
         "gs-", label="Fake2_score_all")
plt.xlabel("epoch")
plt.ylabel("score")
plt.legend()

plt.show()





