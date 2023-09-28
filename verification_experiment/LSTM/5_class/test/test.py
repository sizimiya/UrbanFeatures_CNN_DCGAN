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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        #xt = xt.reshape(1, 1024)
        xt = xt.view(1024, 1)

        yt = self.label[index]

        return xt, yt

    def __len__(self):
        return self.length

testset = CsvDataset_test()

testloader = DataLoader(dataset=testset, batch_size=450, shuffle=True)

#------------------------------------------------------------------------------------------------

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    # def forward(self, x):
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    #     c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    #     out, _ = self.lstm(x, (h0, c0))
    #     out = self.fc(out[:, -1, :])
    #     return out

    def forward(self, x):
        # 展开为时间步序列
        x = x.view(x.size(0), 1024, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
input_size = 1 # 城市特征维度
hidden_size = 64
num_layers = 2
num_classes = 5

D = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)

checkpoint = torch.load('./model/Discriminator_cuda_800.pkl', map_location='cpu')
D.load_state_dict(checkpoint)
D = D.to(device)
D.eval()

# 损失函数和优化器
loss2 = nn.CrossEntropyLoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0001)
#------------------------------------------------------------------------------------------------

count = 0
epoch = 5
iter_count = 0

D_loss_real_cls_all = []
d_real_train_accuracy_all = []

test_loss_all = []  # 存放测试集损失的数组
test_accur_all = []  # 存放测试集准确率的数组

# iter_count_all = []

for i in range(epoch):
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

data_loss = zip(test_loss_all,
                test_accur_all)
header_name = ['test_loss_all',
               'test_accuracy_all']
loss_csv = pd.DataFrame(columns=header_name, data=data_loss)
loss_csv.to_csv('./loss/loss.csv', index=False)
