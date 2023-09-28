import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict
import numpy as np
import os
import math
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt

#随机深度方法
#这是一个自定义的drop_path函数，用于实现DropPath正则化。DropPath是在训练期间随机将一些神经元的输出置零，以减少模型的过拟合。
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

#这是DropPath类，它是一个PyTorch模块，封装了drop_path函数。在模型中，可以使用这个模块来应用DropPath正则化。
class DropPath(nn.Module):
    def __init__(self, drop_porb=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_porb

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#这是PatchEmbed类，它实现了图像的分块嵌入。它将输入图像划分成固定大小的图像块（patch），然后使用卷积层将每个图像块嵌入到一个低维向量中，以供后续处理使用。
class PatchEmbed(nn.Module):    #patch embedding 模块，按照相同的大小分割图像，使用卷积层分割，原始大小16*16的卷积核，步距也是16

    def __init__(self, img_size=32, patch_size=4, in_c=1, embed_dim=256, norm_layer=None):  #img_size就是图片尺寸，patch_size是卷积核大小， in_c是图片通道数，原始图片224*224*3，卷积成14*14*768，拉平后为196*768
        super().__init__()                                   #embed_dim通常代表嵌入维度（embedding dimension）或特征维度（feature dimension）。它是指将输入图像分解为一系列特征向量时，每个特征向量的长度或维度
        img_size = (img_size, img_size)                                     #norm_layer在Vision Transformer模型中通常代表归一化层，用于对特征进行标准化，加速模型的训练和提高模型的鲁棒性。
        patch_size = (patch_size, patch_size)         #"Patch"是指将输入图像分割为一系列固定大小的图像块。它是将图像划分为小块的过程，通常是通过将图像分割成正方形或矩形的小区域。
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])   #//是整除，余数四舍五入，/是正常除
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()    #Identity代表不进行操作

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"

        # flatten: [B, C, H, W]  -> [B, C, HW]
        # transpose: [B, C, HW]  -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)  #flatten是从第3个维度开始展平，也就是HW，transpose是调换不同维度的数据，吧第2和第3维度数据调换。
        x = self.norm(x)
        return x

#这是Attention类，它实现了自注意力机制的一部分。自注意力机制用于学习输入序列中不同位置之间的依赖关系，这在Transformer模型中是非常重要的。这个类包含了注意力的计算和投影操作。
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim  "token"是指将输入图像分割为一系列固定大小的图像块（或路径，patches）后，对每个图像块进行编码得到的表示向量。图像块的数量越多，模型可以捕捉到更细粒度的图像信息，但也会增加计算开销和存储需求。
                 num_heads=8,   #multiple head 中 head的个数
                 qkv_bias=False,  #生成qkv时是否使用偏置，默认不使用
                 qk_scale=None,
                 attn_drop_ration=0.1,
                 proj_drop_ratio=0.1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  #把qkv的dim除以head的数量
        self.scale = qk_scale or head_dim ** -0.5   #就是除以根号下的dk
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)  #使用全连接层得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ration)    #dropout层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape   # B, N, C就是：[batch_size, num_patches + 1, total_embed_dim]  num_patches就是像素数除以embedding dim之后的token的个数，+1是要加class token

        # qkv(): ->[batch_size, num_patches +1, 3 * total_embed_dim]
        #reshape: ->[batch_size, num_patches+1, 3, num_heads, embed_dim_per_head]  #reshape后的数据排列
        #permute: ->[3, batch_size, num_heads, num_patches+1, embed_dim_per_head]  #permute后的数据排列
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #通过全连接层开始生成qkv
                ###B：batch size, N:num_patches +1, 3:qkv三个参数， self.num_heads：heads的数量， C：total_embed_dim
                ###permute 按照维度重新排列，排列顺序是2，0，3，1，4
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]   #make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches +1]
        attn = (q @ k.transpose(-2, -1)) * self.scale   #@是一个运算符，称为"矩阵乘法运算符"（Matrix Multiplication Operator）
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @:multiple -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches+1, num_heads, embed_dim_per_head]
        # reshape: -> [batches_size, num_patches +1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#这是Mlp类，它实现了多层感知机（MLP）模块，用于在Transformer的每个块中对特征进行非线性映射和变换。
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#这是Block类，它实现了Transformer模型中的基本块。一个块包括自注意力（Attention）层、多层感知机（MLP）层和残差连接（Residual Connection）。
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ration=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,          #drop比率
                 attn_drop_ratio=0.,     #drop比率
                 drop_path_ratio=0.,     #drop比率
                 act_layer=nn.GELU,      #激活函数
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ration=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        #note: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ration)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio) ###########################################################################################################

    def forward(self, x):                                      #x+代表残差网络中的与原始数值相加，也就是Encoder Block中的两个加号的位置
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#这是VisionTransformer类，它是整个ViT模型的定义。该类包括了模型的各个组件，包括输入的图像嵌入、多个块、预处理层（如果有的话）、分类头部等。
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=1, num_classes=10,                              #depth是encoder block重复的次数  num_heads是head的个数  mlp_ratio是mlp中节点扩充的倍数
                 embed_dim=256, depth=12, num_heads=8, mlp_ratio=2, qkv_bias=True,                      #representation_size=pre_logits层的作用是通过进一步处理编码器的输出，为后续的分类任务提供更好的特征表示。
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  #输入图像被划分为若干个图块（patches），每个图块将被视为一个独立的令牌。因此，num_tokens 的值为 1，表示只有一个特殊的令牌，即分类任务的标记令牌（cls_token）。
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU


        #patch embdding层
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        #class token 1*768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #（1, 1, embed_dim)分别是batch， 1， 768
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  #不用管

        # positon embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  #分别是batch， num_patches=14*14=196，+1=197， 768
        self.pos_drop = nn.Dropout(p=drop_ratio)

        #Encoder block
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule  #递增等差序列
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ration=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        #Encoder block之后的Layer norm层
        self.norm = norm_layer(embed_dim)

        # Representation layer 也就是Pre_logistic层
        if representation_size and not distilled:      #不要管 distilled
            self.has_logits = True
            self.num_features = representation_size
            #pre_logistic就是一个全连接层加一个激活函数，也就是linear加Tanh
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()


        # Classifier head(s) 就是vit最后一个Linear层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        #以下三行和vit模型无关
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


        # Weight init 初始化部分
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)


    #正向传播部分
    def forward_features(self, x):

        #开始patch embedding
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]

        #将class token复制batch_size份
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        #将class token与patch embedding进行拼接
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        #encoder block之前的第一个加号，与之前的drop out层
        x = self.pos_drop(x + self.pos_embed)
        #encoder block层
        x = self.blocks(x)
        #layer norm层
        x = self.norm(x)
        #extract class token层，(x[:, 0])取数据全部第1维（batch），分割出第2维（也就是class_token）
        #然后进行pre_logists
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)     #这个head对应的就是vit最后全连接层，也就是输出层
        return x

#这是一个辅助函数_init_vit_weights，用于初始化ViT模型的权重。
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

#这是一个用于创建ViT模型的函数vit8，它根据指定的参数创建一个ViT模型实例。这个函数是整个模型的入口。
def vit8(num_classes: int = 10, has_logits: bool = True):

    model = VisionTransformer(img_size=32,
                              patch_size=4,    #patch size设置的越小，计算量越大
                              embed_dim=256,
                              depth=12,       #就是重复堆叠encoder block的次数
                              num_heads=8,
                              mlp_ratio=2,
                              representation_size=256 if has_logits else None,
                              num_classes=10)
    return model




#_____________________测试集____________________________________________________________________________________________________

class CsvDataset_test(Dataset):
    def __init__(self):
        super(CsvDataset_test, self).__init__()
        self.feature_path = 'data_test.csv'
        self.label_path = 'data_label_10_test.csv'
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
        xt = xt.reshape(1, 32, 32)

        yt = self.label[index]

        return xt, yt

    def __len__(self):
        return self.length

testset = CsvDataset_test()

testloader = DataLoader(dataset=testset, batch_size=128, shuffle=True)


loss2 = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vit = vit8()

checkpoint = torch.load('./model/Discriminator_cuda_330.pkl', map_location='cpu')
vit.load_state_dict(checkpoint)

vit = vit.to(device)
vit.eval()
d_optimizer = optim.Adam(vit.parameters(), lr=0.0001)

count = 0
epoch = 10
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
    vit.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        for (x, y) in testloader:
            t_img = x.to(device)
            t_label = y.to(device)

            t_cls_out = vit(t_img)  # 真图片送入判别器D  得到真假输出 100*1 和分类输出100*15(10)
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
