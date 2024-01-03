import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channel=1280, depth=32):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
     # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
     # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x

class FusionModel(nn.Module):

    def __init__(self, model: nn.Module, num_classes, in_dim=32, mode='normal'):
        super(FusionModel, self).__init__()

        self.backbone = nn.Sequential(*list(model.children())[:-1])

        print(self.backbone)
        self.mode = mode
        assert mode in ['normal', 'adaptive'], "模式只有normal和adaptive"
        self.adaptive_fusion = None
        if mode == 'adaptive':
            self.adaptive_fusion = AdaptiveWeightedAdd(in_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.spp = ASPP()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, images: list):
        x1 = self.backbone(images[:, 0, :, :])
        x1=self.spp(x1)

        x2 = self.backbone(images[:, 1, :, :])
        x2 = self.spp(x2)
        x3 = self.backbone(images[:, 2, :, :])
        x3 = self.spp(x3)

        if self.mode == 'normal':
            fusion = x1 + x2 + x3
            fusion = self.avgpool(fusion)
            fusion = torch.flatten(fusion, 1)
        else:

            x1 = self.avgpool(x1)

            x1 = torch.flatten(x1, 1)

            x2 = self.avgpool(x2)

            x2 = torch.flatten(x2, 1)

            x3 = self.avgpool(x3)

            x3 = torch.flatten(x3, 1)

            fusion = self.adaptive_fusion([x1, x2, x3])

        fusion = self.fc(fusion)
        return fusion

    def freeze_backbone(self):
        backbone = [self.backbone]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.backbone]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True

class FusionModel_original(nn.Module):

    def __init__(self, model: nn.Module, num_classes, in_dim=32, mode='normal'):
        super(FusionModel_original, self).__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        print(self.backbone)
        self.mode = mode
        assert mode in ['normal', 'adaptive'], "模式只有normal和adaptive"
        self.adaptive_fusion = None
        if mode == 'adaptive':
            self.adaptive_fusion = AdaptiveWeightedAdd(in_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, images: list):
        x1 = self.backbone(images[:, 0, :, :])
        x2 = self.backbone(images[:, 1, :, :])
        x3 = self.backbone(images[:, 2, :, :])

        if self.mode == 'normal':
            fusion = x1 + x2 + x3
            fusion = self.avgpool(fusion)
            fusion = torch.flatten(fusion, 1)
        else:

            x1 = self.avgpool(x1)
  
            x1 = torch.flatten(x1, 1)

            x2 = self.avgpool(x2)

            x2 = torch.flatten(x2, 1)

            x3 = self.avgpool(x3)

            x3 = torch.flatten(x3, 1)

            fusion = self.adaptive_fusion([x1, x2, x3])

        fusion = self.fc(fusion)
        return fusion

    def freeze_backbone(self):
        backbone = [self.backbone]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.backbone]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True
class AdaptiveWeightedAdd(nn.Module):
    def __init__(self, in_features, list_nums=3):
        super().__init__()
        # SE对二维映射，这个是一维的
        print("in_features--------------------------：",in_features,list_nums,)
        self.attention = nn.Sequential(
            nn.Linear(in_features * list_nums, in_features // (8 * list_nums)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // (8 * list_nums), list_nums),
            nn.Softmax(dim=1)
        )

    def forward(self, features: list):

        attention_weights = self.attention(torch.cat(features, dim=1))

        x = features[0] * torch.unsqueeze(attention_weights[:, 0], dim=1)
        for i, feature in enumerate(features[1:]):

            x += torch.unsqueeze(attention_weights[:, i + 1], dim=1) * feature
        return x


# 定义一个注意力机制，使用全连接层作为注意力模型
class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn_weights = self.softmax(self.fc(x))
        return attn_weights * x


# 定义一个融合模型，使用注意力机制将三个全连接层融合起来
class AdaptiveFusionModel(nn.Module):
    def __init__(self, in_dim):
        super(AdaptiveFusionModel, self).__init__()
        self.att1 = Attention(in_dim)
        self.att2 = Attention(in_dim)
        self.att3 = Attention(in_dim)

    def forward(self, fc1, fc2, fc3):
        attn_fc1 = self.att1(fc1)
        attn_fc2 = self.att2(fc2)
        attn_fc3 = self.att3(fc3)
        fusion = attn_fc1 + attn_fc2 + attn_fc3
        return fusion


if __name__ == '__main__':
    from nets.mobilenetv2 import mobilenetv2
    net = mobilenetv2()
    model = FusionModel(net, 3)
    for k, v in model.named_modules():
        print(k)