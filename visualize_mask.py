from models.vgg import vgg16_bn
import torch
import torch.nn as nn
import sys
# 从命令行参数中获取模型的检查点文件名
checkpoint = sys.argv[1] 
# 导入一些剪枝的辅助函数
from pruning_utils import extract_mask

'''

定义一个自定义的剪枝函数，可以根据不同的标准和回填率（fillback_rate）来选择保留哪些通道
model: 要剪枝的模型
mask_dict: 掩模字典，一个字典，存储了每一层卷积的权重掩码，即哪些权重被保留，哪些被置零
conv1: 一个布尔值，表示是否对第一层卷积进行剪枝，默认为False，处理低级特征
criteria: 一个字符串，表示剪枝的标准，可以是"magnitude"（按权重的绝对值大小排序）或其他自定义的标准，默认为"magnitude"
train_loader: 一个数据加载器，用于训练模型或计算其他标准，如梯度或敏感度，默认为None
fillback_rate: 一个浮点数，表示回填率，即在剪枝后，再随机地将一定比例的权重恢复为非零值，默认为0.0

'''

def prune_model_custom_fillback_time(model, mask_dict, conv1=False, criteria="magnitude", train_loader=None, fillback_rate = 0.0):
    # 定义一个新的掩码字典，用于存储剪枝后的结果
    new_mask_dict = {}
    # 定义一个列表，用于存储每一层卷积保留的通道数
    channels = []
    # 遍历模型中的所有模块
    for i, (name, m) in enumerate(model.named_modules()):
        # 如果是卷积层
        if isinstance(m, nn.Conv2d):
            # 如果是第一层卷积并且conv1为True，或者不是第一层卷积
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                # 获取该层卷积的权重掩码，并将其展平为二维矩阵，形状为[C, H*W*K*K]，其中C是输出通道数，H和W是输入特征图的高和宽，K是卷积核的大小
                mask = mask_dict[name + '.weight_mask']
                mask = mask.view(mask.shape[0], -1)
                # 计算每个输出通道中非零权重的个数，并保存在一个向量中，形状为[C]
                count = torch.sum(mask != 0, 1)
                # 计算当前层卷积的稀疏度，即非零权重在所有权重中的比例
                # sparsity = torch.sum(mask) / mask.numel()
                # 计算当前层卷积保留的输出通道数，即非零权重在每个输出通道中的平均个数乘以回填率后向上取整
                num_channel = count.sum().float() / mask.shape[1]
                num_channel = num_channel + (mask.shape[0] - num_channel) * fillback_rate
                print(num_channel)
                # 将保留的输出通道数分解为整数部分和小数部分
                int_channel = int(num_channel)
                frac_channel = num_channel - int_channel
                # 将保留的输出通道数添加到列表中，并加一以避免为零
                channels.append(int(num_channel) + 1)

                # 如果剪枝标准是按权重绝对值的大小
                if criteria == 'magnitude':
                    # 获取该层卷积的权重掩码
                    mask = mask_dict[name + '.weight_mask']
                    # 计算每个输出通道中所有权重的绝对值之和，并保存在一个向量中，形状为[C]
                    count = m.weight.data.view(mask.shape[0], -1).abs().sum(1)
                    # 找到第int_channel大的权重绝对值之和，作为剪枝的阈值
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)

                    # 将权重绝对值之和大于阈值的输出通道的掩码设为1，表示保留
                    mask[torch.where(count > threshold)[0]] = 1
                    # 将权重绝对值之和小于阈值的输出通道的掩码设为0，表示剪除
                    mask[torch.where(count < threshold)[0]] = 0
                    # 将权重绝对值之和等于阈值的输出通道的掩码，按照小数部分的比例随机地设为1或0，表示部分保留
                    mask[torch.where(count == threshold)[0], :int(frac_channel * mask.shape[1])] = 1
                    mask[torch.where(count == threshold)[0], int(frac_channel * mask.shape[1]):] = 0

                # 将剪枝后的掩码恢复为原来的形状，并保存在新的掩码字典中
                mask = mask.view(*mask_dict[name + '.weight_mask'].shape)
                new_mask_dict[name + '.weight_mask'] = mask
                # 使用自定义的剪枝函数，将剪枝后的掩码应用到模型中，即将被剪除的权重置零
                # prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))

    # 返回新的掩码字典
    return new_mask_dict


state_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
model = vgg16_bn(pretrained=False)
current_mask = extract_mask(state_dict)
import copy
current_mask_copy = copy.deepcopy(current_mask)
print(current_mask.keys())
refill_masks = prune_model_custom_fillback_time(model, current_mask_copy)
from pruning_utils import regroup
regroup_masks = {}
current_mask_copy_2 = copy.deepcopy(current_mask)

for key in current_mask_copy_2:
    mask = current_mask_copy_2[key]
    regroup_masks[key] = regroup(mask.view(mask.shape[0], -1))
    print(regroup_masks[key].numel() / (regroup_masks[key].abs() > 0).float().sum())

all_masks = {'refill': refill_masks, 'imp': current_mask, 'regroup': regroup_masks}

torch.save(all_masks, f"{sys.argv[2]}")