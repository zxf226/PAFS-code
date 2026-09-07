import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from functools import reduce
from utils.projector import DimensionalityReducer


class CosineKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', norm='L2'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm.upper()

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        if self.norm == "L2":
            loss = ((inputs - targets)**2).mean(dim=1)
        else:
            loss = (inputs - targets).mean(dim=1)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets / self.alpha, dim=1)

        loss = -(outputs * labels).mean(dim=1) * (self.alpha ** 2)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


# MiB Losses
class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class CosineLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.crit = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = 1 - self.crit(x, y)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = loss
        return - loss


class AdaptiveFeatureDistillation(nn.Module):
    def __init__(self, reduction='mean', alpha=1):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.reducer = DimensionalityReducer()

    def forward(self, inputs, targets, labels=None, num_classes=None, weights=1, pod=True):
        if pod:
            inputs = self.reducer(inputs)
            targets = self.reducer(targets)
            tot_classes = reduce(lambda a, b: a + b, num_classes)
            new_cls_num = num_classes[-1]
            old_cls_num = tot_classes - new_cls_num
            loss = features_distillation(
                inputs,
                targets,
                labels=labels,
                index_new_class=old_cls_num,
                nb_current_classes=tot_classes,
                nb_new_classes=new_cls_num
            )
        else:
            # inputs = self.reducer(inputs)
            # targets = self.reducer(targets)
            loss = (inputs - targets) ** 2
        loss = loss * weights * self.alpha
        if self.reduction == 'mean':
            if torch.is_tensor(weights):
                mask = torch.where(weights > 0, 1, 0)
                count = torch.sum(mask.expand_as(loss))
                return torch.sum(loss) / count
            elif weights == 1:
                return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
def features_distillation(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="local",  # 通道池化处理方式
    labels=None,
    index_new_class=None,
    pod_deeplab_mask=False,
    pod_deeplab_mask_factor=None,
    pod_factor=0.01,  # POD损失的权重因子
    prepro="pow",  # 预处理方法
    spp_scales=[1, 2, 4],  # SPP的尺度
    pod_options={"switch": {"after": {"extra_channels": "sum", "factor": 0.0001, "type": "local"}}},
    use_pod_schedule=True,
    nb_current_classes=-1,
    nb_new_classes=-1
):

    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)
#     print("list_attentions_a:", list_attentions_a.shape)
    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    #if collapse_channels in ("spatial_tuple", "spp", "spp_noNorm", "spatial_noNorm"):
    normalize = False

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"  # 掩码的位置 Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None

    loss = torch.tensor(0.).to(list_attentions_a[0].device)  # 初始化损失值loss为零，并将其移动到与注意力图相同的设备
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"  # 处理多余的通道
        normalize_per_scale = True  # 对每个尺度进行归一化处理

        if pod_options and pod_options.get("switch"):
            if i < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:  # 检查是否包含before配置，更新相关参数
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get("apply_mask", apply_mask)
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get("mix_new_old", mix_new_old)

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:  # 检查通道数是否相同
            assert i == len(list_attentions_a) - 1  # 确保当前处理的是最后一个注意力图对
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).to(a.dtype).to(a.device)
                _b[:, 0] = b[:, 0] + b[:, index_new_class + 1:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class + 1]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class + 1]
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":  # 平方
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":  # 绝对值
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        if collapse_channels == "spatial":  # 将注意力图沿着宽度和高度方向求和，并将结果拼接起来
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)

        elif collapse_channels == "local":
                # 对局部池化的注意力图进行处理
                a = _local_pod(
                    a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if i == len(list_attentions_a) - 1 and pod_options is not None:  # 选择适当的特征计算函数
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2, eps=1e-6)
            b = F.normalize(b, dim=1, p=2, eps=1e-6)

        a = torch.nan_to_num(a, nan=0.0, posinf=1e4, neginf=-1e4)
        b = torch.nan_to_num(b, nan=0.0, posinf=1e4, neginf=-1e4)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
                diff = a - b
                layer_loss = torch.sqrt((diff * diff).sum(dim=-1) + 1e-6)
                # layer_loss = torch.frobenius_norm(a - b, dim=-1)
        elif difference_function == "frobenius_mix":
            a = torch.nan_to_num(a, nan=0.0, posinf=1e4, neginf=-1e4)
            b = torch.nan_to_num(b, nan=0.0, posinf=1e4, neginf=-1e4)

            diff_old = a[0] - b[0]
            diff_new = a[1] - b[1]

            layer_loss_old = torch.sqrt((diff_old * diff_old).sum(dim=-1) + 1e-6)
            layer_loss_new = torch.sqrt((diff_new * diff_new).sum(dim=-1) + 1e-6)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":  # L1范数
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")
        
        layer_loss = torch.nan_to_num(layer_loss, nan=0.0, posinf=1e4, neginf=0.0)
        
        assert torch.isfinite(layer_loss).all(), layer_loss  # 检查是否所有值都有限
        assert (layer_loss >= 0.).all(), layer_loss  # 检查是否所有值不小于零

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)  # 加权平均
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss  # 缩放
        if use_pod_schedule and nb_new_classes > 0:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)  # 进一步缩放
        loss += layer_loss

    return loss / len(list_attentions_a)

def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))

# x = b * c * h * w
def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []  # 储存池化结果的列表

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale
        if k == 0:
            continue

        nb_regions = scale**2  # 当前尺度下的总区域数

        for i in range(scale):
            for j in range(scale):   # 张量切片的方式获取x的一个子区域

                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]
                horizontal_pool = tensor.mean(dim=1).view(b, -1)  # 水平池化  -1 表示自动推断该维度的大小 c*k
                vertical_pool = tensor.mean(dim=1).view(b, -1)  # 垂直池化
                if not (torch.isfinite(horizontal_pool).all() and torch.isfinite(vertical_pool).all()):
                    horizontal_pool = torch.zeros_like(horizontal_pool)
                    vertical_pool = torch.zeros_like(vertical_pool)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":  # 则根据尺度进行归一化
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(
                        horizontal_pool, dim=1, p=2, eps=1e-6
                    )
                    vertical_pool = F.normalize(
                        vertical_pool, dim=1, p=2, eps=1e-6
                    )

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)