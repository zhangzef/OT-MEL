import argparse
from omegaconf import OmegaConf
import torch


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    return args


def sinkhorn_torch(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算两个3D张量之间的余弦相似度矩阵，并进行归一化处理
    
    参数：
    a: 形状为 (bz, sqa, dim) 的张量
    b: 形状为 (bz, sqb, dim) 的张量
    eps: 防止除以零的小常数
    
    返回：
    归一化后的余弦相似度矩阵，形状为 (bz, sqa, sqb)
    """
    # 计算L2范数（添加eps防止除零）
    a_norm = torch.norm(a, dim=-1, keepdim=True) + eps  # (bz, sqa, 1)
    b_norm = torch.norm(b, dim=-1, keepdim=True) + eps  # (bz, sqb, 1)
    
    # 扩展维度以便进行批量矩阵乘法
    # a: (bz, sqa, 1, dim)
    # b: (bz, 1, sqb, dim)
    a_expanded = a.unsqueeze(2)  
    b_expanded = b.unsqueeze(1)
    
    # 计算点积 (bz, sqa, sqb)
    dot_product = torch.matmul(a_expanded, b_expanded.transpose(-2, -1)).squeeze(-1)
    
    # 计算余弦相似度
    cos_sim = dot_product / (a_norm * b_norm.transpose(1, 2))  # (bz, sqa, sqb)
    
    # 归一化处理：(res + 1) / 2
    normalized = (cos_sim + 1) / 2
    
    return normalized