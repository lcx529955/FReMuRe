import torch
import torch.nn.functional as F
import numpy as np
from lib.Uncertainty import normalize_batch_uncertainty

def memory_computation(
        unc_vals,
        output_dir,
        rel_class_num,
        obj_class_num,
        obj_feature_dim=1024,
        rel_feature_dim=1936,
        obj_weight_type='al',
        rel_weight_type='al',
        obj_mem=False,
        obj_unc=False,
        include_bg_mem=False,
        freq_obj_tensor=None,
        freq_rel_tensor_dict=None,
        temperature=1.0,
        device='cpu'  # 默认在 CPU 上构建 memory，可传入 GPU
):
    # 归一化不确定性统计信息
    unc_vals.stats2()
    unc_list_rel = unc_vals.unc_list_rel
    unc_list_obj = unc_vals.unc_list_obj
    cls_rel_uc = unc_vals.cls_rel_uc
    cls_obj_uc = unc_vals.cls_obj_uc

    obj_emb_path = output_dir + 'obj_embeddings/'
    rel_emb_path = output_dir + 'rel_embeddings/'

    # 初始化对象 memory
    obj_memory = torch.zeros(obj_class_num, obj_feature_dim, device=device)
    obj_weight_sum = torch.zeros(obj_class_num, device=device)

    # 初始化关系 memory
    rel_memory = {k: torch.zeros(v, rel_feature_dim, device=device) for k, v in rel_class_num.items()}
    rel_weight_sum = {k: torch.zeros(v, device=device) for k, v in rel_class_num.items()}

    # 权重类型合法性检查
    if obj_weight_type not in ['simple', 'both', 'ep', 'al']:
        obj_weight_type = None
    if rel_weight_type not in ['simple', 'both', 'ep', 'al']:
        rel_weight_type = None

    for i in unc_list_rel.keys():
        # ===== 对象 memory 构建 =====
        if obj_mem:
            obj_features = torch.tensor(
                np.load(obj_emb_path + f"{i}.npy", allow_pickle=True)
            ).float().to(device)

            obj_u = torch.tensor(unc_list_obj[i][obj_weight_type]).float().to(device)
            idx, cls = torch.where(obj_u != 0)

            if not include_bg_mem:
                valid = cls != 0
                idx = idx[valid]
                cls = cls[valid]

            weights = obj_u[idx, cls]

            # tail class boost
            if freq_obj_tensor is not None:
                weights *= 1.0 / (freq_obj_tensor[cls] + 1e-6)  # or use log(freq + 2)

            weights = F.softmax(weights / temperature, dim=0)
            selected_feats = obj_features[idx] * weights.unsqueeze(1)

            for j in range(len(cls)):
                obj_memory[cls[j]] += selected_feats[j]
                obj_weight_sum[cls[j]] += weights[j]

        # ===== 关系 memory 构建 =====
        for rel_type in rel_class_num.keys():
            rel_features = torch.tensor(
                np.load(rel_emb_path + f"{i}_{rel_type}.npy", allow_pickle=True)
            ).float().to(device)

            rel_u = torch.tensor(unc_list_rel[i][rel_type][rel_weight_type]).float().to(device)
            idx, cls = torch.where(rel_u != 0)
            weights = rel_u[idx, cls]

            if freq_rel_tensor_dict is not None:
                freq_tensor = freq_rel_tensor_dict[rel_type].to(device)
                weights *= 1.0 / (freq_tensor[cls] + 1e-6)

            weights = F.softmax(weights / temperature, dim=0)
            selected_feats = rel_features[idx] * weights.unsqueeze(1)

            for j in range(len(cls)):
                rel_memory[rel_type][cls[j]] += selected_feats[j]
                rel_weight_sum[rel_type][cls[j]] += weights[j]

    # ===== 归一化对象 memory =====
    for c in range(obj_class_num):
        if obj_weight_sum[c] > 0:
            obj_memory[c] /= obj_weight_sum[c]

    # ===== 归一化关系 memory =====
    for rel_type in rel_memory:
        for c in range(rel_class_num[rel_type]):
            if rel_weight_sum[rel_type][c] > 0:
                rel_memory[rel_type][c] /= rel_weight_sum[rel_type][c]

    return rel_memory, obj_memory
