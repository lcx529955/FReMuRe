import os
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from dataloader.action_genome import AG


# ================== 数据统计 ==================

def collect_relation_frequencies(dataset):
    att_n = len(dataset.attention_relationships)
    spa_n = len(dataset.spatial_relationships)
    counter = Counter()
    for video_ann in dataset.gt_annotations:
        for frame_ann in video_ann:
            for obj_ann in frame_ann:
                if isinstance(obj_ann, dict):
                    if 'attention_relationship' in obj_ann:
                        for a in obj_ann['attention_relationship'].tolist():
                            counter[a] += 1
                    if 'spatial_relationship' in obj_ann:
                        for s in obj_ann['spatial_relationship'].tolist():
                            counter[s + att_n] += 1
                    if 'contacting_relationship' in obj_ann:
                        for c in obj_ann['contacting_relationship'].tolist():
                            counter[c + att_n + spa_n] += 1
    return counter


# ================== 工具 ==================

def get_gradient_colors(n, cmap_name='Blues', start=0.95, end=0.25):
    cmap = plt.cm.get_cmap(cmap_name)
    vals = np.linspace(start, end, n)
    return [cmap(v) for v in vals]


# 新增：Contacting 自定义红->橙->橙黄->黄渐变
def get_contacting_colors(n):
    anchor_hex = ['#d73027', '#fc8d59', '#febb2c', '#fee08b']  # 深红 -> 橙 -> 橙黄 -> 浅黄
    if n <= 0:
        return []
    if n == 1:
        return [anchor_hex[0]]
    # 计算插值位置
    anchors = np.linspace(0.0, 1.0, len(anchor_hex))
    target_pos = np.linspace(0.0, 1.0, n)
    # 将 hex 转 RGB
    anchor_rgb = np.array([tuple(int(h[i:i+2], 16) for i in (1, 3, 5)) for h in anchor_hex], dtype=float) / 255.0
    # 插值
    out = []
    for p in target_pos:
        # 找到区间
        idx = np.searchsorted(anchors, p, side='right') - 1
        if idx >= len(anchors) - 1:
            rgb = anchor_rgb[-1]
        else:
            t = (p - anchors[idx]) / (anchors[idx+1] - anchors[idx] + 1e-8)
            rgb = anchor_rgb[idx] * (1 - t) + anchor_rgb[idx+1] * t
        out.append((rgb[0], rgb[1], rgb[2], 1.0))
    return out


# ================== 绘制 全部(紫色) ==================

def build_bar_plot(counter, rel_names, out_path, topk=None, dataset=None):
    """全局频率降序的总图；颜色沿用单轴图的分组渐变：
    Contacting=自定义红橙黄渐变；Spatial=Blues；Attention=Greens。
    颜色分配逻辑：先在各组内按组内频率降序生成梯度 -> 映射到关系索引 -> 全局频率降序排列后保持颜色不变。
    """
    counts = np.array([counter.get(idx, 0) for idx in range(len(rel_names))], dtype=np.int64)
    # 分组信息（依赖 dataset 提供）
    if dataset is None:
        raise ValueError('dataset 参数不���为空以提供分组长度')
    att_n = len(dataset.attention_relationships)
    spa_n = len(dataset.spatial_relationships)
    total_n = len(rel_names)
    # 组索引切片
    att_idx = np.arange(0, att_n)
    spa_idx = np.arange(att_n, att_n + spa_n)
    con_idx = np.arange(att_n + spa_n, total_n)

    # 组内排序并生成颜色
    color_map = {}
    # Attention 组
    att_counts = counts[att_idx]
    att_order = att_idx[np.argsort(-att_counts)]
    att_colors = get_gradient_colors(len(att_idx), cmap_name='Greens', start=0.95, end=0.25)
    for i, rid in enumerate(att_order):
        color_map[rid] = att_colors[i]
    # Spatial 组
    spa_counts = counts[spa_idx]
    spa_order = spa_idx[np.argsort(-spa_counts)]
    spa_colors = get_gradient_colors(len(spa_idx), cmap_name='Blues', start=0.95, end=0.25)
    for i, rid in enumerate(spa_order):
        color_map[rid] = spa_colors[i]
    # Contacting 组 (自定义)
    con_counts = counts[con_idx]
    con_order = con_idx[np.argsort(-con_counts)]
    con_colors = get_contacting_colors(len(con_idx))
    for i, rid in enumerate(con_order):
        color_map[rid] = con_colors[i]

    # 全局排序
    global_order = np.argsort(-counts)
    if topk is not None:
        global_order = global_order[:topk]
    sorted_counts = counts[global_order]
    sorted_names = [rel_names[i] for i in global_order]
    total = sorted_counts.sum() if sorted_counts.sum() > 0 else 1
    freqs = sorted_counts / total

    # 按全局顺序收集颜色
    bar_colors = [color_map[i] for i in global_order]

    plt.figure(figsize=(max(10, 0.5 * len(sorted_names)), 6))
    bars = plt.bar(range(len(sorted_names)), freqs, color=bar_colors, edgecolor='none')
    for i, b in enumerate(bars):
        h = b.get_height()
        plt.text(b.get_x()+b.get_width()/2, h + 0.001, f"{sorted_counts[i]}", ha='center', va='bottom', fontsize=8)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=50, ha='right')
    plt.ylabel('Frequency (relative)')
    # plt.xlabel('Relation Category')  # 去掉底部 label 与单轴图风格一致
    plt.title('All Relations (Global Descending with Group Gradients)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print('Saved:', out_path)


# ================== 绘制 三组单轴拼接 ==================

def build_combined_groups_single_axis(counts_full, dataset, out_path):
    """单个坐标轴：Contacting -> Spatial -> Attention，组内降序。
    纵轴：每组内部的相对频率 (各组内部求和=1)。
    颜色：Contacting=自定义红-橙-橙黄-黄���变, Spatial=Blues, Attention=Greens。"""
    att_n = len(dataset.attention_relationships)
    spa_n = len(dataset.spatial_relationships)
    con_n = len(dataset.contacting_relationships)
    att_counts = counts_full[:att_n]
    spa_counts = counts_full[att_n:att_n+spa_n]
    con_counts = counts_full[att_n+spa_n:]
    groups = [
        # Contacting 放首位
        ('Contacting', con_counts, dataset.contacting_relationships, 'CONTACTING_CUSTOM'),
        ('Spatial',    spa_counts, dataset.spatial_relationships,   'Blues'),
        ('Attention',  att_counts, dataset.attention_relationships, 'Greens'),
    ]
    all_names = []
    all_counts = []
    group_ranges = []
    start = 0
    for title, g_counts, g_names, cmap_name in groups:
        g_counts = np.array(g_counts, dtype=np.int64)
        order = np.argsort(-g_counts)
        g_sorted = g_counts[order]
        names_sorted = [g_names[i] for i in order]
        all_names.extend(names_sorted)
        all_counts.extend(g_sorted.tolist())
        end = start + len(g_sorted)
        group_ranges.append((start, end, title, cmap_name, len(g_sorted), g_sorted.sum()))
        start = end
    all_counts = np.array(all_counts, dtype=np.int64)

    # 计算组内频率
    y_vals = np.zeros_like(all_counts, dtype=float)
    for (s, e, _title, _cmap, _ln, group_sum) in group_ranges:
        denom = group_sum if group_sum > 0 else 1
        y_vals[s:e] = all_counts[s:e] / denom

    # 颜色梯度
    colors = []
    for (s, e, _title, cmap_name, length, _gsum) in group_ranges:
        if cmap_name == 'CONTACTING_CUSTOM':
            colors.extend(get_contacting_colors(length))
        else:
            colors.extend(get_gradient_colors(length, cmap_name=cmap_name, start=0.95, end=0.25))

    plt.figure(figsize=(max(12, 0.38 * len(all_names)), 6))
    ax = plt.gca()
    ax.bar(range(len(all_names)), y_vals, color=colors, edgecolor='none')

    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=65, ha='right', fontsize=7)
    ax.set_ylabel('Frequency (per group)')
    ax.set_title('Per-group Relative Frequency', pad=22)

    # 分隔线与组名 (组名放在图内，避免与主标题重叠)
    ymax = ax.get_ylim()[1]
    label_y = ymax * 0.94  # 放在内部稍低位置
    for (s, e, title, _cmap, _ln, _gsum) in group_ranges:
        if s != 0:
            ax.axvline(x=s-0.5, color='#444', linewidth=0.8, linestyle='--')
        mid = (s + e - 1) / 2
        ax.text(mid, label_y, title, ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.subplots_adjust(top=0.85)  # 给标题留空间
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print('Saved:', out_path)


# ================== 主流程 ==================

def main():
    parser = argparse.ArgumentParser(description='Relation frequency plots (all + single-axis grouped)')
    parser.add_argument('-data_path', default='/home/Users/lcx/datasets/ag/', help='AG dataset root')
    parser.add_argument('-mode_train', default='train')
    parser.add_argument('-mode_test', default='test')
    parser.add_argument('-no_test', action='store_true', help='only train split')
    parser.add_argument('-out_prefix', default='output/relation_frequency', help='output prefix')
    parser.add_argument('-topk', type=int, default=None, help='top K for ALL figure (optional)')
    args = parser.parse_args()

    print('==> Load train dataset ...')
    ds_train = AG(mode=args.mode_train, datasize='large', data_path=args.data_path, filter_nonperson_box_frame=True, filter_small_box=False)
    if not args.no_test:
        print('==> Load test dataset ...')
        ds_test = AG(mode=args.mode_test, datasize='large', data_path=args.data_path, filter_nonperson_box_frame=True, filter_small_box=False)
    else:
        ds_test = None

    print('==> Counting frequencies ...')
    counter = collect_relation_frequencies(ds_train)
    if ds_test is not None:
        counter.update(collect_relation_frequencies(ds_test))

    rel_names = ds_train.relationship_classes
    print('Raw counts (unsorted):')
    for i, name in enumerate(rel_names):
        print(f'{i:02d} {name}: {counter.get(i, 0)}')

    # 图1: 全部（使用分组渐变 & 全局排序）
    all_out = args.out_prefix + '_all_groupcolors.png'
    build_bar_plot(counter, rel_names, all_out, topk=args.topk, dataset=ds_train)

    # 图2: 单轴三组（组内归一化频率 + 分组分隔）
    counts_full = np.array([counter.get(i, 0) for i in range(len(rel_names))], dtype=np.int64)
    groups_out = args.out_prefix + '_groups_csa.png'
    build_combined_groups_single_axis(counts_full, ds_train, groups_out)

    print('Done. Generated:')
    print('  ' + all_out)
    print('  ' + groups_out)


if __name__ == '__main__':
    main()
