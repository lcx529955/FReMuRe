import os
import cv2
import numpy as np
import torch
import math
from typing import List, Dict, Tuple

_PALETTE = (
    np.array([
        255,56,56, 255,159,56, 255,112,31, 255,178,29, 207,210,49,
        72,249,10, 146,204,23, 61,219,134, 26,147,52, 0,149,255,
        212,188,0, 255,99,164, 0,60,91, 112,255,72, 255,0,255, 149,183,0,
        255,255,0, 99,149,255, 149,0,255, 207,0,255, 0,207,255
    ]).reshape(-1,3)
)

def _color(i:int):
    return tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])

def draw_objects(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, class_names: List[str]):
    h,w = img.shape[:2]
    for b,l in zip(boxes, labels):
        x1,y1,x2,y2 = b.astype(int)
        if x2 <= x1 or y2 <= y1:  # skip invalid
            continue
        x1 = max(0,min(x1,w-1)); x2 = max(0,min(x2,w-1)); y1 = max(0,min(y1,h-1)); y2 = max(0,min(y2,h-1))
        cv2.rectangle(img,(x1,y1),(x2,y2),_color(int(l)),2)
        name = class_names[int(l)] if 0 <= int(l) < len(class_names) else str(int(l))
        cv2.putText(img,name,(x1,max(0,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.5,_color(int(l)),1,cv2.LINE_AA)
    return img

def draw_objects_blue(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, class_names: List[str]):
    """统一蓝色目标绘制; 容错: 跳过不符合(4,)形状的box"""
    h,w = img.shape[:2]
    color_box = (255,0,0)
    text_color = (0,0,0)
    if boxes is None:
        return img
    for idx,(b,l) in enumerate(zip(boxes, labels)):
        b_arr = np.asarray(b)
        if b_arr.size < 4:
            continue
        if b_arr.ndim == 2 and b_arr.shape[1] == 4:  # 多框(例如多人)只取第一框
            b_arr = b_arr[0]
        if b_arr.shape[0] != 4:
            continue
        x1,y1,x2,y2 = b_arr.astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        x1 = max(0,min(x1,w-1)); x2 = max(0,min(x2,w-1)); y1 = max(0,min(y1,h-1)); y2 = max(0,min(y2,h-1))
        cv2.rectangle(img,(x1,y1),(x2,y2),color_box,2)
        name = class_names[int(l)] if 0 <= int(l) < len(class_names) else str(int(l))
        cv2.putText(img,name,(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,2,cv2.LINE_AA)
    return img

def build_relations(pred, dataset, score_thresh: float, topk: int):
    """从模型输出构建三类关系列表。
    规则:
      attention: human -> object (保持原方向)
      spatial:   object  -> human (反转 pair_idx 顺序)
      contacting: human -> object (保持原方向)
    过滤: 按 score_thresh 过滤；若某一类全部被过滤则为每个 pair 仍保留其最大概率项(保障三类至少出现)。
    topk: 若 >0, 先整体按分数排序再裁剪；如果裁剪后某类别缺失, 强制补回该类别最高分若干(不超过原 size)。
    返回: list[(score, subj_global_idx, obj_global_idx, predicate_name)]
    """
    pair_idx = pred['pair_idx']  # (R,2) global 索引 (subject, object) 这里 subject=human, object=object (数据集中约定)
    att_dist = torch.softmax(pred['attention_distribution'], dim=1)
    spa_dist = torch.softmax(pred['spatial_distribution'], dim=1)
    con_dist = torch.softmax(pred['contacting_distribution'], dim=1)

    relations_att = []
    relations_spa = []
    relations_con = []

    # Attention: 单标签 (取 argmax 后判断阈值)
    att_scores, att_cls = att_dist.max(1)
    for (p, c, s) in zip(pair_idx, att_cls, att_scores):
        sc = float(s.cpu())
        if sc >= score_thresh:
            relations_att.append((sc, int(p[0]), int(p[1]), dataset.attention_relationships[int(c)]))
    # 若全被过滤, 仍保留最高得分 (强制出现)
    if len(relations_att) == 0 and att_dist.numel() > 0:
        for i, p in enumerate(pair_idx):
            s_val, c_idx = att_dist[i].max(0)
            relations_att.append((float(s_val.cpu()), int(p[0]), int(p[1]), dataset.attention_relationships[int(c_idx)]))

    # Spatial: 多标签 (object -> human) 方向反转
    for p, probs in zip(pair_idx, spa_dist):
        keep = (probs >= score_thresh).nonzero(as_tuple=False).view(-1)
        if keep.numel() == 0:  # 强制保留最大
            c = probs.argmax()
            relations_spa.append((float(probs[c].cpu()), int(p[1]), int(p[0]), dataset.spatial_relationships[int(c)]))
        else:
            for c in keep:
                relations_spa.append((float(probs[c].cpu()), int(p[1]), int(p[0]), dataset.spatial_relationships[int(c)]))

    # Contacting: 多标签 (human -> object)
    for p, probs in zip(pair_idx, con_dist):
        keep = (probs >= score_thresh).nonzero(as_tuple=False).view(-1)
        if keep.numel() == 0:  # 强制保留最大
            c = probs.argmax()
            relations_con.append((float(probs[c].cpu()), int(p[0]), int(p[1]), dataset.contacting_relationships[int(c)]))
        else:
            for c in keep:
                relations_con.append((float(probs[c].cpu()), int(p[0]), int(p[1]), dataset.contacting_relationships[int(c)]))

    # 合并并排序
    all_rels = relations_att + relations_spa + relations_con
    all_rels.sort(key=lambda x: x[0], reverse=True)

    original_counts = {
        'att': len(relations_att),
        'spa': len(relations_spa),
        'con': len(relations_con)
    }

    if topk > 0 and len(all_rels) > topk:
        cut = all_rels[:topk]
        # 确认三类是否都在
        have_att = any(r[3] in dataset.attention_relationships for r in cut)
        have_spa = any(r[3] in dataset.spatial_relationships for r in cut)
        have_con = any(r[3] in dataset.contacting_relationships for r in cut)
        if not (have_att and have_spa and have_con):
            # 补齐缺失类别: 取该类别内最高分关系加入(若不在 cut)
            needed = []
            if not have_att and relations_att:
                needed.append(relations_att[0])
            if not have_spa and relations_spa:
                needed.append(relations_spa[0])
            if not have_con and relations_con:
                needed.append(relations_con[0])
            # 合并再去重再按分数截断
            idset = set()
            merged = []
            def key(r): return (r[1], r[2], r[3])
            for r in cut + needed:
                k = key(r)
                if k not in idset:
                    idset.add(k)
                    merged.append(r)
            merged.sort(key=lambda x: x[0], reverse=True)
            if len(merged) > topk:
                merged = merged[:topk]
            all_rels = merged
        else:
            all_rels = cut

    # 调试打印（可注释）
    print(f"[build_relations] kept: att={original_counts['att']} spa={original_counts['spa']} con={original_counts['con']} total={len(all_rels)} (after topk={topk})")
    return all_rels

def render_graph(frame_boxes: np.ndarray, frame_labels: np.ndarray, frame_rels, frame_global_ids: np.ndarray,
                 all_labels: np.ndarray, class_names: List[str], img_h: int,
                 rel_eval: Dict[Tuple[int,int,str], bool]=None,
                 is_gt: bool=False) -> np.ndarray:
    """渲染关系图:
    对象统一蓝色; 正确关系绿色; 错误关系红色; GT 图全部绿色。
    frame_rels: list[(score, subj_global, obj_global, predicate_name)] (预测) 或 (None, subj_global, obj_global, predicate_name) (GT)
    rel_eval: 可选, key=(subj_local_idx,obj_local_idx,predicate_name)->True/False
    is_gt: 是否为 GT 图
    """
    N = frame_boxes.shape[0]
    graph_w = max(600, int(img_h*0.9))
    canvas = np.ones((img_h, graph_w, 3), dtype=np.uint8)*255
    if N == 0:
        return canvas
    # 布局
    positions = []
    if N <= 12:
        cx, cy = graph_w//2, img_h//2
        radius = int(min(graph_w, img_h)*0.40)
        for k in range(N):
            ang = 2*math.pi * k / N if N>0 else 0
            x = int(cx + radius * math.cos(ang))
            y = int(cy + radius * math.sin(ang))
            positions.append((x,y))
    else:
        cols = math.ceil(math.sqrt(N))
        rows = math.ceil(N/cols)
        cell_w = graph_w / (cols+1)
        cell_h = img_h / (rows+1)
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k>=N: break
                x = int((c+1)*cell_w)
                y = int((r+1)*cell_h)
                positions.append((x,y)); k+=1
    node_r = 26 if N <= 12 else max(14, int(300/max(N,1)))
    font_scale_node = 0.55 if node_r >= 22 else 0.5
    # 建局部映射
    local_map = {int(g): i for i,g in enumerate(frame_global_ids.tolist())}
    # 聚合关系
    edge_predicates: Dict[Tuple[int,int], List[str]] = {}
    rel_eval = rel_eval or {}
    bad_local = 0
    for rel in frame_rels:
        if not isinstance(rel, (list, tuple)) or len(rel) != 4:
            bad_local += 1
            continue
        _, sg, og, pred_name = rel
        if sg in local_map and og in local_map and sg!=og:
            si = local_map[sg]; oi = local_map[og]
            edge_predicates.setdefault((si,oi), []).append(pred_name)
    if bad_local>0:
        print(f"[render_graph] 跳过异常关系条目 {bad_local} 个")
    # 画节点
    OBJ_COLOR_FILL = (255,0,0)  # 蓝
    for idx,(x,y) in enumerate(positions):
        cv2.circle(canvas,(x,y), node_r, OBJ_COLOR_FILL, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas,(x,y), node_r, (0,0,0), 2, lineType=cv2.LINE_AA)
        label_id = int(frame_labels[idx])
        cls_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        cx, cy = graph_w//2, img_h//2
        vx, vy = x - cx, y - cy
        norm = math.hypot(vx, vy) + 1e-6
        ox = x + int(vx / norm * (node_r + 8))
        oy = y + int(vy / norm * (node_r + 8))
        short = cls_name if len(cls_name) <= 12 else cls_name[:11] + '…'
        cv2.putText(canvas, short, (ox - node_r, oy), cv2.FONT_HERSHEY_SIMPLEX, font_scale_node, (0,0,0), 1, cv2.LINE_AA)
    # 画边
    GREEN = (0,200,0)
    RED = (0,0,255)
    for (si,oi), plist in edge_predicates.items():
        sx,sy = positions[si]; ox,oy = positions[oi]
        dx = ox - sx; dy = oy - sy
        dist = math.hypot(dx,dy)
        if dist < 1e-4: continue
        ux,uy = dx/dist, dy/dist
        start = (int(sx + ux*node_r), int(sy + uy*node_r))
        end = (int(ox - ux*node_r), int(oy - uy*node_r))
        # 分类 predicates
        correct_list = []
        wrong_list = []
        for p in plist:
            key = (si,oi,p)
            ok = True if is_gt else rel_eval.get(key, False)
            if ok: correct_list.append(p)
            else: wrong_list.append(p)
        arrow_color = GREEN if (is_gt or (len(wrong_list)==0 and len(correct_list)>0)) else RED
        cv2.arrowedLine(canvas, start, end, arrow_color, 3, tipLength=0.18)
        # 文本绘制: 先正确(绿) 后错误(红)
        mx = int((start[0]+end[0])/2); my = int((start[1]+end[1])/2)
        perp = (-uy, ux)
        base_x = int(mx + perp[0]*14); base_y = int(my + perp[1]*14)
        line_gap = 16
        if correct_list:
            txt = '|'.join(sorted(set(correct_list)))
            if len(txt)>40: txt = txt[:39]+'…'
            cv2.putText(canvas, txt, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
            base_y += line_gap
        if wrong_list:
            txt2 = '|'.join(sorted(set(wrong_list)))
            if len(txt2)>40: txt2 = txt2[:39]+'…'
            cv2.putText(canvas, txt2, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
    return canvas

def _iou_matrix(a: np.ndarray, b: np.ndarray):
    # a:(N,4) b:(M,4) boxes in xyxy
    if a.size==0 or b.size==0:
        return np.zeros((a.shape[0], b.shape[0]))
    area_a = (a[:,2]-a[:,0]).clip(min=0) * (a[:,3]-a[:,1]).clip(min=0)
    area_b = (b[:,2]-b[:,0]).clip(min=0) * (b[:,3]-b[:,1]).clip(min=0)
    ious = np.zeros((a.shape[0], b.shape[0]))
    for i, box in enumerate(a):
        xx1 = np.maximum(float(box[0]), b[:,0])
        yy1 = np.maximum(float(box[1]), b[:,1])
        xx2 = np.minimum(float(box[2]), b[:,2])
        yy2 = np.minimum(float(box[3]), b[:,3])
        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        inter = w*h
        denom = area_a[i] + area_b - inter + 1e-6
        ious[i] = inter / denom
    return ious

def _build_gt_frame(dataset, video_index: int, frame_local_idx: int):
    frame_gt = dataset.gt_annotations[video_index][frame_local_idx]
    gt_boxes_list = []
    gt_labels_list = []
    # person bbox 可能是 (K,4); 取第一个
    person_bbox = frame_gt[0]['person_bbox']
    person_bbox = np.asarray(person_bbox)
    if person_bbox.ndim == 2 and person_bbox.shape[1] == 4:
        if person_bbox.shape[0] == 0:
            # 无人, 返回空
            return np.zeros((0,4)), np.zeros((0,)), []
        person_box = person_bbox[0]
    else:
        person_box = person_bbox.reshape(-1)
    if person_box.size != 4:
        return np.zeros((0,4)), np.zeros((0,)), []
    gt_boxes_list.append(person_box)
    gt_labels_list.append(1)
    rels = []
    # objects
    for obj in frame_gt[1:]:
        box = np.asarray(obj['bbox']).reshape(-1)
        if box.size != 4:
            continue
        gt_boxes_list.append(box)
        gt_labels_list.append(obj['class'])
    gt_boxes = np.vstack(gt_boxes_list) if gt_boxes_list else np.zeros((0,4))
    gt_labels = np.array(gt_labels_list) if gt_labels_list else np.zeros((0,))
    # relations
    for idx_obj, obj in enumerate(frame_gt[1:], start=1):
        # attention
        if 'attention_relationship' in obj:
            for att in obj['attention_relationship'].tolist():
                predicate = dataset.attention_relationships[att]
                rels.append((None, 0, idx_obj, predicate))
        # spatial (object -> human)
        if 'spatial_relationship' in obj:
            for spa in obj['spatial_relationship'].tolist():
                predicate = dataset.spatial_relationships[spa]
                rels.append((None, idx_obj, 0, predicate))
        # contacting (human -> object)
        if 'contacting_relationship' in obj:
            for con in obj['contacting_relationship'].tolist():
                predicate = dataset.contacting_relationships[con]
                rels.append((None, 0, idx_obj, predicate))
    return gt_boxes, gt_labels, rels

def _visualize_single_frame(f_idx, frames, video_name, out_dir, boxes_all, labels_all, rels_all,
                           dataset, conf, video_index, frame_type=""):
    """可视化单帧的辅助函数"""
    global_ids = np.where(boxes_all[:,0]==f_idx)[0]
    if global_ids.size==0:
        return False

    frame_boxes = boxes_all[global_ids,1:5]
    frame_labels = labels_all[global_ids]
    frame_rels = [r for r in rels_all if isinstance(r,(list,tuple)) and len(r)==4 and r[1] in global_ids and r[2] in global_ids and int(boxes_all[r[1],0])==f_idx]

    # GT 信息
    try:
        frame_local_idx = f_idx  # frames 顺序与 gt_annotations 顺序一致
        gt_boxes, gt_labels, gt_rels = _build_gt_frame(dataset, video_index, frame_local_idx)
    except Exception:
        gt_boxes = np.zeros((0,4)); gt_labels = np.zeros((0,)); gt_rels=[]

    # 读取帧图像
    img_path = str(os.path.join(dataset.frames_path, frames[f_idx]))
    if not os.path.exists(img_path):
        return False
    img_ori = cv2.imread(img_path)
    if img_ori is None:
        return False

    # 预测目标图
    pred_img = img_ori.copy()
    draw_objects_blue(pred_img, frame_boxes, frame_labels, dataset.object_classes)

    # GT 目标图
    gt_img = img_ori.copy()
    draw_objects_blue(gt_img, gt_boxes, gt_labels, dataset.object_classes)

    # 关系正确性评估 (pred -> gt)
    rel_eval = {}
    if gt_boxes.shape[0]>0 and frame_boxes.shape[0]>0:
        ious = _iou_matrix(frame_boxes, gt_boxes)
        mapping = ious.argmax(1)
        mapping_ious = ious[np.arange(ious.shape[0]), mapping]
        mapping[mapping_ious < 0.5] = -1  # 无匹配
        gt_rel_set = set((sg, og, pn) for (_, sg, og, pn) in gt_rels if isinstance(pn,str))
        local_map = {int(g): i for i,g in enumerate(global_ids.tolist())}
        for rel in frame_rels:
            if not (isinstance(rel,(list,tuple)) and len(rel)==4):
                continue
            (score, sg_global, og_global, pn) = rel
            if sg_global not in local_map or og_global not in local_map:
                continue
            si = local_map[sg_global]; oi = local_map[og_global]
            if si >= mapping.shape[0] or oi >= mapping.shape[0]:
                continue
            gt_si = mapping[si]; gt_oi = mapping[oi]
            key_local = (si, oi, pn)
            if gt_si>=0 and gt_oi>=0 and (gt_si, gt_oi, pn) in gt_rel_set:
                rel_eval[key_local] = True
            else:
                rel_eval[key_local] = False

    # 预测关系图
    pred_graph = render_graph(frame_boxes, frame_labels, frame_rels, global_ids, labels_all, dataset.object_classes, pred_img.shape[0], rel_eval=rel_eval, is_gt=False)
    gt_global_ids = np.arange(len(gt_boxes))
    gt_graph = render_graph(gt_boxes, gt_labels, gt_rels, gt_global_ids, gt_labels, dataset.object_classes, gt_img.shape[0], rel_eval=None, is_gt=True)

    H = pred_img.shape[0]
    if pred_graph.shape[0]!=H: pred_graph = cv2.resize(pred_graph, (pred_graph.shape[1], H))
    if gt_img.shape[0]!=H: gt_img = cv2.resize(gt_img, (gt_img.shape[1], H))
    if gt_graph.shape[0]!=H: gt_graph = cv2.resize(gt_graph, (gt_graph.shape[1], H))

    # 2x2 排版: 上行 预测(目标+graph); 下行 GT(目标+graph)
    row_pred = np.hstack([pred_img, pred_graph])
    row_gt = np.hstack([gt_img, gt_graph])

    # 对齐宽度: 以最大宽度为准, 另一行右侧填白
    W = max(row_pred.shape[1], row_gt.shape[1])
    def pad_row(row):
        if row.shape[1] == W:
            return row
        pad_w = W - row.shape[1]
        pad = np.ones((row.shape[0], pad_w, 3), dtype=row.dtype)*255
        return np.hstack([row, pad])
    row_pred = pad_row(row_pred)
    row_gt = pad_row(row_gt)
    combined = np.vstack([row_pred, row_gt])

    save_name = os.path.join(out_dir, f"{video_name}__{frames[f_idx].replace('/', '_')}__{frame_type}.jpg")
    ok = cv2.imwrite(save_name, combined)
    if not ok:
        print(f"[visualize] 保存失败: {save_name}")
        return False
    else:
        print(f"[visualize] 保存成功: {save_name}  关系数: pred={len(frame_rels)} gt={len(gt_rels)} ({frame_type})")
        return True

def visualize_video(pred, dataset, video_index: int, im_info: torch.Tensor, conf):
    # 构建并保存两张图: 关系数最多的帧和关系数最少的帧
    frames = dataset.video_list[video_index]
    video_name = dataset.valid_video_names[video_index]
    out_dir = os.path.join(conf.save_path, conf.vis_out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if conf.mode == 'predcls':
        labels_all = pred['labels'].cpu().numpy()
    else:
        labels_all = pred['pred_labels'].cpu().numpy()

    boxes_all = pred['boxes'].cpu().numpy()  # (N,5) frame,x1,y1,x2,y2
    rels_all = build_relations(pred, dataset, conf.vis_score_thresh, conf.vis_topk)

    # 过滤非法关系项
    valid_rels = []
    bad_cnt = 0
    bad_samples = []
    for r in rels_all:
        if isinstance(r,(list,tuple)) and len(r)==4:
            valid_rels.append(r)
        else:
            bad_cnt += 1
            if len(bad_samples) < 5:
                bad_samples.append(r)
    if bad_cnt>0:
        print(f"[visualize] 丢弃异常关系条目: {bad_cnt}, 示例: {bad_samples}")
    rels_all = valid_rels

    # 统计每帧关系数
    frame_rel_count = {i:0 for i in range(len(frames))}
    for r in rels_all:
        if not (isinstance(r,(list,tuple)) and len(r)==4):
            continue
        _score, subj_idx, obj_idx, _pname = r
        if subj_idx >= boxes_all.shape[0] or obj_idx >= boxes_all.shape[0]:
            continue
        f_subj = int(boxes_all[subj_idx,0])
        f_obj = int(boxes_all[obj_idx,0])
        if f_subj == f_obj and 0 <= f_subj < len(frames):
            frame_rel_count[f_subj] += 1

    # 找到关系数最多和最少的帧
    if not frame_rel_count:
        print(f"[visualize] 视频 {video_name} 没有关系数据")
        return

    # 只考虑有目标的帧
    frames_with_objects = set()
    for i in range(len(frames)):
        if np.sum(boxes_all[:,0]==i) > 0:
            frames_with_objects.add(i)

    if not frames_with_objects:
        print(f"[visualize] 视频 {video_name} 没有检测到目标")
        return

    # 过滤关系计数，只考虑有目标的帧
    valid_frame_rel_count = {i: count for i, count in frame_rel_count.items() if i in frames_with_objects}

    if not valid_frame_rel_count:
        print(f"[visualize] 视频 {video_name} 没有有效帧")
        return

    # 找到关系数最多和最少的帧
    max_rel_frame = max(valid_frame_rel_count.items(), key=lambda kv: kv[1])[0]
    min_rel_frame = min(valid_frame_rel_count.items(), key=lambda kv: kv[1])[0]

    max_rel_count = valid_frame_rel_count[max_rel_frame]
    min_rel_count = valid_frame_rel_count[min_rel_frame]

    print(f"[visualize] 视频 {video_name}: 最多关系帧{max_rel_frame}({max_rel_count}个), 最少关系帧{min_rel_frame}({min_rel_count}个)")

    # 可视化两帧
    success_count = 0
    max_type = f"max_rel_{max_rel_count}"
    if _visualize_single_frame(max_rel_frame, frames, video_name, out_dir, boxes_all, labels_all, rels_all,
                              dataset, conf, video_index, max_type):
        success_count += 1

    # 如果最多和最少是同一帧，就不重复保存
    if max_rel_frame != min_rel_frame:
        min_type = f"min_rel_{min_rel_count}"
        if _visualize_single_frame(min_rel_frame, frames, video_name, out_dir, boxes_all, labels_all, rels_all,
                                  dataset, conf, video_index, min_type):
            success_count += 1
    else:
        print(f"[visualize] 视频 {video_name}: 最多关系帧和最少关系帧是同一帧，只保存一张图")

    print(f"[visualize] 视频 {video_name} 完成，成功保存 {success_count} 张图")
