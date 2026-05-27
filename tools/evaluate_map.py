import json
import os

def compute_ap(ranked_list, good_list, ok_list, junk_list):
    """
    计算单个查询任务的Average Precision(AP)，严格对齐Oxford 5K官方C++算法。
    输入：
        ranked_list: 系统检索出的有序图像文件名列表 (['img1.jpg','img2.jpg', ...])
        good_list: 官方ground truth的good 列表
        ok_list: 官方ground truth的ok 列表
        junk_list: 官方ground truth的junk 列表
    输出: 
        浮点数，代表该查询的 AP 值
    """
    # 将列表转换为集合，O(1)复杂度极大加速查找
    positives=set(good_list + ok_list)
    junks=set(junk_list)
    
    n_positives=len(positives)
    if n_positives==0:
        return 0.0

    intersect_size=0.0
    j=0.0 # 记录有效的检索数量（排除 junk）
    
    old_recall=0.0
    old_precision=1.0
    ap=0.0

    for img_name in ranked_list:
        if img_name in junks:
            # 碰到 junk，当它不存在，直接跳过
            continue
            
        if img_name in positives:
            # 命中正样本
            intersect_size+=1.0
            
        j+=1.0
        
        # 计算当前的Recall和Precision
        recall=intersect_size/n_positives
        precision=intersect_size/j
        
        # 梯形法计算PR曲线下的面积
        ap+=(recall-old_recall)*((old_precision+precision)/2.0)
        
        old_recall=recall
        old_precision=precision
        
    return ap

def evaluate_system(groundtruth_path,system_results_dict):
    """
    计算整个系统的 mAP (Mean Average Precision)
    输入：
        groundtruth_path: parsed_groundtruth.json 的文件路径
        system_results_dict: 字典格式的系统检索结果{'query_name': ['rank1.jpg','rank2.jpg', ...]}
    输出：
        系统的mAP值
    """
    if not os.path.exists(groundtruth_path):
        raise FileNotFoundError(f"找不到 Ground Truth 文件: {groundtruth_path}")
        
    with open(groundtruth_path,'r',encoding='utf-8') as f:
        gt_data=json.load(f)
        
    ap_list=[]
    
    for query_name,gt_info in gt_data.items():
        # 如果系统结果里没有跑这个query，AP 记为 0
        if query_name not in system_results_dict:
            print(f"警告: 系统结果中缺失查询任务 '{query_name}'")
            ap_list.append(0.0)
            continue
            
        ranked_list=system_results_dict[query_name]
        
        # 调用核心算分函数
        ap=compute_ap(
            ranked_list=ranked_list,
            good_list=gt_info.get('good',[]),
            ok_list=gt_info.get('ok',[]),
            junk_list=gt_info.get('junk',[])
        )
        
        ap_list.append(ap)
        # print(f"[{query_name}] AP: {ap:.4f}")
        
    # 计算均值得到 mAP
    mAP=sum(ap_list)/len(ap_list) if ap_list else 0.0
    return mAP

# if __name__ == '__main__':
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     GT_FILE = os.path.join(BASE_DIR, 'data', 'parsed_groundtruth.json')
    
#     mock_system_results = {
#         'all_souls_1': [
#             'all_souls_000091.jpg', # good (Hit)
#             'all_souls_000026.jpg', # good (Hit)
#             'oxford_000001.jpg',    # unrelated (False Positive)
#             'all_souls_000220.jpg', # junk (Ignored)
#             'oxford_003410.jpg',    # good (Hit)
#         ]
#     }
    
#     if os.path.exists(GT_FILE):
#         print("开始评测系统...")
#         # 为了演示，我们先拿一部分 GT 来测，实际中会评测所有的 55 个 query
#         mAP_score = evaluate_system(GT_FILE, mock_system_results)
#         print(f"\n评测完成！系统的 mAP 得分为: {mAP_score:.4f}")
#         print("(注：得分极低是因为模拟数据只给出了5张图，而实际正确的有几十张)")
#     else:
#         print("请先运行 parse_groundtruth.py 生成 JSON 标注文件！")