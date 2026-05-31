import numpy as np
import cv2

def pure_python_ransac_homography(src_pts, dst_pts, threshold=5.0, max_iters=1000):
    """
    返回: 最大内点数量
    """
    num_pts = len(src_pts)
    best_inliers = 0
    
    # 将 src_pts 提前转为齐次坐标，方便后续批量重投影
    src_homo = np.hstack((src_pts, np.ones((num_pts, 1))))
    
    # 如果点数少于4，直接失败
    if num_pts < 4:
        return 0

    for _ in range(max_iters):
        # 1. 随机抽样 4 对点
        idx = np.random.choice(num_pts, 4, replace=False)
        src_samp = src_pts[idx]
        dst_samp = dst_pts[idx]
        
        # 2. 假设模型：调用 OpenCV 计算单应性矩阵 H
        H, mask = cv2.findHomography(
            src_samp.astype(np.float32),
            dst_samp.astype(np.float32),
            method=0  # 使用普通 DLT，不启用 RANSAC/LMEDS，与原逻辑保持一致
        )
        
        # 如果点共线等退化情况导致 H 计算失败
        if H is None:
            continue
            
        # 3. 全局验证：将所有 src 点重投影到 dst 空间
        # pred = H * x
        pred_homo = (H @ src_homo.T).T
        
        # 将齐次坐标转换回 2D 坐标 (x/w, y/w)
        w = pred_homo[:, 2:]
        w[w == 0] = 1e-8  # 防止除以 0
        pred_pts = pred_homo[:, :2] / w
        
        # 4. 计算重投影误差 (欧氏距离) 并统计内点
        errors = np.linalg.norm(pred_pts - dst_pts, axis=1)
        inliers_count = int(np.sum(errors < threshold))
        
        # 5. 记录最优共识
        if inliers_count > best_inliers:
            best_inliers = inliers_count
            
            # 提早熔断优化：如果内点率已经极高（例如 > 95%），可以提前结束迭代
            if best_inliers > num_pts * 0.95:
                break
                
    return best_inliers