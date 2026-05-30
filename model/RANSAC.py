import numpy as np

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    # 计算所有点到质心的平均距离
    dist = np.mean(np.linalg.norm(pts - mean, axis=1))
    
    # 防止除零
    scale = np.sqrt(2) / (dist + 1e-8)
    
    # 构建 3x3 的归一化变换矩阵 T
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    
    # 将坐标转为齐次坐标后进行矩阵乘法变换
    pts_homo = np.hstack((pts, np.ones((len(pts), 1))))
    pts_norm = (T @ pts_homo.T).T[:, :2]
    
    return pts_norm, T

def compute_homography_dlt(src_pts, dst_pts):
    """
    利用直接线性变换 (DLT) 和 SVD 奇异值分解，从 4 对点中计算单应性矩阵 H
    """
    # 1. 坐标归一化
    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)
    
    # 2. 构建 8x9 的矩阵 A (每对点提供 2 个方程)
    A = []
    for i in range(4):
        x, y = src_norm[i][0], src_norm[i][1]
        u, v = dst_norm[i][0], dst_norm[i][1]
        A.append([-x, -y, -1,  0,  0,  0, u*x, u*y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    
    # 3. 奇异值分解 SVD 解 Ah = 0
    # h 为 V 的最后一行 (对应最小奇异值的特征向量)
    _, _, Vh = np.linalg.svd(A)
    H_norm = Vh[-1, :].reshape(3, 3)
    
    # 4. 反归一化：H = T_dst^-1 * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    
    # 归一化 H 的最后一个元素为 1
    H = H / (H[2, 2] + 1e-8)
    return H

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
        
        # 2. 假设模型：计算单应性矩阵 H
        try:
            H = compute_homography_dlt(src_samp, dst_samp)
        except np.linalg.LinAlgError:
            # 如果这 4 个点共线导致 SVD 失败，直接跳过
            continue
            
        # 3. 全局验证：将所有 src 点重投影到 dst 空间
        # pred = H * x
        pred_homo = (H @ src_homo.T).T
        
        # 将齐次坐标转换回 2D 坐标 (x/w, y/w)
        w = pred_homo[:, 2:]
        w[w == 0] = 1e-8 # 防止除以 0
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