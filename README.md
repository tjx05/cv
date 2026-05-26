# cv
cv课设——图像检索

 Image-Retrieval-System
 ┣  data/                 # 1. 数据驱动层 (Data Layer)
 ┃ ┣  gt_files_170407/    # 官方原始 Ground Truth 文本标注数据
 ┃ ┣  images/             # 图像数据库 (如 Oxford 5k)
 ┃ ┗  parsed_groundtruth.json # 经脚本清洗后的标准化 GT 标注文件
 ┃
 ┣  model/                # 2. 核心算法层 (Model Layer)
 ┃ ┗  ROOTSIFT_feature.py # 核心特征类：封装了基于 DoG 的 RootSIFT 提取与数学映射逻辑
 ┃
 ┣  scripts/              # 3. 业务逻辑层 (Pipeline Scripts)
 ┃ ┣  parse_groundtruth.py    # 预处理：解析官方 GT 文件并生成 JSON
 ┃ ┣  feature_extract.py      # 离线阶段：全库 SIFT 描述子并行提取
 ┃ ┣  extract_keypoints.py    # 离线阶段：全库关键点坐标 (x,y) 提取 (用于后续 RANSAC)
 ┃ ┣  build_vocab.py          # 离线阶段：使用 MiniBatchKMeans 构建视觉词典 (BoW)
 ┃ ┣  build_index.py          # 离线阶段：计算 TF-IDF 权重并构建倒排索引库
 ┃ ┃
 ┃ ┣  baseline_retrieval.py   # 在线阶段：纯 BoW 基线检索测试
 ┃ ┣  aqe_retrieval.py        # 在线阶段：BoW + AQE (平均查询扩展) 测试
 ┃ ┣  ransac_retrieval.py     # 在线阶段：BoW + RANSAC 空间校验测试
 ┃ ┣  ransac_aqe_retrieval.py # 在线阶段：终极融合架构 (BoW + RANSAC 初审 + AQE + RANSAC 终审)
 ┃ ┃
 ┃ ┗  visualize.py            # 可视化：生成带红绿外框的检索结果对比图
 ┃
 ┣  tools/                # 4. 评估与工具层 (Tools)
 ┃ ┗  evaluate_map.py     # 测评脚本：严格计算系统平均精度均值 (mAP)
 ┃
 ┣ outputs/              # 5. 缓存与产出层 (Outputs)
 ┃ ┗ (自动生成)              # 存放生成的 .npy 特征、vocab.pkl、index.pkl 及最终可视化图片
 ┃
 ┣  config.py             # 6. 全局配置中心 (Configuration)
 ┗  zj.ipynb              # 7. 实验草稿本 (Jupyter Notebook 探针测试)

 ## 快速开始
```bash
# 1. 解析官方 Ground Truth 标注数据
python scripts/parse_groundtruth.py

# 2. 并行提取全图特征与关键点坐标
python scripts/feature_extract.py
python scripts/extract_keypoints.py

# 3. 训练视觉词典 (耗时受 K 值及采样率影响)
python scripts/build_vocab.py

# 4. 计算 TF-IDF 权重并构建倒排索引库
python scripts/build_index.py

# 5. 运行检索架构并计算 mAP (四选一进行消融测试)
python scripts/baseline_retrieval.py      # 测试纯 BoW 基线
python scripts/aqe_retrieval.py           # 测试 BoW + AQE
python scripts/ransac_retrieval.py        # 测试 BoW + RANSAC
python scripts/ransac_aqe_retrieval.py    # 测试终极融合架构

# 6. 生成检索结果对比图 (输出至 outputs 目录)
python scripts/visualize.py