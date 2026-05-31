import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n" + "="*50)
    print(f" 🚀 正在运行：{script_name}")
    print("="*50)
    
    start_time = time.time()
    
    # 执行脚本
    subprocess.run([sys.executable, script_name], check=True)
    
    end_time = time.time()
    cost_seconds = end_time - start_time
    
    # 换算成分钟和秒
    mins = int(cost_seconds // 60)
    secs = cost_seconds % 60
    
    print(f"\n✅ [{script_name}] 执行完毕！")
    print(f"⏱️ 单独耗时：{mins} 分 {secs:.2f} 秒")

if __name__ == "__main__":
    print("========================================")
    print("    🔥  CV 检索全流程及消融实验启动  ")
    print("========================================")
    
    total_start = time.time()

    # ==========================================
    # 第一阶段：离线特征提取与建库
    # ==========================================
    run_script("scripts/feature_extract.py")
    run_script("scripts/extract_keypoints.py")
    run_script("scripts/build_vocab.py")
    run_script("scripts/build_index.py")
    
    # ==========================================
    # 第二阶段：在线检索评测与性能对比
    # ==========================================
    run_script("scripts/baseline_retrieval.py")      # 测试纯 BoW 基线
    run_script("scripts/aqe_retrieval.py")           # 测试 BoW + AQE
    run_script("scripts/ransac_retrieval.py")        # 测试 BoW + RANSAC
    run_script("scripts/ransac_aqe_retrieval.py")    # 测试终极融合架构

    total_end = time.time()
    total_cost = total_end - total_start
    total_mins = int(total_cost // 60)
    total_secs = total_cost % 60

    print("\n" + "★"*50)
    print("    🎉  全流程执行大满贯！")
    print(f"    ⏳  总计耗时：{total_mins} 分 {total_secs:.2f} 秒")
    print("★"*50)