import os
import json

def parse_oxford_groundtruth(gt_dir, output_path):
    """
    解析Oxford 5K原始txt标注文件，生成统一格式的字典并保存为JSON。
    """
    print(f"开始解析Ground Truth文件夹:{gt_dir}")
    dataset_info={}
    
    # 检查目录是否存在
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"找不到目录: {gt_dir}。请检查路径设置！")

    # 遍历所有txt文件
    for filename in os.listdir(gt_dir):
        if not filename.endswith('.txt'):
            continue
            
        # 提取query_name(例如从 'all_souls_1_query.txt'提取'all_souls_1')
        parts=filename.replace('.txt','').rsplit('_',1)
        if len(parts)!=2:
            continue
            
        query_name,label_type=parts[0],parts[1]
        
        # 初始化该query的字典结构
        if query_name not in dataset_info:
            dataset_info[query_name]={'good':[],'ok':[],'junk':[]}

        filepath=os.path.join(gt_dir, filename)

        # 解析查询文件 (获取Bounding Box和Query图片名)
        if label_type=='query':
            with open(filepath,'r') as f:
                content=f.read().strip().split(' ')
                # 清洗掉官方原版文件名有时带的oxc1_前缀，并统一加上 .jpg 后缀
                img_name=content[0].replace('oxc1_','')+'.jpg'
                
                # 提取边界框坐标:[xmin,ymin,xmax,ymax]
                bbox=[float(x) for x in content[1:]] 
                
                dataset_info[query_name]['query_img']=img_name
                dataset_info[query_name]['bbox']=bbox
                
        # 2. 解析答案文件(good,ok,junk)
        elif label_type in ['good','ok','junk']:
            with open(filepath,'r') as f:
                # 读取每行，过滤空行，并统一加上 .jpg 后缀
                img_list=[line.strip()+'.jpg' for line in f.readlines() if line.strip()]
                dataset_info[query_name][label_type]=img_list

    # 将解析好的完整字典保存为本地 JSON 文件
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(dataset_info,f,indent=4,ensure_ascii=False)
        
    print(f"解析完成！共成功处理了{len(dataset_info)}个查询任")
    print(f"结构化数据已保存至: {output_path}")
    
    return dataset_info

if __name__=='__main__':
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GT_DIRECTORY=os.path.join(BASE_DIR,'data','gt_files_170407')
    OUTPUT_FILE=os.path.join(BASE_DIR,'data','parsed_groundtruth.json')
    
    # 执行解析
    info=parse_oxford_groundtruth(GT_DIRECTORY,OUTPUT_FILE)
    
    # 打印其中一个验证结果，确保数据是对的
    test_query='all_souls_1'
    if test_query in info:
        print(f"查询原图: {info[test_query]['query_img']}")
        print(f"查询坐标框: {info[test_query]['bbox']}")
        print(f"Good 样本数: {len(info[test_query]['good'])}")
        print(f"Ok 样本数: {len(info[test_query]['ok'])}")
        print(f"Junk 样本数: {len(info[test_query]['junk'])}")