import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def organize_dataset(root_dir):
    """组织数据集结构，返回病例列表"""
    # 假设数据结构：
    # root_dir/
    # ├── CT/
    # │   ├── case_001.nii.gz
    # │   └── case_002.nii.gz
    # └── MR/
    #     ├── case_001.nii.gz
    #     └── case_002.nii.gz
    
    ct_files = glob.glob(os.path.join(root_dir, 'CT', '*.npy'))
    mr_files = glob.glob(os.path.join(root_dir, 'MR', '*.npy'))
    
    # 提取唯一病例ID（假设文件名格式为case_id.nii.gz）
    ct_case_ids = [os.path.basename(f).split('.')[0] for f in ct_files]
    mr_case_ids = [os.path.basename(f).split('.')[0] for f in mr_files]

    # 验证病例匹配性
    common_cases = set(ct_case_ids) & set(mr_case_ids)
    if len(common_cases) != len(ct_case_ids) or len(common_cases) != len(mr_case_ids):
        print("警告：存在不匹配的CT/MR病例")
    
    return sorted(list(common_cases))

def split_dataset(case_ids, test_size=0.2, random_state=42):
    """划分训练集和测试集"""
    train_ids, test_ids = train_test_split(
        case_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return train_ids, test_ids

def create_dataset_structure(output_dir):
    """创建标准目录结构"""
    os.makedirs(os.path.join(output_dir, 'train', 'CT'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'MR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'CT'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'MR'), exist_ok=True)

def copy_files(src_root, dst_root, case_ids, modality):
    """复制指定病例的.npy文件"""
    for case_id in case_ids:
        # 源路径和目标路径使用.npy后缀
        src_path = os.path.join(src_root, modality, f"{case_id}.npy")
        dst_dir = os.path.join(dst_root, modality)
        dst_path = os.path.join(dst_dir, f"{case_id}.npy")
        
        # 确保目标目录存在
        os.makedirs(dst_dir, exist_ok=True)
        
        if os.path.exists(src_path):
            try:
                shutil.copy(src_path, dst_path)
                print(f"成功复制 {modality}/{case_id}.npy 到 {dst_path}")
            except Exception as e:
                print(f"复制失败 {src_path}: {str(e)}")
        else:
            print(f"警告：找不到 {modality} 文件 {case_id}.npy")

def main():
    # 参数配置
    # input_dir = "/path/to/raw_dataset"  # 原始数据目录
    # output_dir = "/path/to/split_dataset"  # 划分后输出目录
    input_dir = '/home/featurize/data/Task1_Liu'
    output_dir = '/home/featurize/data/Task1_Liu_split'
    test_ratio = 0.2  # 测试集比例
    random_seed = 42  # 随机种子
    
    # 步骤1：整理数据集
    case_ids = organize_dataset(input_dir)
    print(f"找到 {len(case_ids)} 个完整病例")
    
    # 步骤2：划分数据集
    train_ids, test_ids = split_dataset(case_ids, test_ratio, random_seed)
    print(f"训练集：{len(train_ids)} 病例")
    print(f"测试集：{len(test_ids)} 病例")
    
    # 步骤3：创建目录结构
    create_dataset_structure(output_dir)
    
    # 步骤4：复制文件
    # 复制训练集
    copy_files(input_dir, os.path.join(output_dir, 'train'), train_ids, 'CT')
    copy_files(input_dir, os.path.join(output_dir, 'train'), train_ids, 'MR')
    
    # 复制测试集
    copy_files(input_dir, os.path.join(output_dir, 'val'), test_ids, 'CT')
    copy_files(input_dir, os.path.join(output_dir, 'val'), test_ids, 'MR')

if __name__ == "__main__":
    main()