"""
对数据采取 N4偏置场校正（N4BiasFieldCorrection）

请配置好文件夹等

参考时间，180个病例需要大于4小时
"""


import os
import SimpleITK as sitk
from tqdm import tqdm

# ===== ===== 配置 ===== ===== 
data_roots = [
    '/home/featurize/data/Task1/brain',
    '/home/featurize/data/Task1/pelvis',
]

output_roots = [
    '/home/featurize/data/Task1_N4BFC/brain',
    '/home/featurize/data/Task1_N4BFC/pelvis',
]

ignore_ids = ['overview'] # 忽略的病例ID列表 # 默认忽略其中的overview文件夹
# ===== =====  配置 ===== ===== 

def perform_n4_correction(input_image, mask_image=None):
    """
    执行N4偏置场校正
    """
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    
    # 设置N4校正参数
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    
    if mask_image:
        corrected_image = corrector.Execute(input_image, mask_image)
    else:
        corrected_image = corrector.Execute(input_image)
    
    return corrected_image

def process_folder(data_root, output_root):
    """
    处理单个文件夹中的所有扫描
    """
    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)
    
    # 获取所有病例文件夹
    case_folders = [f for f in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, f)) and f not in ignore_ids]
    
    for case in tqdm(case_folders, desc=f"Processing {os.path.basename(data_root)}"):
        case_path = os.path.join(data_root, case)
        output_case_path = os.path.join(output_root, case)
        os.makedirs(output_case_path, exist_ok=True)
        
        # 处理CT图像
        ct_path = os.path.join(case_path, 'ct.nii.gz')
        if os.path.exists(ct_path):
            try:
                ct_image = sitk.ReadImage(ct_path)
                mask_image = sitk.ReadImage(os.path.join(case_path, 'mask.nii.gz'))
                
                # 执行N4校正
                corrected_ct = perform_n4_correction(ct_image, mask_image)
                
                # 保存结果
                sitk.WriteImage(corrected_ct, os.path.join(output_case_path, 'ct.nii.gz'))
            except Exception as e:
                print(f"Error processing CT for case {case}: {str(e)}")
        
        # 处理MR图像
        mr_path = os.path.join(case_path, 'mr.nii.gz')
        if os.path.exists(mr_path):
            try:
                mr_image = sitk.ReadImage(mr_path)
                mask_image = sitk.ReadImage(os.path.join(case_path, 'mask.nii.gz'))
                
                # 执行N4校正
                corrected_mr = perform_n4_correction(mr_image, mask_image)
                
                # 保存结果
                sitk.WriteImage(corrected_mr, os.path.join(output_case_path, 'mr.nii.gz'))
            except Exception as e:
                print(f"Error processing MR for case {case}: {str(e)}")
        
        # 复制mask文件
        mask_path = os.path.join(case_path, 'mask.nii.gz')
        if os.path.exists(mask_path):
            try:
                mask_image = sitk.ReadImage(mask_path)
                sitk.WriteImage(mask_image, os.path.join(output_case_path, 'mask.nii.gz'))
            except Exception as e:
                print(f"Error copying mask for case {case}: {str(e)}")

def main():
    for data_root, output_root in zip(data_roots, output_roots):
        print(f"Processing data from {data_root}")
        print(f"Saving results to {output_root}")
        process_folder(data_root, output_root)

if __name__ == "__main__":
    main()