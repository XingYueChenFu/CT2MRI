import os
import glob
import SimpleITK as sitk
import numpy as np
import cv2

def apply_window_level(image, window_center, window_width):
    """应用窗宽窗位调整到SimpleITK图像"""
    image = sitk.Cast(image, sitk.sitkFloat32)
    min_val = window_center - window_width/2
    max_val = window_center + window_width/2
    
    window_filter = sitk.IntensityWindowingImageFilter()
    window_filter.SetWindowMinimum(min_val)
    window_filter.SetWindowMaximum(max_val)
    window_filter.SetOutputMinimum(0)
    window_filter.SetOutputMaximum(255)
    
    return window_filter.Execute(image)

def adjust_contrast_volume(np_volume, target_contrast):
    """调整整个volume的对比度"""
    adjusted_volume = cv2.convertScaleAbs(
        np_volume,
        alpha=target_contrast/(np_volume.mean() + 1e-8),
        beta=0
    )
    return adjusted_volume

def resize_volume(np_volume, target_h, target_w):
    """统一调整整个volume的空间尺寸（三维操作）"""
    current_z, current_h, current_w = np_volume.shape
    
    # 高度方向调整
    if current_h > target_h:
        start_h = (current_h - target_h) // 2
        resized = np_volume[:, start_h:start_h+target_h, :]
    else:
        pad_h = target_h - current_h
        resized = np.pad(np_volume, 
                        ((0, 0), (pad_h//2, pad_h - pad_h//2), (0, 0)),
                        mode='constant')
    
    # 宽度方向调整
    current_w_new = resized.shape[2]
    if current_w_new > target_w:
        start_w = (current_w_new - target_w) // 2
        resized = resized[:, :, start_w:start_w+target_w]
    else:
        pad_w = target_w - current_w_new
        resized = np.pad(resized,
                        ((0, 0), (0, 0), (pad_w//2, pad_w - pad_w//2)),
                        mode='constant')
    
    return resized

def process_ct_images(root_dir, output_dir, window_center, window_width, 
                     target_contrast, target_height, target_width):
    """
    处理所有CT图像的主函数
    :param target_height: 目标高度（每个slice的Y轴）
    :param target_width: 目标宽度（每个slice的X轴）
    """
    BENCH_SLICE = 100
    BENCH_D = 213

    os.makedirs(output_dir, exist_ok=True)
    
    file_pattern = os.path.join(root_dir, '**', 'ct.nii.gz')
    for input_path in glob.glob(file_pattern, recursive=True):
        try:
            parent_folder = os.path.basename(os.path.dirname(input_path))
            output_filename = f"{parent_folder}.npy"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"Skipping existing: {output_path}")
                continue

            # 读取并预处理数据
            img = sitk.ReadImage(input_path)
            
            windowed = apply_window_level(img, window_center, window_width)
            np_volume = sitk.GetArrayFromImage(windowed)  # 维度：(Z,Y,X)
            
            # Z轴切片选择（保持原始depth处理逻辑）
            slice_num = np_volume.shape[0]
            start_slice = BENCH_SLICE + (slice_num - BENCH_D)
            np_volume = np_volume[start_slice:start_slice+64]

            # 调整对比度
            adjusted = adjust_contrast_volume(np_volume, target_contrast)

            # 统一调整空间尺寸（三维操作）
            resized_volume = resize_volume(adjusted, target_height, target_width)
            

            # 保存结果（维度顺序保持Z,Y,X）
            np.save(output_path, resized_volume)
            print(f"Processed: {input_path} -> {output_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")

def process_mr_images(root_dir, output_dir, target_contrast, target_height, target_width):
    """
    处理所有MR图像的主函数
    :param root_dir: 包含MR图像的根目录
    :param output_dir: 处理后的输出目录
    :param target_contrast: 目标对比度
    :param target_height: 目标高度
    :param target_width: 目标宽度
    """
    BENCH_SLICE = 100
    BENCH_D = 213

    os.makedirs(output_dir, exist_ok=True)
    
    file_pattern = os.path.join(root_dir, '**', 'mr.nii.gz')
    for input_path in glob.glob(file_pattern, recursive=True):
        try:
            parent_folder = os.path.basename(os.path.dirname(input_path))
            output_filename = f"{parent_folder}.npy"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"Skipping existing: {output_path}")
                continue

            # 处理流程
            img = sitk.ReadImage(input_path)
            np_volume = sitk.GetArrayFromImage(img)  # (Z,Y,X)

            # Z轴切片选择
            slice_num = np_volume.shape[0]
            start_slice = BENCH_SLICE + (slice_num - BENCH_D)
            np_volume = np_volume[start_slice:start_slice+64]

            # 线性归一化到0-255
            np_volume = (np_volume - np_volume.min()) / (np_volume.max() - np_volume.min()) * 255

            # 调整对比度
            adjusted = adjust_contrast_volume(np_volume, target_contrast)

            # 调整空间尺寸
            resized_volume = resize_volume(adjusted, target_height, target_width)

            # 归一化到(-1,1)范围
            final_volume = (resized_volume - 127.5) / 127.5

            # 保存结果
            np.save(output_path, final_volume)
            print(f"Processed: {input_path} -> {output_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            
            
if __name__ == "__main__":
    input_dir = "/home/featurize/data/nii/"
    output_ct_dir = "/home/featurize/data/Task1_Liu/CT"
    output_mri_dir = "/home/featurize/data/Task1_Liu/MR"
    
    # 目标尺寸参数（根据实际需求修改）
    TARGET_HEIGHT = 256  # 统一调整后的高度（Y轴）
    TARGET_WIDTH = 256   # 统一调整后的宽度（X轴）
    WINDOW_CENTER = 32   # CT窗位
    WINDOW_WIDTH = 80    # CT窗宽
    TARGET_CONTRAST = 30 # 对比度调整参数
    
    process_ct_images(
        root_dir=input_dir,
        output_dir=output_ct_dir,
        window_center=WINDOW_CENTER,
        window_width=WINDOW_WIDTH,
        target_contrast=TARGET_CONTRAST,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH
    )
    
    process_mr_images(
        root_dir=input_dir,
        output_dir=output_mri_dir,
        target_contrast=TARGET_CONTRAST,
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH
    )
    