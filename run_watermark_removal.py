#!/usr/bin/env python3
"""
运行去水印模型
"""
import os
import sys
from fastdemark.utils import remove_watermark

def main():
    # 设置参数
    image_path = "resources/water_input.png"
    mask_path = "resources/watermark_mask.png"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图像 {image_path}")
        return
    
    if not os.path.exists(mask_path):
        print(f"错误：找不到掩码图像 {mask_path}")
        return
    
    # 模型参数
    max_dim = 512          # 图像最大维度，较小值训练更快
    reg_noise = 0.03       # 正则化噪声
    input_depth = 32       # 输入噪声深度
    lr = 0.01              # 学习率
    show_step = 200        # 每200步显示一次结果
    training_steps = 2000  # 总训练步数
    
    print("=" * 60)
    print("FastDemark 去水印模型")
    print("=" * 60)
    print(f"输入图像: {image_path}")
    print(f"掩码图像: {mask_path}")
    print(f"最大维度: {max_dim}")
    print(f"训练步数: {training_steps}")
    print(f"学习率: {lr}")
    print("=" * 60)
    
    try:
        # 运行去水印模型
        remove_watermark(
            image_path=image_path,
            mask_path=mask_path,
            max_dim=max_dim,
            reg_noise=reg_noise,
            input_depth=input_depth,
            lr=lr,
            show_step=show_step,
            training_steps=training_steps
        )
        
        print("\n" + "=" * 60)
        print("去水印处理完成！")
        print("输出图像已保存为: water_input-output.jpg")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
