#!/usr/bin/env python3
"""
创建水印掩码图像
"""
from PIL import Image, ImageDraw
import numpy as np

def create_watermark_mask(original_image_path, mask_output_path, watermark_region='bottom_right', watermark_ratio=0.3):
    """
    创建水印掩码图像
    
    Args:
        original_image_path: 原始图像路径
        mask_output_path: 掩码输出路径
        watermark_region: 水印位置 ('bottom_right', 'bottom_left', 'top_right', 'top_left')
        watermark_ratio: 水印区域占图像的比例
    """
    # 读取原图获取尺寸
    original_img = Image.open(original_image_path)
    width, height = original_img.size
    
    # 创建白色背景的掩码图像
    mask = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(mask)
    
    # 计算水印区域大小
    watermark_width = int(width * watermark_ratio)
    watermark_height = int(height * watermark_ratio)
    
    # 根据水印位置计算坐标
    if watermark_region == 'bottom_right':
        x1 = width - watermark_width
        y1 = height - watermark_height
        x2 = width
        y2 = height
    elif watermark_region == 'bottom_left':
        x1 = 0
        y1 = height - watermark_height
        x2 = watermark_width
        y2 = height
    elif watermark_region == 'top_right':
        x1 = width - watermark_width
        y1 = 0
        x2 = width
        y2 = watermark_height
    elif watermark_region == 'top_left':
        x1 = 0
        y1 = 0
        x2 = watermark_width
        y2 = watermark_height
    else:
        raise ValueError("watermark_region must be one of: 'bottom_right', 'bottom_left', 'top_right', 'top_left'")
    
    # 在水印区域绘制黑色矩形（表示需要去除的区域）
    draw.rectangle([x1, y1, x2, y2], fill='black')
    
    # 保存掩码
    mask.save(mask_output_path)
    print(f"水印掩码已保存到: {mask_output_path}")
    print(f"水印区域: {watermark_region}, 坐标: ({x1}, {y1}) to ({x2}, {y2})")
    
    return mask

if __name__ == "__main__":
    # 创建右下角水印掩码
    create_watermark_mask(
        original_image_path='resources/water_input.png',
        mask_output_path='resources/watermark_mask.png',
        watermark_region='bottom_right',
        watermark_ratio=0.25  # 水印区域占图片25%
    )
