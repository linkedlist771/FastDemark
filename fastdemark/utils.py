from torch import optim
import torch
from torch.cpu import is_available
from tqdm import tqdm
# from tqdm.auto import tqdm
# from helper import *
from fastdemark.model import SkipEncoderDecoder, input_noise
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

def pil_to_np_array(pil_image):
    ar = np.array(pil_image)
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.

def np_to_torch_array(np_array):
    return torch.from_numpy(np_array)[None, :]

def torch_to_np_array(torch_array):
    return torch_array.detach().cpu().numpy()[0]

def read_image(path, image_size = -1):
    pil_image = Image.open(path)
    return pil_image

def crop_image(image, crop_factor = 64):
    shape = (image.size[0] - image.size[0] % crop_factor, image.size[1] - image.size[1] % crop_factor)
    bbox = [int((image.shape[0] - shape[0])/2), int((image.shape[1] - shape[1])/2), int((image.shape[0] + shape[0])/2), int((image.shape[1] + shape[1])/2)]
    return image.crop(bbox)

def get_image_grid(images, nrow = 3):
    torch_images = [torch.from_numpy(x) for x in images]
    grid = make_grid(torch_images, nrow)
    return grid.numpy()
    
def visualize_sample(*images_np, nrow = 3, size_factor = 10):
    c = max(x.shape[0] for x in images_np)
    images_np = [x if (x.shape[0] == c) else np.concatenate([x, x, x], axis = 0) for x in images_np]
    grid = get_image_grid(images_np, nrow)
    plt.figure(figsize = (len(images_np) + size_factor, 12 + size_factor))
    plt.axis('off')
    plt.imshow(grid.transpose(1, 2, 0))
    plt.show()

def max_dimension_resize(image_pil, mask_pil, max_dim):
    w, h = image_pil.size
    aspect_ratio = w / h
    if w > max_dim:
        h = int((h / w) * max_dim)
        w = max_dim
    elif h > max_dim:
        w = int((w / h) * max_dim)
        h = max_dim
    return image_pil.resize((w, h)), mask_pil.resize((w, h))

def preprocess_images(image_path, mask_path, max_dim):
    image_pil = read_image(image_path).convert('RGB')
    mask_pil = read_image(mask_path).convert('RGB')

    image_pil, mask_pil = max_dimension_resize(image_pil, mask_pil, max_dim)

    image_np = pil_to_np_array(image_pil)
    mask_np = pil_to_np_array(mask_pil)

    print('Visualizing mask overlap...')

    visualize_sample(image_np, mask_np, image_np * mask_np, nrow = 3, size_factor = 10)

    return image_np, mask_np

def get_available_device() -> str:
    device = 'cpu'

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device

def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps, tqdm_length=100):
    DTYPE = torch.FloatTensor
    device = get_available_device()

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down = [128] * 5,
        num_channels_up = [128] * 5,
        num_channels_skip = [128] * 5
    ).type(DTYPE).to(device)

    objective = torch.nn.MSELoss().type(DTYPE).to(device)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    print('\nStarting training...\n')

    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_saved

        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        if step % show_step == 0:
            output_image = torch_to_np_array(output)
            visualize_sample(image_np, output_image, nrow = 2, size_factor = 10)

        progress_bar.set_postfix(Loss = loss.item())

        optimizer.step()

    output_image = torch_to_np_array(output)
    visualize_sample(output_image, nrow = 1, size_factor = 10)

    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))

    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')

    pil_image.save(output_path)

def create_dynamic_mask(image_path, watermark_region='bottom_right', watermark_ratio=0.25):
    """
    动态创建水印掩码图像
    
    Args:
        image_path: 原始图像路径
        watermark_region: 水印位置 ('bottom_right', 'bottom_left', 'top_right', 'top_left')
        watermark_ratio: 水印区域占图像的比例
    
    Returns:
        mask_path: 创建的掩码文件路径
    """
    # 读取原图获取尺寸
    original_img = Image.open(image_path)
    width, height = original_img.size
    
    print(f"输入图像尺寸: {width} x {height}")
    
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
    
    # 生成掩码文件路径
    mask_path = image_path.rsplit('.', 1)[0] + '_mask.png'
    
    # 保存掩码
    mask.save(mask_path)
    
    print(f"动态创建掩码: {mask_path}")
    print(f"水印区域: {watermark_region}, 坐标: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"水印区域大小: {watermark_width} x {watermark_height}")
    
    return mask_path

def run_complete_watermark_removal(image_path, watermark_region='bottom_right', watermark_ratio=0.25, 
                                 max_dim=512, training_steps=2000, lr=0.01):
    """
    完整的去水印流程：检查图片->创建mask->运行模型
    
    Args:
        image_path: 输入图像路径
        watermark_region: 水印位置
        watermark_ratio: 水印区域比例
        max_dim: 图像最大维度
        training_steps: 训练步数
        lr: 学习率
    """
    print("=" * 60)
    print("FastDemark 完整去水印流程")
    print("=" * 60)
    
    # 1. 检查输入图片是否存在
    import os
    if not os.path.exists(image_path):
        print(f"错误：找不到输入图像 {image_path}")
        return
    
    # 2. 动态检查图片大小并创建mask
    print("步骤 1: 分析输入图片...")
    mask_path = create_dynamic_mask(image_path, watermark_region, watermark_ratio)
    
    # 3. 设置模型参数
    print("\n步骤 2: 设置模型参数...")
    reg_noise = 0.03
    input_depth = 32
    show_step = min(200, training_steps // 100)  # 动态调整显示间隔
    
    print(f"最大维度: {max_dim}")
    print(f"训练步数: {training_steps}")
    print(f"学习率: {lr}")
    print(f"显示间隔: 每 {show_step} 步")
    
    # 4. 运行去水印模型
    print("\n步骤 3: 开始去水印处理...")
    print("=" * 60)
    
    try:
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
        print("✅ 去水印处理完成！")
        output_filename = image_path.split('/')[-1].split('.')[0] + '-output.jpg'
        print(f"📁 输出图像: {output_filename}")
        print(f"📁 掩码图像: {mask_path}")
        print("=" * 60)
        
        # 清理临时掩码文件
        if os.path.exists(mask_path):
            os.remove(mask_path)
            print(f"🗑️  已清理临时掩码文件: {mask_path}")
        
    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    import os
    
    # 默认参数
    default_image = "resources/water_input.png"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 尝试使用默认图片
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(current_dir, default_image)
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"使用方法: python -m fastdemark.utils [图片路径]")
        print(f"或者将图片放到: {default_image}")
        sys.exit(1)
    
    # 运行完整的去水印流程
    run_complete_watermark_removal(
        image_path=image_path,
        watermark_region='bottom_right',  # 右下角水印
        watermark_ratio=0.25,            # 水印占25%
        max_dim=512,                     # 最大维度512像素
        training_steps=2000,             # 训练2000步
        lr=0.01                          # 学习率0.01
    )