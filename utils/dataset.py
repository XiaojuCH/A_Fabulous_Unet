import os
from PIL import Image
from torch.utils.data import Dataset

class TearMeniscusDataset(Dataset):
    def __init__(self, data_root, joint_transform=None, transform=None, target_transform=None):
        """
        适配目录结构:
        root/
          ├── Colour1/
          │     ├── Original/ (图片)
          │     └── Label/    (标签)
          ├── Colour2/
          ├── Infrared1/
          ...
        """
        self.data_root = data_root
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        
        self.images_path = []
        self.masks_path = []
        
        # 支持的图片扩展名 (Linux下区分大小写，所以要写全)
        valid_exts = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.bmp']

        # 1. 遍历根目录下的所有子文件夹 (Colour1, Colour2, Infrared1...)
        if not os.path.exists(data_root):
            raise ValueError(f"Data root not found: {data_root}")

        subfolders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        
        print(f"Scanning subfolders: {subfolders}")

        for sub in subfolders:
            sub_path = os.path.join(data_root, sub)
            img_dir = os.path.join(sub_path, 'Original')
            mask_dir = os.path.join(sub_path, 'Label')

            # 如果某个子文件夹里没有Original或Label，跳过
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                print(f"Skipping {sub}: 'Original' or 'Label' folder missing.")
                continue

            # 2. 遍历 Original 里的文件
            files = sorted(os.listdir(img_dir))
            for filename in files:
                # 检查是否是图片
                if not any(filename.endswith(ext) for ext in valid_exts):
                    continue

                img_filepath = os.path.join(img_dir, filename)
                
                # 3. 寻找对应的 Label
                # 策略A: 假设文件名完全一致 (Color1_000000.PNG -> Color1_000000.PNG)
                mask_filepath = os.path.join(mask_dir, filename)
                
                if os.path.exists(mask_filepath):
                    self.images_path.append(img_filepath)
                    self.masks_path.append(mask_filepath)
                else:
                    # 策略B: 扩展名可能不一样 (例如图是.jpg, 标签是.png)
                    name_no_ext = os.path.splitext(filename)[0]
                    found = False
                    for ext in valid_exts:
                        potential_mask = os.path.join(mask_dir, name_no_ext + ext)
                        if os.path.exists(potential_mask):
                            self.images_path.append(img_filepath)
                            self.masks_path.append(potential_mask)
                            found = True
                            break
                    
                    if not found:
                        # print(f"Warning: Label not found for {filename} in {sub}")
                        pass

        print(f"✅ Loaded {len(self.images_path)} images from {len(subfolders)} subfolders.")
        if len(self.images_path) == 0:
            raise ValueError("No images found! Please check your data_root path.")

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        try:
            # 读取图片
            image = Image.open(img_path).convert('RGB')
            # 读取标签 (一定要转成灰度 L，0是背景，255是前景)
            mask = Image.open(mask_path).convert('L')

            # 同步增强
            if self.joint_transform:
                image, mask = self.joint_transform(image, mask)

            # 独立变换
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask
        except Exception as e:
            print(f"Error loading file: {img_path}")
            raise e