import os
import json
import shutil

DATA_ROOT = "./results"
JSON_DIR = "./data_splits" # 你的 json 所在目录
# 我们刚刚狙击出的 4 个天选 ID
TARGET_IDS = ['Color1_000534', 'Infrared3_000083', 'Color1_000627', 'Infrared3_000552']

def harvest():
    os.makedirs(os.path.join(DATA_ROOT, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_ROOT, "masks_gt"), exist_ok=True)
    
    found = 0
    for fold in range(5):
        json_path = os.path.join(JSON_DIR, f"fold_{fold}.json")
        if not os.path.exists(json_path): continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            for split in ['train', 'val']:
                for item in data[split]:
                    if item['id'] in TARGET_IDS:
                        # 拷贝原图
                        shutil.copy(item['image'], os.path.join(DATA_ROOT, "images", f"{item['id']}.png"))
                        
                        # 拷贝 GT (自动替换为 Cleaned_Label)
                        clean_label = item['label'].replace("/Label/", "/Cleaned_Label/")
                        if os.path.exists(clean_label):
                            shutil.copy(clean_label, os.path.join(DATA_ROOT, "masks_gt", f"{item['id']}.png"))
                        
                        print(f"✅ 已成功收集素材: {item['id']}")
                        found += 1
                        
    print(f"\n🎉 收集完成！共找到 {found}/4 张图片的素材。现在可以去跑画图脚本了！")

if __name__ == "__main__":
    harvest()