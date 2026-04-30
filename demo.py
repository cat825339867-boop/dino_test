# 这个文件用于比较一个小图和一个文件夹中所有小图相似度

import torch
from PIL import Image
from torchvision import transforms

# 1. 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载 DINOv3 模型
# 小模型：dinov3_vits16
# 中模型：dinov3_vitb16
model = torch.hub.load(
    "facebookresearch/dinov3",
    "dinov3_vits16",
    pretrained=True
).to(device)

model.eval()

# 3. 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

# 4. 读取图片
img = Image.open("test.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

# 5. 提取向量
with torch.no_grad():
    embedding = model(x)

print("向量 shape:", embedding.shape)
print("向量前10个值:", embedding[0][:10])