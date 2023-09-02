import torch

import ultralytics

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 默认下载的是cpu版本的pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    #检查是否有GPU可用

print(device)

ultralytics.checks()