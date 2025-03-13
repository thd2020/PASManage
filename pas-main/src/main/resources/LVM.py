import argparse
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
 
from PIL import Image

class LVMClassifier(torch.nn.Module):
    def __init__(self,
                 num_classes = 3):
        super().__init__()
        self.img_encoder = torchvision.models.resnet50(pretrained=True)
        self.img_encoder.fc = nn.Sequential(
            nn.Linear(self.img_encoder.fc.in_features, 256),
            nn.ReLU())
        self.img_norm = nn.LayerNorm(256)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        x = self.img_norm(x)
        result = self.fc(x)
        # return x, y
        return result


if __name__ == '__main__':
    types = ['normal', 'accreta', 'increta']
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_path', type=str, default='/media/lmj/4326a9f8-e9c8-4a59-9cd1-6f16d91ab1f5/hwx/rerun_PAS/result/123/LVM/frozen/best.pth', help='model checkpoint path')
    parser.add_argument('-device', type=str, default='cuda:0', help='device')
    parser.add_argument('-img_path', type=str, default='/home/lmj/xyx/sda2/data_demo/分割图片/植入/1656334_漆旭/1.2.410.200001.1.11801.787041058.3.20200722.1151215142.677.9 - 副本.jpg', help='image path')
    args = parser.parse_args()
    
    model = LVMClassifier()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 3))
    pretrained = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(pretrained["state_dict"], strict=True)
    model = model.to(args.device)
    # model.eval()
    preprocessor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(args.img_path).convert('RGB')
    image_tensor = preprocessor(image).unsqueeze(0).to(args.device)
    digits = model(image_tensor)
    probabilities = torch.softmax(digits, 1).squeeze(0).cpu().detach().tolist()
    type = torch.argmax(torch.softmax(digits, 1)).squeeze(0).cpu().detach().item()
    print(f"probabilities: {types[0]} {probabilities[0]:.4f}, {types[1]}: {probabilities[1]:.4f}, {types[2]}: {probabilities[2]:.4f}")
    print(f"predict type: {types[type]}")