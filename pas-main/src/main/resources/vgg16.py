import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VggMulti(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # 加载vgg16并移除分类头
        vgg = models.vgg16(pretrained=True)
        vgg = nn.Sequential(*list(vgg.children())[:-1])
        self.img_encoder = nn.Sequential(
            vgg,
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
        )
        self.img_norm = nn.LayerNorm(256)

        self.clinical_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.clinical_norm = nn.LayerNorm(256)

        self.fusion_norm = nn.LayerNorm(512)
        self.fusion_encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, data):
        x, y = data[0], data[1]

        x = self.img_encoder(x)
        x = self.img_norm(x)

        y = self.clinical_encoder(y)
        y = self.clinical_norm(y)

        fusion = torch.cat((x, y), dim=1)
        fusion = self.fusion_norm(fusion)
        fusion = self.fusion_encoder(fusion)

        result = self.fc(fusion)

        return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, required=True)
    parser.add_argument('-age', type=int, required=True)
    parser.add_argument('-placenta_previa', type=int, required=True)
    parser.add_argument('-c_section_count', type=int, required=True)
    parser.add_argument('-had_abortion', type=int, required=True)
    return parser.parse_args()

def load_model(num_classes=3, ckpt_path="/home/lmj/xyx/PASManage/pas-main/src/main/resources/vgg.pth"):
    model = VggMulti(num_classes)
    weight = torch.load(ckpt_path)['state_dict']
    model.load_state_dict(weight, strict=True)
    model = model.to(device)
    model.eval()
    return model

def load_and_preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)
def process_clinical_data(age, pp, cs, abort):
    age_processed = 1 if age >= 35 else 0
    return torch.tensor([[age_processed, pp, cs, abort]]).float().to(device)

def main():
    args = parse_args()
    model = load_model()
    
    # Process image
    image = load_and_preprocess_image(args.img_path)
    
    # Process clinical data 
    clinical_data = process_clinical_data(
        args.age,
        args.placenta_previa,
        args.c_section_count,
        args.had_abortion
    )

    # Get prediction
    with torch.no_grad():
        output = model([image, clinical_data])
        probs = torch.softmax(output, dim=1).cpu().numpy()
        pred_class = np.argmax(probs)
        
    # Print results in expected format
    print(f"probabilities: normal: {probs[0][0]:.4f}, accreta: {probs[0][1]:.4f}, increta: {probs[0][2]:.4f}")
    print(f"predict type: {['normal', 'accreta', 'increta'][pred_class]}")

if __name__ == "__main__":
    main()
