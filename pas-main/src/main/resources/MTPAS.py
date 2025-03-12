import torch
import torch.nn as nn
from image_encoder_src.image_encoder import ImageEncoderViT
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MTPASClassifier(torch.nn.Module):
    def __init__(self,
                 img_size: 1024,
                 num_classes: 3):
        super().__init__()
        self.img_encoder = ImageEncoderViT(
            img_size=img_size,
            use_abs_pos=False
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_norm = nn.LayerNorm(256)

        self.clinical_encoder = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU())

        self.clinical_norm = nn.LayerNorm(256)

        self.fusion_norm = nn.LayerNorm(512)

        self.fusion_encoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu"
        )
        self.fusion_encoder = nn.TransformerEncoder(self.fusion_encoderLayer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes))

    def forward(self, data):
        x, y = data[0], data[1]

        x = self.img_encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.img_norm(x)

        y = y.squeeze(1)
        y = self.clinical_encoder(y)
        y = self.clinical_norm(y)

        fusion = torch.cat((x, y), dim=1)
        fusion = self.fusion_norm(fusion)
        fusion = self.fusion_encoder(fusion)

        result = self.fc(fusion)

        # return x, y
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, required=True)
    parser.add_argument('-age', type=int, required=True)
    parser.add_argument('-placenta_previa', type=int, required=True)
    parser.add_argument('-c_section_count', type=int, required=True)
    parser.add_argument('-had_abortion', type=int, required=True)
    return parser.parse_args()

def load_model(img_size=224, num_classes=3, ckpt_path="pas-main/src/main/resources/mtpas.pth"):
    model = MTPASClassifier(img_size, num_classes)
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
    return torch.tensor([[age_processed, pp, cs, abort]]).to(device)

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
        output = model(image, clinical_data)
        probs = torch.softmax(output, dim=1).numpy()
        pred_class = np.argmax(probs)
        
    # Print results in expected format
    print(f"probabilities: normal: {probs[0][0]:.4f}, accreta: {probs[0][1]:.4f}, increta: {probs[0][2]:.4f}")
    print(f"predict type: {['normal', 'accreta', 'increta'][pred_class]}")

if __name__ == '__main__':
    main()
