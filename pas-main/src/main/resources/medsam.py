import torch
import torch.nn as nn
from image_encoder_src.image_encoder import ImageEncoderViT

class MedSAMClassifier(torch.nn.Module):
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

        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.img_norm(x)

        result = self.fc(x)

        return result


if __name__ == '__main__':
    model = MedSAMClassifier(224, 3)
    weight = torch.load("/mnt/sda1/hwx/rerun_PAS/result/123/medsam/frozen/last.pth")["state_dict"]
    model.load_state_dict(weight, strict=True)
    model = model.to("cuda:0")
    model.eval()

    dummy_img = torch.randn(5, 3, 224, 224).to("cuda:0")
    dummy_clinical = torch.randn(5, 3).to("cuda:0")
    # output_x, output_y = model((dummy_img, dummy_clinical))
    # print(output_x.shape, output_y.shape)
    # print(torch.cat((output_x, output_y), dim=1).shape)
    output = model((dummy_img, dummy_clinical))
    print(output)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, torch.tensor([0, 0, 1, 2, 1]).to("cuda:0"))
    print(loss)
