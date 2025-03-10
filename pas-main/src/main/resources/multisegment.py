import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import sys
import cfg
import torch.nn.functional as F
import torchvision.transforms as transforms

# Add the ssam directory to Python path
ssam_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'ssam')
sys.path.append(ssam_path)

parser = argparse.ArgumentParser(
    description=(
        "Run multi-target segmentation on a single image using SSAM model. "
        "The model requires target classes and prompts for each target."
    )
)

def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """ return given network
    """

    if net == 'sam':
        from models.sam import SamPredictor, sam_model_registry
        from models.sam.utils.transforms import ResizeLongestSide

        net = sam_model_registry['vit_b'](args, checkpoint=args.model_path).to(args.gpu_device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        # net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net, device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net

def load_model(args):
    # Initialize model similar to val.py
    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=True, gpu_device=GPUdevice, distribution=args.distributed)
    checkpoint = torch.load(args.model_path)
    filtered_state_dict = {}
    state_dict = checkpoint['state_dict']
    # for k, v in state_dict.items():
    #     if k in net.state_dict():
    #         if v.shape == net.state_dict()[k].shape:
    #             filtered_state_dict[k] = v
    # filtered_state_dict = state_dict
    net.load_state_dict(state_dict)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    return net

def preprocess_image(image_path, image_size):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transforms.Resize((image_size, image_size))(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    unique, counts = torch.unique(image, return_counts=True)
    return image

def process_prompts(args):
    import json
    prompts = json.loads(args.prompts)
    if args.prompt_type == 'point':
        # Convert format from {"target_name": [[x1,y1], [x2,y2],...], ...} 
        # to tensor of shape (B, N, 2) and list of class names
        all_points = []
        all_labels = []
        for class_name, points in prompts.items():
            # Convert string representation of points to actual list of points
            if isinstance(points, str):
                points = json.loads(points)
            # Add points as numerical coordinates
            all_points.extend([list(map(float, point)) for point in points])
            all_labels.extend([class_name] * len(points))
        # Convert to tensor format (B, N, 2)
        point_coords = torch.tensor([all_points], dtype=torch.float32)  # Add batch dimension
        point_labels = [all_labels]  # List of lists for batch format
        return point_coords, point_labels
    elif args.prompt_type == 'box':
        # Expect format: {"target_name": [x1,y1,x2,y2], ...}
        return {k: torch.tensor(v) for k, v in prompts.items()}
    else: # mask
        # Expect format: {"target_name": "mask_path", ...}
        return {k: torch.from_numpy(np.load(v)) for k, v in prompts.items()}

def main(args):
    # Set up device
    device = torch.device(f'cuda:{args.gpu_device}')
    
    # Load model
    net = load_model(args)
    net = net.to(device)
    
    # Preprocess image
    imgs = preprocess_image(args.img_path, args.image_size)
    imgs = imgs.to(device)
    
    # Process prompts
    point_coords, point_labels = process_prompts(args)
    point_coords = point_coords.to(device)
    
    # Run inference
    with torch.no_grad():
        imge = net.image_encoder(imgs)
        se, de, te = net.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
            texts=[args.targets]
        )
        pred = net.mask_decoder(
            image_embeddings=imge,
            text_embeddings=te,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de
        )

        
        unique, counts = torch.unique(pred, return_counts=True)

        # Convert predicted logits to class indices
        pred_masks = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    
    # Save masks for each target
    os.makedirs(args.work_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    
    # Save individual class masks
    num_classes = len(args.targets)
    unique, counts = torch.unique(pred_masks, return_counts=True)
    masks = F.one_hot(pred_masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
    unique, counts = torch.unique(masks, return_counts=True)
    
    for idx, target in enumerate(args.targets):
        if idx == 0:  # Skip background class
            continue
        mask = masks[0, idx].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        save_path = os.path.join(args.work_dir, f"{base_name}_{target}_mask.jpg")
        cv2.imwrite(save_path, mask)

if __name__ == '__main__':
    args = cfg.parse_args()
    args.targets.insert(0, 'background')
    main(args)
