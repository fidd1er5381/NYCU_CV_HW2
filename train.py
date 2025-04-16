import os
import json
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class DigitDataset(Dataset):
    """Dataset for digit detection."""
    
    def __init__(self, root_dir, json_file, transform=None, train=True):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory with images
            json_file: Path to annotation file
            transform: Image transformations
            train: Whether this is for training
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Read JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.images_info = data['images']
        if train:
            self.annotations = data['annotations']
            
            # Create image ID to annotations mapping
            self.img_to_anns = {}
            for ann in self.annotations:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.images_info)
    
    def __getitem__(self, idx):
        """
        Get item by index.
        
        Returns:
            image and target for training, or image and id for testing
        """
        img_info = self.images_info[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Read image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if not self.train:
            return image, img_id
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height] format
            # Convert to [x1, y1, x2, y2] format
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            
            boxes.append([x1, y1, x2, y2])
            # Note: In COCO format, class IDs start from 1
            labels.append(ann['category_id'])  # Digit class (class IDs 1-10 map to digits 0-9)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        return image, target


def get_transform(train=True):
    """
    Define data transformations.
    
    Args:
        train: Whether these are training transformations
        
    Returns:
        Composition of transforms
    """
    transforms_list = []
    
    # Basic transform
    transforms_list.append(transforms.ToTensor())
    
    # Data augmentation for training
    if train:
        # Random brightness, contrast, saturation and hue adjustment
        transforms_list.append(transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ))
        
        # Random rotation (small angles, as digit orientation is important)
        transforms_list.append(transforms.RandomRotation(10))
        
        # Random perspective transform (simulate different viewpoints)
        transforms_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))
        
        # Random distortion to enhance model robustness to noise
        transforms_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
    
    # Normalize images (using ImageNet mean and std)
    transforms_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transforms_list)


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, in_planes, reduction_ratio=16):
        """
        Initialize the channel attention module.
        
        Args:
            in_planes: Number of input channels
            reduction_ratio: Channel reduction ratio
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size=7):
        """
        Initialize the spatial attention module.
        
        Args:
            kernel_size: Convolution kernel size
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, gate_channels, reduction_ratio=16, kernel_size=7):
        """
        Initialize CBAM.
        
        Args:
            gate_channels: Number of input channels
            reduction_ratio: Channel reduction ratio
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(gate_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        """Forward pass."""
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


def get_model_with_cbam(num_classes):
    """
    Create Faster R-CNN model with CBAM.
    
    Args:
        num_classes: Number of classes to detect
        
    Returns:
        Modified Faster R-CNN model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    
    # Get input features dimension of the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace classifier head to adapt to our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Add CBAM before ROI head
    # Note: This requires modifying internal model structure
    model.roi_heads.cbam = CBAM(256)  # FPN output channels is 256
    
    # Save original box_roi_pool forward pass
    original_forward = model.roi_heads.box_roi_pool.forward
    
    # Define a new forward function wrapping the original one
    def forward_with_cbam(x, boxes, image_shapes):
        features = original_forward(x, boxes, image_shapes)
        return model.roi_heads.cbam(features)
    
    # Replace forward function
    model.roi_heads.box_roi_pool.forward = forward_with_cbam
    
    # Adjust anchor generator
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler=None, scaler=None):
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        lr_scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        Average training loss
    """
    model.train()
    
    running_loss = 0.0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Clear optimizer gradients
        optimizer.zero_grad()
        
        # Use mixed precision training if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal training flow
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            losses.backward()
            optimizer.step()
        
        # Update learning rate (if using scheduler)
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        running_loss += losses.item()
    
    return running_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        data_loader: Validation data loader
        device: Device to evaluate on
        
    Returns:
        mAP score
    """
    model.eval()
    
    # Parameters for mAP calculation
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Make predictions
            outputs = model(images)
            
            # Collect all predictions and targets
            all_predictions.extend(outputs)
            all_targets.extend(targets)
    
    # Calculate mAP
    from torchvision.ops import box_iou
    
    # Set parameters for mAP calculation
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # IoU thresholds
    
    # Calculate AP per class
    num_classes = 11  # Number of classes (0-9 + background)
    ap_per_class = np.zeros(num_classes)
    ap_per_class_count = np.zeros(num_classes)  # For calculating average per class
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        pred_labels = pred['labels'].cpu()
        
        target_boxes = target['boxes'].cpu()
        target_labels = target['labels'].cpu()
        
        # Skip image if no target boxes
        if len(target_boxes) == 0:
            continue
        
        # Calculate AP separately for each class
        for cls in range(1, num_classes):  # Skip background class 0
            # Get predictions and targets for current class
            cls_pred_indices = (pred_labels == cls)
            cls_target_indices = (target_labels == cls)
            
            cls_pred_boxes = pred_boxes[cls_pred_indices]
            cls_pred_scores = pred_scores[cls_pred_indices]
            cls_target_boxes = target_boxes[cls_target_indices]
            
            # Skip if no targets for this class
            if len(cls_target_boxes) == 0:
                continue
            
            # AP is 0 if no predictions for this class
            if len(cls_pred_boxes) == 0:
                continue
            
            # Calculate IoU matrix
            iou_matrix = box_iou(cls_pred_boxes, cls_target_boxes)
            
            # Sort predictions by score
            score_sorted_indices = torch.argsort(cls_pred_scores, descending=True)
            cls_pred_boxes = cls_pred_boxes[score_sorted_indices]
            cls_pred_scores = cls_pred_scores[score_sorted_indices]
            
            # Calculate AP for each IoU threshold
            ap_at_iou = []
            for iou_threshold in iou_thresholds:
                # Calculate TP and FP
                tp = torch.zeros(len(cls_pred_boxes))
                fp = torch.zeros(len(cls_pred_boxes))
                
                # Already matched target boxes
                matched_targets = set()
                
                # For each prediction
                for i, pred_box in enumerate(cls_pred_boxes):
                    # All predictions are FP if no targets
                    if len(cls_target_boxes) == 0:
                        fp[i] = 1
                        continue
                    
                    # Find target box with highest IoU with current prediction
                    if i < len(iou_matrix):
                        iou_scores = iou_matrix[score_sorted_indices[i]]
                        max_iou, max_idx = torch.max(iou_scores, dim=0)
                        
                        # If max IoU above threshold and target not matched yet
                        if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                            tp[i] = 1
                            matched_targets.add(max_idx.item())
                        else:
                            fp[i] = 1
                
                # Calculate cumulative TP and FP
                cum_tp = torch.cumsum(tp, dim=0)
                cum_fp = torch.cumsum(fp, dim=0)
                
                # Calculate precision and recall
                precision = cum_tp / (cum_tp + cum_fp)
                recall = cum_tp / len(cls_target_boxes)
                
                # Calculate AP (using 11-point interpolation)
                ap = 0
                for t in np.arange(0, 1.1, 0.1):
                    if torch.sum(recall >= t) == 0:
                        p = 0
                    else:
                        p = torch.max(precision[recall >= t])
                    ap += p / 11
                
                ap_at_iou.append(ap)
            
            # Calculate average AP across IoU thresholds
            ap_per_class[cls] += sum(ap_at_iou) / len(ap_at_iou)
            ap_per_class_count[cls] += 1
    
    # Calculate mean AP across classes
    valid_classes = ap_per_class_count > 0
    mAP = np.sum(ap_per_class[valid_classes] / ap_per_class_count[valid_classes]) / np.sum(valid_classes)
    
    print(f"Validation mAP: {mAP:.4f}")
    
    return mAP  # Return mAP, higher is better


def main():
    """Main training function."""
    # Data paths
    train_dir = "train"
    valid_dir = "valid"
    train_json = "train.json"
    valid_json = "valid.json"
    
    # Create datasets
    train_dataset = DigitDataset(
        train_dir, 
        train_json, 
        transform=get_transform(train=True), 
        train=True
    )
    valid_dataset = DigitDataset(
        valid_dir, 
        valid_json, 
        transform=get_transform(train=False), 
        train=True
    )
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    
    # Adjust batch size according to your hardware
    batch_size = 8
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x)), 
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x)), 
        num_workers=4
    )
    
    # Create model
    # 10 digit classes (class IDs 1-10 map to digits 0-9) + background class (0)
    num_classes = 11
    model = get_model_with_cbam(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0001)
    
    # Cosine annealing learning rate scheduler
    num_epochs = 20
    total_steps = len(train_loader) * num_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=1e-6
    )
    
    # Mixed precision training if GPU supports it
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Early stopping settings
    best_val_metric = 0.0  # Higher mAP is better
    patience = 5
    patience_counter = 0
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, lr_scheduler, scaler)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
        val_metric = evaluate(model, valid_loader, device)
        print(f"Validation mAP: {val_metric:.4f}")
        
        # Save model for each epoch
        torch.save(model.state_dict(), f"fasterrcnn_model_epoch{epoch+1}.pth")
        
        # Save best model
        if val_metric > best_val_metric:  # Note: Higher mAP is better
            best_val_metric = val_metric
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "fasterrcnn_model_best.pth")
            patience_counter = 0
            print(f"Saved best model, validation mAP: {val_metric:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs")
            break
    
    print(f"Training complete! Best model at epoch {best_epoch} with mAP {best_val_metric:.4f}")


if __name__ == "__main__":
    main()