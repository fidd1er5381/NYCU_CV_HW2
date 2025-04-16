import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import DBSCAN
from torchvision.ops import box_iou
from train_model import get_model, DigitDataset, get_transform


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TestDigitDataset(Dataset):
    """Dataset class for test images."""
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize the TestDigitDataset.
        
        Args:
            root_dir: Directory containing test images
            transform: Optional transform to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(os.listdir(root_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img_id = int(img_name.split('.')[0])  # Extract ID from filename
        
        # Read image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_id


def calculate_map(predictions, ground_truth, iou_threshold=0.5):
    """
    Calculate mAP (mean Average Precision).
    
    Args:
        predictions: List of model predictions, each containing 
                    'image_id', 'category_id', 'bbox', 'score' keys
        ground_truth: List of annotations, each containing 
                     'image_id', 'category_id', 'bbox' keys
        iou_threshold: IoU threshold, default is 0.5
        
    Returns:
        mAP: Mean Average Precision
    """
    if not predictions or not ground_truth:
        print("Warning: Predictions or ground truth is empty")
        return 0.0
    
    # Group GT and predictions by image_id and category_id
    gt_by_img_cls = {}
    for gt in ground_truth:
        img_id = gt['image_id']
        cls_id = gt['category_id']
        key = (img_id, cls_id)
        if key not in gt_by_img_cls:
            gt_by_img_cls[key] = []
        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        bbox = gt['bbox']
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        gt_by_img_cls[key].append([x1, y1, x2, y2])
    
    # Group predictions by image_id and category_id, sort by score
    pred_by_img_cls = {}
    for pred in predictions:
        img_id = pred['image_id']
        cls_id = pred['category_id']
        score = pred['score']
        key = (img_id, cls_id)
        if key not in pred_by_img_cls:
            pred_by_img_cls[key] = []
        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        bbox = pred['bbox']
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        pred_by_img_cls[key].append((score, [x1, y1, x2, y2]))
    
    # Calculate AP for each class
    aps = []
    
    # Get all class IDs
    all_cls_ids = set(
        [k[1] for k in gt_by_img_cls.keys()] + 
        [k[1] for k in pred_by_img_cls.keys()]
    )
    
    for cls_id in all_cls_ids:
        # Collect all image IDs containing this class
        img_ids = set(
            [k[0] for k in gt_by_img_cls.keys() if k[1] == cls_id] + 
            [k[0] for k in pred_by_img_cls.keys() if k[1] == cls_id]
        )
        
        # Collect all predictions and ground truths for this class
        all_gt = []
        all_pred_scores = []
        all_pred_boxes = []
        
        for img_id in img_ids:
            key = (img_id, cls_id)
            
            # Collect ground truth
            if key in gt_by_img_cls:
                gt_boxes = gt_by_img_cls[key]
                all_gt.extend([1] * len(gt_boxes))  # Mark as matched/unmatched
            
            # Collect predictions
            if key in pred_by_img_cls:
                preds = pred_by_img_cls[key]
                # Sort by score
                preds.sort(key=lambda x: x[0], reverse=True)
                
                scores = [p[0] for p in preds]
                boxes = [p[1] for p in preds]
                
                all_pred_scores.extend(scores)
                all_pred_boxes.extend(boxes)
        
        # Skip if no ground truth for this class
        if not all_gt:
            continue
        
        # AP is 0 if no predictions for this class
        if not all_pred_scores:
            aps.append(0)
            continue
        
        # Calculate TP and FP
        tp = np.zeros(len(all_pred_scores))
        fp = np.zeros(len(all_pred_scores))
        
        # Sort all prediction boxes by score
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        
        for i, pred_idx in enumerate(sorted_indices):
            pred_box = all_pred_boxes[pred_idx]
            
            # Find the best matching ground truth
            best_iou = -np.inf
            best_gt_idx = -1
            
            for gt_idx, gt_flag in enumerate(all_gt):
                if gt_flag == 0:  # Already matched
                    continue
                
                gt_box = all_gt[gt_idx]
                
                # Calculate IoU
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Mark as TP if best IoU exceeds threshold, otherwise FP
            if best_iou >= iou_threshold:
                tp[i] = 1
                all_gt[best_gt_idx] = 0  # Mark as matched
            else:
                fp[i] = 1
        
        # Calculate cumulative values
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        recall = cumsum_tp / len(all_gt)
        
        # Calculate AP (using 11-point interpolation)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        aps.append(ap)
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0
    
    return mAP


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = intersection / (box1_area + box2_area - intersection)
    
    return iou


def run_inference(model, test_loader, device, output_file):
    """
    Run inference for Task1: Detect each digit.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test images
        device: Device to run inference on
        output_file: Path to save results
        
    Returns:
        List of detection results
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader):
            images = list(image.to(device) for image in images)
            
            # Run prediction
            outputs = model(images)
            
            # Process predictions for each image
            for i, output in enumerate(outputs):
                img_id = img_ids[i]
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Filter low confidence predictions
                threshold = 0.4  # Lower threshold for better recall
                keep = scores >= threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Apply non-maximum suppression (NMS)
                if len(boxes) > 0:
                    final_boxes = []
                    final_scores = []
                    final_labels = []
                    
                    # Group by class, apply NMS per class
                    unique_labels = np.unique(labels)
                    for cls in unique_labels:
                        cls_indices = np.where(labels == cls)[0]
                        cls_boxes = boxes[cls_indices]
                        cls_scores = scores[cls_indices]
                        
                        # Copy boxes for NMS
                        cls_boxes_xyxy = cls_boxes.copy()
                        
                        # Calculate areas
                        areas = (
                            (cls_boxes_xyxy[:, 2] - cls_boxes_xyxy[:, 0]) * 
                            (cls_boxes_xyxy[:, 3] - cls_boxes_xyxy[:, 1])
                        )
                        # Sort by score in descending order
                        order = cls_scores.argsort()[::-1]
                        
                        keep_indices = []
                        while order.size > 0:
                            i = order[0]  # Select highest scoring box
                            keep_indices.append(i)
                            
                            if order.size == 1:
                                break
                                
                            # Calculate IoU
                            xx1 = np.maximum(
                                cls_boxes_xyxy[i, 0], 
                                cls_boxes_xyxy[order[1:], 0]
                            )
                            yy1 = np.maximum(
                                cls_boxes_xyxy[i, 1], 
                                cls_boxes_xyxy[order[1:], 1]
                            )
                            xx2 = np.minimum(
                                cls_boxes_xyxy[i, 2], 
                                cls_boxes_xyxy[order[1:], 2]
                            )
                            yy2 = np.minimum(
                                cls_boxes_xyxy[i, 3], 
                                cls_boxes_xyxy[order[1:], 3]
                            )
                            
                            w = np.maximum(0.0, xx2 - xx1)
                            h = np.maximum(0.0, yy2 - yy1)
                            inter = w * h
                            
                            iou = inter / (
                                areas[i] + areas[order[1:]] - inter
                            )
                            
                            # Keep boxes with IoU below threshold
                            inds = np.where(iou <= 0.3)[0]
                            order = order[inds + 1]
                        
                        # Add kept boxes
                        final_boxes.extend(cls_boxes[keep_indices])
                        final_scores.extend(cls_scores[keep_indices])
                        final_labels.extend([cls] * len(keep_indices))
                    
                    boxes = np.array(final_boxes)
                    scores = np.array(final_scores)
                    labels = np.array(final_labels)
                
                # Create annotations for each detected digit [x_min, y_min, w, h]
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create COCO format annotation
                    ann = {
                        'image_id': int(img_id),
                        'category_id': int(label),  # Digit class (1-10 maps to 0-9)
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'score': float(score)
                    }
                    
                    results.append(ann)
    
    # Save results to pred.json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Task1 predictions saved to {output_file}")
    return results


def calculate_digit_numbers(results, output_file):
    """
    Calculate Task2 predictions (entire numbers in images).
    
    Args:
        results: List of Task1 detection results
        output_file: Path to save results
        
    Returns:
        List of number predictions
    """
    # Group digits by image with position info
    img_to_digits_with_pos = {}
    
    for ann in results:
        img_id = ann['image_id']
        label = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        score = ann['score']
        
        # Calculate digit center coordinates
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        if img_id not in img_to_digits_with_pos:
            img_to_digits_with_pos[img_id] = []
        
        # Save digit and position info
        img_to_digits_with_pos[img_id].append({
            'digit': label,
            'center_x': center_x,
            'center_y': center_y,
            'bbox': bbox,
            'score': score,
            'width': bbox[2],
            'height': bbox[3]
        })
    
    # Calculate numbers from detected digits
    number_predictions = []
    
    # Get all possible image IDs
    all_img_ids = set()
    for result in results:
        all_img_ids.add(result['image_id'])
    
    # Process each image
    for img_id in range(1, max(all_img_ids) + 1):
        if img_id in img_to_digits_with_pos and img_to_digits_with_pos[img_id]:
            digits_info = img_to_digits_with_pos[img_id]
            
            # Sort by score, keep high confidence predictions
            digits_info = sorted(
                digits_info, 
                key=lambda d: d['score'], 
                reverse=True
            )
            
            # Get average digit width and height for grouping threshold
            avg_width = np.mean([d['width'] for d in digits_info])
            avg_height = np.mean([d['height'] for d in digits_info])
            
            # Get coordinates for analysis
            x_coords = [d['center_x'] for d in digits_info]
            y_coords = [d['center_y'] for d in digits_info]
            
            # Vertical threshold for grouping
            y_threshold = avg_height * 0.6
            
            # Use DBSCAN to find rows
            if len(digits_info) > 1:
                try:
                    # Format y coordinates for DBSCAN
                    y_array = np.array(y_coords).reshape(-1, 1)
                    
                    # Cluster with DBSCAN
                    dbscan = DBSCAN(eps=y_threshold, min_samples=1)
                    clusters = dbscan.fit_predict(y_array)
                    
                    # Group by cluster
                    groups = {}
                    for i, cluster_id in enumerate(clusters):
                        if cluster_id not in groups:
                            groups[cluster_id] = []
                        groups[cluster_id].append(digits_info[i])
                    
                    # Sort digits within each row by x coordinate
                    sorted_rows = []
                    for cluster_id in sorted(groups.keys()):
                        row = groups[cluster_id]
                        # Sort by x coordinate
                        sorted_row = sorted(row, key=lambda d: d['center_x'])
                        sorted_rows.append(sorted_row)
                    
                    # Sort rows by y coordinate (top to bottom)
                    sorted_rows = sorted(
                        sorted_rows, 
                        key=lambda row: np.mean([d['center_y'] for d in row])
                    )
                    
                    # Reorder digits
                    digits = []
                    for row in sorted_rows:
                        row_digits = [d['digit'] for d in row]
                        digits.extend(row_digits)
                    
                except Exception as e:
                    print(f"DBSCAN clustering failed, fallback to basic sort: {e}")
                    # Fallback to basic sorting method
                    y_values = [d['center_y'] for d in digits_info]
                    min_y, max_y = min(y_values), max(y_values)
                    
                    # Check if all digits are in one row
                    if max_y - min_y < avg_height * 1.5:
                        # Sort by x coordinate (left to right)
                        sorted_digits_info = sorted(
                            digits_info, 
                            key=lambda d: d['center_x']
                        )
                        digits = [d['digit'] for d in sorted_digits_info]
                    else:
                        # Multiple rows: use simpler row grouping
                        sorted_by_y = sorted(
                            digits_info, 
                            key=lambda d: d['center_y']
                        )
                        
                        # Group rows
                        groups = []
                        current_group = [sorted_by_y[0]]
                        current_y = sorted_by_y[0]['center_y']
                        
                        for digit_info in sorted_by_y[1:]:
                            if abs(digit_info['center_y'] - current_y) <= y_threshold:
                                current_group.append(digit_info)
                            else:
                                # Start new group
                                groups.append(current_group)
                                current_group = [digit_info]
                                current_y = digit_info['center_y']
                        
                        # Add last group
                        if current_group:
                            groups.append(current_group)
                        
                        # Sort digits within each group by x coordinate
                        digits = []
                        for group in groups:
                            sorted_group = sorted(
                                group, 
                                key=lambda d: d['center_x']
                            )
                            group_digits = [d['digit'] for d in sorted_group]
                            digits.extend(group_digits)
            else:
                # Only one digit
                digits = [digits_info[0]['digit']]
            
            # Convert class IDs to actual digits (category_id 1-10 maps to 0-9)
            converted_digits = [d-1 if d <= 10 else d for d in digits]
            
            # Filter out background class or anomalies
            valid_digits = [d for d in converted_digits if 0 <= d <= 9]
            
            # Combine into number
            if valid_digits:
                number = int(''.join(map(str, valid_digits)))
                number_predictions.append({
                    'image_id': img_id, 
                    'pred_label': number
                })
            else:
                number_predictions.append({'image_id': img_id, 'pred_label': -1})
        else:
            # No digits detected
            number_predictions.append({'image_id': img_id, 'pred_label': -1})
    
    # Convert predictions to DataFrame
    df = pd.DataFrame(number_predictions)
    
    # Save as CSV
    df.to_csv(output_file, index=False)
    
    print(f"Task2 predictions saved to {output_file}")
    return number_predictions


def visualize_results(image_path, predictions, output_path=None):
    """
    Visualize detection results.
    
    Args:
        image_path: Path to the original image
        predictions: List of detection results for this image
        output_path: Path to save visualization (optional)
    """
    # Read original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes and labels
    for pred in predictions:
        bbox = pred['bbox']
        x, y, w, h = [int(v) for v in bbox]
        label = pred['category_id']
        score = pred['score']
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label (convert category_id to actual digit)
        actual_digit = label - 1 if label <= 10 else label
        text = f"{actual_digit}: {score:.2f}"
        cv2.putText(
            image, text, (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    # Display or save image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def main():
    """Main function to run inference."""
    # Set paths
    test_dir = "test"
    model_path = "fasterrcnn_model_best.pth"
    task1_output = "pred.json"
    task2_output = "pred.csv"
    
    # Create test dataset
    test_dataset = TestDigitDataset(
        test_dir, 
        transform=get_transform(train=False)
    )
    
    # Adjust batch size based on hardware
    batch_size = 8
    
    # Print test dataset size
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x)), 
        num_workers=4
    )
    
    # Load model
    # 10 digit classes (class IDs 1-10 map to digits 0-9) + background (0)
    num_classes = 11
    model = get_model(num_classes)
    
    # Try to load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Continuing with initialized model...")
    
    model.to(device)
    model.eval()
    
    # Run Task1 inference
    results = run_inference(model, test_loader, device, task1_output)
    
    # Run Task2 inference
    calculate_digit_numbers(results, task2_output)
    
    # Visualize some example results
    for i in range(5):  # Show first 5 test images
        img_id = i + 1
        img_path = os.path.join(test_dir, f"{img_id}.png")
        
        # Get predictions for this image
        img_preds = [pred for pred in results if pred['image_id'] == img_id]
        
        # Visualize results
        visualize_results(img_path, img_preds, f"prediction_{img_id}.png")
    
    print("Inference and visualization completed!")


if __name__ == "__main__":
    main()