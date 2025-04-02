import numpy as np
from typing import List, Tuple

def calculate_metrics(detected_regions: List[Tuple[float, float]],
                     ground_truth_regions: List[Tuple[float, float]]) -> dict:
    """
    計算評估指標
    
    Args:
        detected_regions: 檢測到的語音區段
        ground_truth_regions: 真實的語音區段
        
    Returns:
        dict: 包含各項評估指標的字典
    """
    # 計算重疊區域
    overlap = calculate_overlap(detected_regions, ground_truth_regions)
    
    # 計算指標
    total_detected = sum(end - start for start, end in detected_regions)
    total_truth = sum(end - start for start, end in ground_truth_regions)
    
    if total_detected == 0 or total_truth == 0:
        return {
            'accuracy': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    precision = overlap / total_detected if total_detected > 0 else 0
    recall = overlap / total_truth if total_truth > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def calculate_overlap(regions1: List[Tuple[float, float]],
                     regions2: List[Tuple[float, float]]) -> float:
    """
    計算兩組區間的重疊時間
    """
    overlap = 0.0
    i = j = 0
    
    while i < len(regions1) and j < len(regions2):
        start = max(regions1[i][0], regions2[j][0])
        end = min(regions1[i][1], regions2[j][1])
        
        if start < end:
            overlap += end - start
            
        if regions1[i][1] < regions2[j][1]:
            i += 1
        else:
            j += 1
            
    return overlap 