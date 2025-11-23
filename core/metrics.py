"""Utility functions for metrics calculation."""

import json
from typing import Dict, Any, List
from rapidfuzz import fuzz


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from text using balanced brace matching.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON object if found, None otherwise
    """
    start = text.find('{')
    if start == -1:
        return None
    
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                break
    
    return None


def calculate_fuzzy_metrics(predicted: Dict, ground_truth: Dict, threshold: float = 85.0) -> Dict[str, Any]:
    """
    Calculate P/R/F1 metrics with fuzzy matching support.
    
    Args:
        predicted: Predicted dictionary with keys and list values
        ground_truth: Ground truth dictionary with keys and list values
        threshold: Minimum similarity score (0-100) for fuzzy match
        
    Returns:
        Dictionary with tp, fp, fn counts and fuzzy_matches list
    """
    # Normalize predicted items
    pred_items = []
    for key, values in predicted.items():
        if not isinstance(values, list):
            values = [values]
        for val in values:
            pred_items.append((key, str(val).strip().lower()))
    
    # Normalize ground truth items
    gt_items = []
    for key, values in ground_truth.items():
        if not isinstance(values, list):
            values = [values]
        for val in values:
            gt_items.append((key, str(val).strip().lower()))
    
    # Track matches
    matched_gt = set()
    matched_pred = set()
    fuzzy_matches = []
    
    # Find matches (exact + fuzzy)
    for pred_idx, (pred_key, pred_val) in enumerate(pred_items):
        best_match_score = 0
        best_match_idx = -1
        
        for gt_idx, (gt_key, gt_val) in enumerate(gt_items):
            if gt_idx in matched_gt:
                continue
            
            # Keys must match exactly
            if pred_key != gt_key:
                continue
            
            # Check value match (exact or fuzzy)
            if pred_val == gt_val:
                similarity = 100
            else:
                similarity = fuzz.token_sort_ratio(pred_val, gt_val)
            
            if similarity >= threshold and similarity > best_match_score:
                best_match_score = similarity
                best_match_idx = gt_idx
        
        # Mark matches
        if best_match_idx >= 0:
            matched_pred.add(pred_idx)
            matched_gt.add(best_match_idx)
            fuzzy_matches.append({
                "key": pred_key,
                "predicted": pred_val,
                "ground_truth": gt_items[best_match_idx][1],
                "similarity": best_match_score
            })
    
    tp = len(matched_pred)
    fp = len(pred_items) - len(matched_pred)
    fn = len(gt_items) - len(matched_gt)
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fuzzy_matches": fuzzy_matches
    }


def normalize_dict_values_to_lists(data: Dict) -> Dict:
    """
    Normalize dictionary values to lists.
    
    Args:
        data: Dictionary with mixed value types
        
    Returns:
        Dictionary with all values as lists
    """
    normalized = {}
    for key, value in data.items():
        if isinstance(value, str):
            if ',' in value:
                normalized[key] = [item.strip() for item in value.split(',')]
            else:
                normalized[key] = [value] if value else []
        elif isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [str(value)] if value else []
    return normalized

