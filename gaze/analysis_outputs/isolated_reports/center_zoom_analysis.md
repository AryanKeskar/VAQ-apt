# Isolated Experiment Analysis Report: Center Zoom

## 1. Executive Summary
- **Experiment Strategy**: Center Zoom
- **Total Evaluated Samples**: 240
- **Overall Mean Cosine Similarity**: `0.6473`
- **Overall Median Cosine Similarity**: `0.5663`
- **Standard Deviation**: `0.2852`

## 2. Performance Across Context Resolutions

| Context Resolution | Mean Score (124px Gaze) | Mean Score (512px Gaze) | Combined Mean Score |
| :---: | :---: | :---: | :---: |
| **224px** | `0.5606` | `0.6708` | **`0.6157`** |
| **228px** | `0.6066` | `0.6278` | **`0.6172`** |
| **320px** | `0.6059` | `0.5840` | **`0.5949`** |
| **384px** | `0.6267` | `0.6659` | **`0.6463`** |
| **512px** | `0.6268` | `0.6708` | **`0.6488`** |
| **640px** | `0.6112` | `0.6376` | **`0.6244`** |
| **704px** | `0.6802` | `0.7221` | **`0.7011`** |
| **896px** | `0.7897` | `0.6694` | **`0.7296`** |

**Peak Performing Context Resolution**: `896px` with a combined mean score of **`0.7296`**.

## 3. Question-Category Performance Breakdown

| Question Category | Sample Count | Mean Score | Std Dev |
| :--- | :---: | :---: | :---: |
| Gaze & Spatial Attention | 16 | `0.7751` | `0.1840` |
| Equipment & Tools | 32 | `0.7331` | `0.2942` |
| Food & Objects | 80 | `0.7081` | `0.3211` |
| Action Recognition | 16 | `0.7004` | `0.0322` |
| General Object QA | 96 | `0.5378` | `0.2534` |