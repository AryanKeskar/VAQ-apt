# Isolated Experiment Analysis Report: Resize

## 1. Executive Summary
- **Experiment Strategy**: Resize
- **Total Evaluated Samples**: 240
- **Overall Mean Cosine Similarity**: `0.7018`
- **Overall Median Cosine Similarity**: `0.7037`
- **Standard Deviation**: `0.2730`

## 2. Performance Across Context Resolutions

| Context Resolution | Mean Score (124px Gaze) | Mean Score (512px Gaze) | Combined Mean Score |
| :---: | :---: | :---: | :---: |
| **224px** | `0.6662` | `0.7086` | **`0.6874`** |
| **228px** | `0.6834` | `0.7003` | **`0.6919`** |
| **320px** | `0.7000` | `0.6477` | **`0.6738`** |
| **384px** | `0.7006` | `0.6743` | **`0.6874`** |
| **512px** | `0.7234` | `0.7181` | **`0.7207`** |
| **640px** | `0.7136` | `0.7516` | **`0.7326`** |
| **704px** | `0.6935` | `0.7375` | **`0.7155`** |
| **896px** | `0.6935` | `0.7172` | **`0.7053`** |

**Peak Performing Context Resolution**: `640px` with a combined mean score of **`0.7326`**.

## 3. Question-Category Performance Breakdown

| Question Category | Sample Count | Mean Score | Std Dev |
| :--- | :---: | :---: | :---: |
| Equipment & Tools | 32 | `1.0000` | `0.0000` |
| Food & Objects | 80 | `0.7457` | `0.2874` |
| Action Recognition | 16 | `0.7105` | `0.0035` |
| Gaze & Spatial Attention | 16 | `0.5955` | `0.0330` |
| General Object QA | 96 | `0.5822` | `0.2659` |