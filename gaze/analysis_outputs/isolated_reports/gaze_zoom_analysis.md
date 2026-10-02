# Isolated Experiment Analysis Report: Gaze Zoom

## 1. Executive Summary
- **Experiment Strategy**: Gaze Zoom
- **Total Evaluated Samples**: 240
- **Overall Mean Cosine Similarity**: `0.6509`
- **Overall Median Cosine Similarity**: `0.6038`
- **Standard Deviation**: `0.3007`

## 2. Performance Across Context Resolutions

| Context Resolution | Mean Score (124px Gaze) | Mean Score (512px Gaze) | Combined Mean Score |
| :---: | :---: | :---: | :---: |
| **224px** | `0.5752` | `0.6524` | **`0.6138`** |
| **228px** | `0.6202` | `0.6453` | **`0.6327`** |
| **320px** | `0.5664` | `0.6220` | **`0.5942`** |
| **384px** | `0.6782` | `0.6659` | **`0.6720`** |
| **512px** | `0.6388` | `0.6149` | **`0.6269`** |
| **640px** | `0.5953` | `0.6998` | **`0.6475`** |
| **704px** | `0.6414` | `0.6729` | **`0.6571`** |
| **896px** | `0.7698` | `0.7564` | **`0.7631`** |

**Peak Performing Context Resolution**: `896px` with a combined mean score of **`0.7631`**.

## 3. Question-Category Performance Breakdown

| Question Category | Sample Count | Mean Score | Std Dev |
| :--- | :---: | :---: | :---: |
| Gaze & Spatial Attention | 16 | `0.7682` | `0.1955` |
| Equipment & Tools | 32 | `0.7560` | `0.3051` |
| Action Recognition | 16 | `0.7150` | `0.0071` |
| Food & Objects | 80 | `0.6958` | `0.3259` |
| General Object QA | 96 | `0.5483` | `0.2892` |