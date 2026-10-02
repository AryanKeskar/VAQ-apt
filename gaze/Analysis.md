# Analysis & Design Blueprint: Experiment Results Analysis Script

## 1. Overview & Objectives
The goal of this Python script is to perform automated, modular analysis and comparative benchmarking on the VQA (Visual Question Answering) experiment results located in `gaze/experiment_results/`.

The analysis pipeline maintains **strict isolation between the three distinct experiment strategies**, evaluating each strategy independently before running a head-to-head comparative ranking to determine **which strategy performs best overall**.

The 3 core experiment strategies are:
1. **Experiment 1: Full-Image Resize Strategy** (`resize_experiment_124.json`, `resize_experiment_512.json`)
2. **Experiment 2: Center-Crop Zoom Strategy** (`center_zoom_experiment_124.json`, `center_zoom_experiment_512.json`)
3. **Experiment 3: Gaze-Centered Zoom Strategy** (`gaze_zoom_experiment.json`, `gaze_zoom_experiment_512.json`)

---

## 2. Core Functional Modules

### A. Data Ingestion & Schema Normalization (`DataLoader`)
- **Isolated Experiment Parsers**: Separate ingestion routines for each of the 3 experiment types.
- **Schema Unification**: Standardize divergent key names into a consistent internal schema:
  - `question` / `Question` $\rightarrow$ `question`
  - `model_response` / `Predicted Answer` $\rightarrow$ `predicted_answer`
  - `ground_truth_answers` / `Ground Truth Answers` $\rightarrow$ `ground_truth`
  - `final_score` / `Cosine Similarity Score` $\rightarrow$ `similarity_score`
- **Metadata Tagging**: Explicitly tag each sample with `experiment_type` (`Resize`, `CenterZoom`, `GazeZoom`), `gaze_patch_size` (`124`, `512`), and `context_resolution` (`224` to `896`).

---

### B. Isolated Experiment Analysis (`IndependentExperimentAnalyzer`)
Each experiment strategy is analyzed independently across its own dataset:
1. **Resize Experiment Evaluation**:
   - Resolution scaling profile (224px to 896px).
   - Impact of gaze patch size (124px vs 512px).
   - Category-level accuracy distribution.
2. **Center Zoom Experiment Evaluation**:
   - Center-crop context resolution scaling profile.
   - Impact of gaze patch size (124px vs 512px).
   - Performance breakdown across visual query types.
3. **Gaze Zoom Experiment Evaluation**:
   - Gaze-centered crop resolution scaling profile.
   - Impact of gaze patch size (124px vs 512px).
   - Performance breakdown across visual query types.

---

### C. Head-to-Head Strategy Comparator & Winner Determination (`StrategyComparator`)
Once individual metrics are computed, the comparator module evaluates all three strategies side-by-side to determine **which strategy performs best**:

1. **Overall Leaderboard & Ranking**:
   - Rank strategies by overall mean/median cosine similarity score across all configurations.
2. **Resolution-Specific Winners**:
   - Identify the winning strategy for each context resolution (224, 228, 320, 384, 512, 640, 704, 896).
3. **Gaze Patch Size Breakdown**:
   - Best strategy when using 124px local gaze patch.
   - Best strategy when using 512px local gaze patch.
4. **Question-Category Winners**:
   - Compare strategy accuracy per question intent (e.g. Object queries vs. Action queries vs. Spatial queries).
5. **Statistical Significance Testing**:
   - Run paired t-tests / Wilcoxon signed-rank tests between strategies to verify if performance differences (e.g. Gaze Zoom vs. Center Zoom vs. Resize) are statistically significant ($p < 0.05$).

---

### D. Error & Hallucination Analysis (`ErrorAnalyzer`)
- **Strategy-Specific Failures**: Extract low-score samples (score < 0.4) independently for each strategy to identify unique failure modes (e.g. cropped out target object in Center Zoom vs distorted aspect ratio in Resize).
- **Cross-Strategy Consistency**: Track sample instances where all 3 strategies failed vs. instances where Gaze Zoom succeeded while Center Zoom / Resize failed.

---

### E. Visualization & Output Delivery (`Visualizer` & `ReportGenerator`)

All visualizations and tables will be generated and stored in easily accessible formats:

#### 1. Image Charts (`.png` High-Resolution Files)
Charts will be generated using `matplotlib` / `seaborn` and saved as standalone `.png` images in `gaze/analysis_outputs/plots/`:
- **`resolution_scaling_comparison.png`**: Multi-line chart showing Context Resolution vs. Accuracy for all 3 strategies.
- **`strategy_winner_heatmap.png`**: Heatmap grid showing Strategy vs. Resolution color-coded by performance rank.
- **`question_category_bar_chart.png`**: Grouped bar chart comparing strategy performance across question categories.
- **`gaze_patch_impact_chart.png`**: Bar chart comparing 124px vs 512px gaze patch effects per strategy.

#### 2. Formatted Tables
Tables will be delivered in **two complementary formats**:
- **Inline Markdown Tables**: Beautifully formatted Markdown tables embedded directly inside `.md` reports for instant reading in IDEs or Markdown viewers.
- **Standalone `.csv` Files**: Saved in `gaze/analysis_outputs/tables/` (`overall_leaderboard.csv`, `resolution_matrix.csv`, `category_breakdown.csv`) for opening in Excel or data tools.

#### 3. Integrated Master Report (`comparative_leaderboard.md`)
The master Markdown report will **embed the `.png` charts inline alongside the tables** (e.g., `![Resolution Scaling](plots/resolution_scaling_comparison.png)`). Opening `comparative_leaderboard.md` will display the complete analysis—charts, formatted tables, and text explanations—all together in one document.

---

## 3. Proposed Output Directory Structure

```
gaze/
├── analyze_results.py              # Main entry point script
├── experiment_results/             # Target JSON result data
└── analysis_outputs/               # Output directory for plots, tables, and reports
    ├── comparative_leaderboard.md  # Master report (embeds charts + markdown tables)
    ├── isolated_reports/
    │   ├── resize_analysis.md
    │   ├── center_zoom_analysis.md
    │   └── gaze_zoom_analysis.md
    ├── plots/                      # High-res PNG chart image files
    │   ├── resolution_scaling_comparison.png
    │   ├── strategy_winner_heatmap.png
    │   ├── question_category_bar_chart.png
    │   └── gaze_patch_impact_chart.png
    └── tables/                     # CSV table files for Excel / Data viewers
        ├── overall_leaderboard.csv
        ├── resolution_matrix.csv
        └── category_breakdown.csv
```

---

## 4. Key Questions & Hypotheses to Answer

1. **Which strategy achieves the overall highest VQA accuracy: Full-Image Resize, Center-Crop Zoom, or Gaze-Centered Zoom?**
2. **Is Gaze-Centered Zoom consistently superior to Center-Crop Zoom across all resolutions, or only at specific crop sizes?**
3. **Under what conditions (resolution / question type) does Full-Image Resize outperform localized cropping strategies?**
4. **Are the accuracy differences between the top-performing strategies statistically significant?**
