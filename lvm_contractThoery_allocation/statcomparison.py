import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the data
human_df = pd.read_excel("human_perception.xlsx")
vlm_df = pd.read_excel("VLM.xlsx")
real_df = pd.read_excel("real.xlsx")

# Compute means for human and VLM assessments
human_scores = human_df.iloc[:, 1:].mean(axis=1)  # Average human perception per image
vlm_scores = vlm_df.iloc[:, 1:].mean(axis=1)  # Average VLM perception per image
real_scores = real_df.iloc[:, 1]  # Actual difficulty levels

# Error Analysis
human_mae = np.mean(np.abs(human_scores - real_scores))
vlm_mae = np.mean(np.abs(vlm_scores - real_scores))

human_rmse = np.sqrt(np.mean((human_scores - real_scores) ** 2))
vlm_rmse = np.sqrt(np.mean((vlm_scores - real_scores) ** 2))

human_real_pearson, human_real_p = stats.pearsonr(human_scores, real_scores)
vlm_real_pearson, vlm_real_p = stats.pearsonr(vlm_scores, real_scores)

human_real_spearman, human_real_spearman_p = stats.spearmanr(human_scores, real_scores)
vlm_real_spearman, vlm_real_spearman_p = stats.spearmanr(vlm_scores, real_scores)

# Statistical significance of errors
human_errors = np.abs(human_scores - real_scores)
vlm_errors = np.abs(vlm_scores - real_scores)

t_stat_error, p_value_error = stats.ttest_rel(human_errors, vlm_errors)
wilcoxon_stat_error, wilcoxon_p_error = stats.wilcoxon(human_errors, vlm_errors)


# Error Comparison Summary
def format_p(p_val):
    return "<0.001" if p_val < 0.001 else round(p_val, 6)


error_comparison = pd.DataFrame({
    "Metric": ["Mean Absolute Error (MAE)", "Root Mean Square Error (RMSE)",
               "Pearson Correlation with Real", "Spearman Correlation with Real",
               "Pearson p-value", "Spearman p-value"],
    "Human": [human_mae, human_rmse, human_real_pearson, human_real_spearman, format_p(human_real_p), format_p(human_real_spearman_p)],
    "VLM": [vlm_mae, vlm_rmse, vlm_real_pearson, vlm_real_spearman, format_p(vlm_real_p), format_p(vlm_real_spearman_p)]
})

print(error_comparison)