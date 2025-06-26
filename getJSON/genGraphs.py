import os
import re
import shutil
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger

warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = (12, 8)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
TEMP_PLOT_DIR = os.path.join(BASE_DIR, "temp_plots")

# Create temp plot directory if it doesn't exist
if not os.path.exists(TEMP_PLOT_DIR):
    os.makedirs(TEMP_PLOT_DIR)

# Global list to track all saved plots for combining
saved_plots = []

def save_plot(filename, fig=None):
    """Save plot with timestamp and track for combining"""
    global saved_plots
    
    if fig is None:
        fig = plt.gcf()
    
    filepath = os.path.join(TEMP_PLOT_DIR, f'{filename}.pdf')
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    # Track the saved plot for later combining
    saved_plots.append(filepath)
    
    return filepath

def normalize_model_name(llm_name):
    """Extract base model name by removing vision indicators and other suffixes"""
    # Remove vision input indicators
    base_name = llm_name.replace('*ImageInput*', '').strip()
    # Remove any trailing version numbers or suffixes that might vary
    base_name = base_name.replace('-qat', '').replace('-it', '').strip()
    return base_name

def extract_model_info(llm_name):
    """Extract comprehensive model information from LLM name"""
    llm_lower = llm_name.lower()
    
    # Determine if it's a vision model
    Image_Input = '*imageinput*' in llm_lower or 'vision' in llm_lower or 'vl' in llm_lower
    
    # Determine prompt type
    if 'np' in llm_lower or 'ner' in llm_lower:
        prompt_type = 'NER'
    else:
        prompt_type = 'Normal'

    # Extract parameter size (look for numbers followed by 'b')
    param_match = re.search(r'(\d+(?:\.\d+)?)b', llm_lower)
    param_size = float(param_match.group(1)) if param_match else None
    
    # Extract model family
    if 'qwen' in llm_lower:
        family = 'Qwen'
    elif 'gpt' in llm_lower:
        family = 'GPT'
        if 'mini' in llm_lower:
            param_size = 8
        elif 'nano' in llm_lower:
            param_size = 2
        else:
            param_size = 200
    elif 'gemm' in llm_lower or 'gemini' in llm_lower:
        family = 'Gemma/Gemini'
    elif 'llama' in llm_lower:
        family = 'Llama'
    elif 'granite' in llm_lower:
        family = 'Granite'
    elif 'mistral' in llm_lower or 'devstral' in llm_lower:
        family = 'Mistral'
    elif 'nuextract' in llm_lower or 'nuextract' in llm_lower:
        family = 'NuExtract'
    else:
        family = "GLiNER"  # Default to GLiNER if no match
    
    # Create base model name (without vision indicators)
    base_name = normalize_model_name(llm_name)
    
    # Extract token size category
    if param_size:
        if param_size < 2:
            token_category = 'Small (<2B)'
        elif param_size < 10:
            token_category = 'Medium (2-10B)'
        elif param_size < 50:
            token_category = 'Large (10-50B)'
        else:
            token_category = 'XLarge (>50B)'
    else:
        token_category = 'Unknown'
    
    return family, param_size, Image_Input, base_name, token_category, prompt_type

def create_overall_performance_plot(df, metric_name='F1score', filename_prefix='01'):
    """
    Create an overall performance visualization for a given metric, grouped by base model names.
    This combines text/vision variants and averages across hospitals.
    """
    if 'Base_Model' not in df.columns:
        df['Base_Model'] = df['LLM'].apply(normalize_model_name)
    
    metric_mean_col = f'{metric_name}_Mean'
    metric_std_col = f'{metric_name}_Std'
    
    agg_dict = {
        metric_name: ['mean', 'std', 'count'],
        'Family': 'first',
        'Source': 'first',
        'Image_Input': lambda x: any(x),
        'Prompt_Type': lambda x: 'NER' in x.values
    }
    
    other_metrics = ['F1score', 'Accuracy', 'Precision', 'Recall']
    if metric_name in other_metrics:
        other_metrics.remove(metric_name)
    for m in other_metrics:
        agg_dict[m] = 'mean'

    grouped_models = df.groupby('Base_Model').agg(agg_dict).round(2)
    
    new_cols = [metric_mean_col, metric_std_col, 'Count', 'Family', 'Source', 'Has_Vision', 'Has_NER_Prompt'] + [f'{m}_Mean' for m in other_metrics]
    grouped_models.columns = new_cols
    grouped_models = grouped_models.reset_index()
    
    grouped_models_sorted = grouped_models.sort_values(metric_mean_col, ascending=True)
    
    # Changed to a 2x1 layout for better PDF readability
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 24))
    
    bars = ax1.barh(range(len(grouped_models_sorted)), grouped_models_sorted[metric_mean_col])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    family_colors = {
        'Qwen': colors[0], 'GPT': colors[1], 'Gemma/Gemini': colors[2],
        'Llama': colors[3], 'Granite': colors[4], 'Mistral': colors[5], 'NuExtract': colors[6]
    }
    
    for i, (idx, row) in enumerate(grouped_models_sorted.iterrows()):
        bar_color = family_colors.get(row['Family'], colors[7])
        bars[i].set_color(bar_color)
        
        hatch = ''
        if row['Has_Vision']:
            hatch += '///'
        if row['Has_NER_Prompt']:
            hatch += 'x'
        
        if hatch:
            bars[i].set_hatch(hatch)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)
    
    model_labels = []
    for __, row in grouped_models_sorted.iterrows():
        label = f"{row['Base_Model']}"
        extras = []
        if row['Has_Vision']:
            extras.append("Image Input")
        if row['Has_NER_Prompt']:
            extras.append("NER Prompt")
        if extras:
            label += f" ({', '.join(extras)})"
        label += f" (n={int(row['Count'])})"
        model_labels.append(label)
    
    ax1.set_yticks(range(len(grouped_models_sorted)))
    ax1.set_yticklabels(model_labels, fontsize=9)
    ax1.set_xlabel(f'Average {metric_name}', fontsize=12)
    ax1.set_title(f'Overall {metric_name} Performance - Models Grouped by Base Name\n(Averaged across hospitals and input types)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(grouped_models_sorted.iterrows()):
        ax1.text(row[metric_mean_col] + 1, i, f"{row[metric_mean_col]:.1f}±{row[metric_std_col]:.1f}", 
                va='center', fontsize=8)
    
    family_summary = grouped_models.groupby('Family').agg({
        metric_mean_col: ['mean', 'std', 'count']
    }).round(2)
    family_summary.columns = [f'Avg_{metric_name}', f'Std_{metric_name}', 'Model_Count']
    family_summary = family_summary.reset_index().sort_values(f'Avg_{metric_name}', ascending=False)
    
    bars2 = ax2.bar(family_summary['Family'], family_summary[f'Avg_{metric_name}'], 
                    yerr=family_summary[f'Std_{metric_name}'], capsize=5)
    
    for i, family in enumerate(family_summary['Family']):
        bars2[i].set_color(family_colors.get(family, colors[7]))
    
    for bar, count in zip(bars2, family_summary['Model_Count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title(f'Average {metric_name} by Model Family', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model Family', fontsize=12)
    ax2.set_ylabel(f'Average {metric_name}', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=family) for family, color in family_colors.items() if family in grouped_models['Family'].values]
    legend_elements.append(Patch(facecolor='lightgray', hatch='///', edgecolor='black', linewidth=1.5, label='Given Image Input'))
    legend_elements.append(Patch(facecolor='lightgray', hatch='x', edgecolor='black', linewidth=1.5, label='Used NER Prompt'))
    
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    save_plot(f'{filename_prefix}_overall_{metric_name.lower()}_grouped_models')
    
    return grouped_models

def load_and_prepare_data():
    """Load and prepare the hospital data with comprehensive feature engineering"""
    # Load the data
    hospitals = pd.read_csv('Hospital.csv')
    # Clean unnecessary columns
    cols_to_drop = [col for col in hospitals.columns if 'Unnamed' in col]
    hospitals = hospitals.drop(columns=cols_to_drop, errors='ignore')
    
    # Extract model information
    model_info = hospitals['LLM'].apply(extract_model_info)
    hospitals[['Family', 'Param_Size', 'Image_Input', 'Base_Model', 'Token_Category', 'Prompt_Type']] = pd.DataFrame(
        model_info.tolist(), index=hospitals.index
    )
        
    # Create input type column
    hospitals['Input_Type'] = hospitals['Image_Input'].map({True: 'Vision', False: 'Text-Only'})
    
    return hospitals

def calculate_summary_statistics(df):
    """Calculate comprehensive summary statistics"""
    # Overall statistics
    overall_stats = df.agg({
        'Accuracy': ['mean', 'std', 'min', 'max'],
        'F1score': ['mean', 'std', 'min', 'max'],
        'Precision': ['mean', 'std', 'min', 'max'],
        'Recall': ['mean', 'std', 'min', 'max'],
        'False Positives': ['mean', 'sum'],
        'False Negatives': ['mean', 'sum'],
        'Incorrect Extractions': ['mean', 'sum']
    }).round(3)
    
    # By model family
    family_stats = df.groupby('Family').agg({
        'Accuracy': ['mean', 'std', 'count'],
        'F1score': ['mean', 'std'],
        'False Positives': ['mean', 'sum'],
        'False Negatives': ['mean', 'sum']
    }).round(3)
    
    # By input type
    input_stats = df.groupby('Input_Type').agg({
        'Accuracy': ['mean', 'std', 'count'],
        'F1score': ['mean', 'std'],
        'False Positives': ['mean', 'sum'],
        'False Negatives': ['mean', 'sum']
    }).round(3)
    
    # By hospital
    hospital_stats = df.groupby('Hospital').agg({
        'Accuracy': ['mean', 'std'],
        'F1score': ['mean', 'std'],
        'False Positives': ['mean', 'sum'],
        'False Negatives': ['mean', 'sum']
    }).round(3)
    

    
    return {
        'overall': overall_stats,
        'family': family_stats,
        'input_type': input_stats,
        'hospital': hospital_stats
    }

def create_matrix_plots(df):
    """Create key matrix plots - performance heatmaps only"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Performance heatmap by Family and Input Type
    pivot_f1 = df.groupby(['Family', 'Input_Type'])['F1score'].mean().unstack()
    sns.heatmap(pivot_f1, annot=True, cmap='RdYlBu_r', fmt='.2f', ax=axes[0])
    axes[0].set_title('Average F1 Score: Family vs Input Type', fontweight='bold')
    
    # Performance heatmap by Family and Hospital
    pivot_acc = df.groupby(['Family', 'Hospital'])['Accuracy'].mean().unstack()
    sns.heatmap(pivot_acc, annot=True, cmap='viridis', fmt='.2f', ax=axes[1])
    axes[1].set_title('Average Accuracy: Family vs Hospital', fontweight='bold')
    
    plt.tight_layout()
    save_plot('04_performance_heatmaps')

def create_error_analysis_plots(df):
    """Create a plot showing the distribution of error types by model family."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Melt the dataframe to have a single column for error counts
    error_data = df.melt(id_vars=['Family', 'Hospital', 'Input_Type'], 
                        value_vars=['False Positives', 'False Negatives'],
                        var_name='Error_Type', value_name='Error_Count')
    
    # Create the boxplot
    sns.boxplot(data=error_data, x='Family', y='Error_Count', hue='Error_Type', ax=ax)
    ax.set_title('Error Distribution by Model Family', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlabel("Model Family")
    ax.set_ylabel("Error Count")
    
    plt.tight_layout()
    save_plot('05_error_analysis_distribution')
    
    return df

def create_source_f1_comparison(df):
    """
    Create a bar chart comparing the overall F1 score by source.
    """
    # Filter out OpenRouter if it has limited data
    df_sources = df[df['Source'] != "OpenRouter"].copy()
    
    # Calculate overall F1 averages by source
    source_f1_avg = df_sources.groupby('Source')['F1score'].agg(['mean', 'std', 'count']).round(2)
    source_f1_avg.columns = ['Mean_F1', 'Std_F1', 'Count']
    source_f1_avg = source_f1_avg.sort_values('Mean_F1', ascending=False)
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    bars = ax.bar(source_f1_avg.index, source_f1_avg['Mean_F1'], 
                  yerr=source_f1_avg['Std_F1'], capsize=5)
    ax.set_title('Average F1 Score by Source', fontsize=16, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Source')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and counts on bars
    for i, (source, row) in enumerate(source_f1_avg.iterrows()):
        ax.text(i, row['Mean_F1'] + row['Std_F1'] + 0.5, 
                f"{row['Mean_F1']:.2f} ± {row['Std_F1']:.2f}\n(n={int(row['Count'])})", 
                ha='center', va='bottom', fontsize=10)
    
    # Color bars differently
    colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#8A2BE2', '#00CED1']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
        
    plt.tight_layout()
    save_plot('06_source_f1_comparison')

def plot_vision_impact(df):
    """
    Plots the impact of using image input on F1 scores using a stacked bar chart
    that ensures all components are positive.
    """
    # Find base models that have both vision and text-only versions
    model_counts = df.groupby('Base_Model')['Input_Type'].nunique()
    models_with_both = model_counts[model_counts > 1].index
    
    if models_with_both.empty:
        print("No models found with both vision and text-only versions. Skipping vision impact plot.")
        return

    plot_df = df[df['Base_Model'].isin(models_with_both)]
    
    # Pivot data to have vision and text-only scores as columns
    pivot_df = plot_df.groupby(['Base_Model', 'Input_Type'])['F1score'].mean().unstack()
    
    # Ensure both columns exist
    if 'Vision' not in pivot_df.columns or 'Text-Only' not in pivot_df.columns:
        print("Skipping vision impact plot due to missing data.")
        return
        
    # Calculate components for a positive-only stacked bar
    pivot_df['Lower_Score'] = pivot_df[['Vision', 'Text-Only']].min(axis=1)
    pivot_df['Difference'] = (pivot_df['Vision'] - pivot_df['Text-Only']).abs()
    pivot_df['Vision_is_better'] = pivot_df['Vision'] > pivot_df['Text-Only']
    
    # Sort by the vision score
    pivot_df = pivot_df.sort_values('Vision', ascending=False)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 10))
    x = range(len(pivot_df.index))
    
    # Plot the base (lower score)
    ax.bar(x, pivot_df['Lower_Score'], width=0.7, label='Common F1 Score', color='#1f77b4')
    
    # Plot the difference on top with conditional coloring
    for i, model in enumerate(pivot_df.index):
        color = '#2ca02c' if pivot_df.loc[model, 'Vision_is_better'] else '#d62728'
        ax.bar(i, pivot_df.loc[model, 'Difference'], 
               bottom=pivot_df.loc[model, 'Lower_Score'], 
               width=0.7, color=color)

    # Add labels and titles
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    ax.set_title('Impact of Image Input on F1 Score', fontsize=16, fontweight='bold')
    ax.set_xlabel('Base Model', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels for the scores
    for i, model in enumerate(pivot_df.index):
        vision_score = pivot_df.loc[model, 'Vision']
        text_score = pivot_df.loc[model, 'Text-Only']
        higher_score = max(vision_score, text_score)
        
        # Place text for the higher score above the bar
        ax.text(i, higher_score + 1, f'{higher_score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Common F1 Score'),
        Patch(facecolor='#2ca02c', label='Improvement with Vision'),
        Patch(facecolor='#d62728', label='Decline with Vision')
    ]
    ax.legend(handles=legend_elements, title='Component')

    plt.tight_layout()
    save_plot('13_vision_impact_comparison')

def plot_prompt_comparison_overall(df):
    """
    Create a bar chart comparing the average F1 scores for different prompt types.
    """
    plt.figure(figsize=(10, 6))
    
    prompt_perf = df.groupby('Prompt_Type')['F1score'].mean().sort_values(ascending=False)
    
    sns.barplot(x=prompt_perf.index, y=prompt_perf.values, palette='viridis')
    
    plt.title('Overall F1 Score Comparison by Prompt Type', fontsize=16, fontweight='bold')
    plt.xlabel('Prompt Type', fontsize=12)
    plt.ylabel('Average F1 Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, score in enumerate(prompt_perf.values):
        plt.text(i, score + 0.5, f'{score:.2f}', ha='center', va='bottom')
        
    plt.tight_layout()
    save_plot('11_prompt_comparison_overall')

def plot_prompt_type_comparison(df):
    """
    Create a stacked bar chart to compare F1 scores for models tested with both NER and Normal prompts.
    """
    # Find models that have results for both prompt types
    model_prompt_counts = df.groupby('Base_Model')['Prompt_Type'].nunique()
    models_with_both_prompts = model_prompt_counts[model_prompt_counts > 1].index
    
    if models_with_both_prompts.empty:
        print("No models found with both NER and Normal prompts. Skipping prompt type comparison plot.")
        return

    plot_df = df[df['Base_Model'].isin(models_with_both_prompts)]
    
    # Pivot data to have NER and Normal scores as columns
    pivot_df = plot_df.groupby(['Base_Model', 'Prompt_Type'])['F1score'].mean().unstack()
    
    # Ensure both columns exist
    if 'NER' not in pivot_df.columns or 'Normal' not in pivot_df.columns:
        print("Skipping prompt type comparison plot due to missing data.")
        return
        
    # Calculate components for a positive-only stacked bar
    pivot_df['Lower_Score'] = pivot_df[['NER', 'Normal']].min(axis=1)
    pivot_df['Difference'] = (pivot_df['NER'] - pivot_df['Normal']).abs()
    pivot_df['NER_is_better'] = pivot_df['NER'] > pivot_df['Normal']
    
    # Sort by the NER score
    pivot_df = pivot_df.sort_values('NER', ascending=False)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 10))
    x = range(len(pivot_df.index))
    
    # Plot the base (lower score)
    ax.bar(x, pivot_df['Lower_Score'], width=0.7, label='Common F1 Score', color='#1f77b4')
    
    # Plot the difference on top with conditional coloring
    for i, model in enumerate(pivot_df.index):
        color = '#2ca02c' if pivot_df.loc[model, 'NER_is_better'] else '#d62728'
        ax.bar(i, pivot_df.loc[model, 'Difference'], 
               bottom=pivot_df.loc[model, 'Lower_Score'], 
               width=0.7, color=color)

    # Add labels and titles
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    ax.set_title('Impact of NER Prompt on F1 Score', fontsize=16, fontweight='bold')
    ax.set_xlabel('Base Model', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels for the scores
    for i, model in enumerate(pivot_df.index):
        ner_score = pivot_df.loc[model, 'NER']
        normal_score = pivot_df.loc[model, 'Normal']
        higher_score = max(ner_score, normal_score)
        
        # Place text for the higher score above the bar
        ax.text(i, higher_score + 1, f'{higher_score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Common F1 Score'),
        Patch(facecolor='#2ca02c', label='Improvement with NER Prompt'),
        Patch(facecolor='#d62728', label='Decline with NER Prompt')
    ]
    ax.legend(handles=legend_elements, title='Component')

    plt.tight_layout()
    save_plot('12_prompt_type_comparison')

def generate_comprehensive_summary_text(df, stats, grouped_models):
    """Generate a comprehensive text summary of the analysis."""
    summary_lines = []
    summary_lines.append("COMPREHENSIVE LLM PERFORMANCE ANALYSIS")
    summary_lines.append("========================================")
    summary_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")

    # Overall Performance
    summary_lines.append("OVERALL PERFORMANCE METRICS:")
    summary_lines.append(f"   • Average F1 Score: {stats['overall']['F1score']['mean']:.3f} (Std: {stats['overall']['F1score']['std']:.3f})")
    summary_lines.append(f"   • Average Accuracy: {stats['overall']['Accuracy']['mean']:.3f} (Std: {stats['overall']['Accuracy']['std']:.3f})")
    summary_lines.append(f"   • Average Precision: {stats['overall']['Precision']['mean']:.3f} (Std: {stats['overall']['Precision']['std']:.3f})")
    summary_lines.append(f"   • Average Recall: {stats['overall']['Recall']['mean']:.3f} (Std: {stats['overall']['Recall']['std']:.3f})")
    summary_lines.append("")

    # Grouped Model Statistics
    summary_lines.append("GROUPED MODEL F1 SCORE STATISTICS:")
    summary_lines.append(f"   • Unique Base Models: {len(grouped_models)}")
    summary_lines.append(f"   • Total Test Instances: {grouped_models['Count'].sum()}")
    grouped_models_sorted = grouped_models.sort_values('F1score_Mean', ascending=True)
    summary_lines.append(f"   • Best Performing Model: {grouped_models_sorted.iloc[-1]['Base_Model']} (F1: {grouped_models_sorted.iloc[-1]['F1score_Mean']:.2f})")
    summary_lines.append(f"   • Worst Performing Model: {grouped_models_sorted.iloc[0]['Base_Model']} (F1: {grouped_models_sorted.iloc[0]['F1score_Mean']:.2f})")
    summary_lines.append(f"   • Overall Average F1: {grouped_models['F1score_Mean'].mean():.2f}")
    summary_lines.append(f"   • Models with Vision: {grouped_models['Has_Vision'].sum()}")
    summary_lines.append("")

    return '\n'.join(summary_lines)

def create_summary_page(df, grouped_models, stats):
    """Create summary page using the same text as the comprehensive summary"""
    summary_text = generate_comprehensive_summary_text(df, stats, grouped_models)
    
    fig = plt.figure(figsize=(12, 16))
    plt.text(0.05, 0.95, summary_text, family='monospace', va='top', ha='left', wrap=True, fontsize=10)
    plt.axis('off')
    
    summary_path = save_plot('00_executive_summary')
    return summary_path

def create_combined_pdf():
    """Combine all saved plots into a single PDF report."""
    global saved_plots
    if not saved_plots:
        print("No plots were saved. Cannot create a combined PDF.")
        return None

    # Sort plots by filename to ensure a consistent order
    saved_plots.sort()

    merger = PdfMerger()
    for pdf_path in saved_plots:
        if os.path.exists(pdf_path):
            merger.append(pdf_path)
        else:
            print(f"Warning: Plot file not found at {pdf_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_pdf_path = os.path.join(BASE_DIR, f'comprehensive_llm_analysis_all_plots_{timestamp}.pdf')
    
    try:
        merger.write(combined_pdf_path)
        merger.close()
        return combined_pdf_path
    except Exception as e:
        print(f"Error creating combined PDF: {e}")
        return None

def cleanup_output_directory():
    """Clean up individual plot files after combining them."""
    global saved_plots
    print("Cleaning up temporary plot files...")
    try:
        if os.path.exists(TEMP_PLOT_DIR):
            shutil.rmtree(TEMP_PLOT_DIR)
    except Exception as e:
        print(f"Could not remove {TEMP_PLOT_DIR}: {e}")
    print("Cleanup complete.")

def main():
    """Main function to run the comprehensive analysis."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    df = load_and_prepare_data()
    
    grouped_models = create_overall_performance_plot(df, metric_name='F1score', filename_prefix='01')
    create_overall_performance_plot(df, metric_name='Accuracy', filename_prefix='02')
    create_overall_performance_plot(df, metric_name='Precision', filename_prefix='03')
    create_overall_performance_plot(df, metric_name='Recall', filename_prefix='04')

    create_matrix_plots(df)
    
    df = create_error_analysis_plots(df)
    create_source_f1_comparison(df)
    
    # New plots
    plot_vision_impact(df)
    plot_prompt_comparison_overall(df)
    plot_prompt_type_comparison(df)

    stats = calculate_summary_statistics(df)
    
    create_summary_page(df, grouped_models, stats)
    
    summary_text = generate_comprehensive_summary_text(df, stats, grouped_models)
    print(summary_text)
    
    combined_pdf_path = create_combined_pdf()
    if combined_pdf_path:
        print(f"Successfully created combined PDF: {combined_pdf_path}")
    else:
        print("Could not create combined PDF.")
        
    cleanup_output_directory()

if __name__ == "__main__":
    main()