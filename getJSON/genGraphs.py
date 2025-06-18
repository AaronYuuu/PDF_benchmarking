"""
Comprehensive LLM Benchmarking Analysis Script
==============================================

This script analyzes LLM performance for clinical data extraction from cancer reports.
It provides detailed visualizations and comparisons across multiple dimensions:
- Model families (Qwen, GPT, Gemma, Llama, etc.)
- Input types (text vs vision-enabled)
- Parameter sizes
- Hospital templates
- Source platforms (Ollama vs others)

Author: Generated for PDF_benchmarking project
Date: June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger
import glob
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
output_dir = '../comprehensive_analysis_outputs'
os.makedirs(output_dir, exist_ok=True)

# Global list to track all saved plots for combining
saved_plots = []

def save_plot(filename, fig=None):
    """Save plot with timestamp and track for combining"""
    global saved_plots
    
    if fig is None:
        fig = plt.gcf()
    
    filepath = os.path.join(output_dir, f'{filename}.pdf')
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
    is_vision = '*imageinput*' in llm_lower or 'vision' in llm_lower or 'vl' in llm_lower
    
    # Extract parameter size (look for numbers followed by 'b')
    param_match = re.search(r'(\d+(?:\.\d+)?)b', llm_lower)
    param_size = float(param_match.group(1)) if param_match else None
    
    # Extract model family
    if 'qwen' in llm_lower:
        family = 'Qwen'
    elif 'gpt' in llm_lower:
        family = 'GPT'
    elif 'gemm' in llm_lower or 'gemini' in llm_lower:
        family = 'Gemma/Gemini'
    elif 'llama' in llm_lower:
        family = 'Llama'
    elif 'granite' in llm_lower:
        family = 'Granite'
    elif 'mistral' in llm_lower or 'devstral' in llm_lower:
        family = 'Mistral'
    else:
        family = 'Other'
    
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
    
    return family, param_size, is_vision, base_name, token_category

def overall_metrics(df):
    """
    Create overall F1 score visualization for all models grouped by common base names
    This combines text/vision variants and averages across hospitals
    """
    # Add normalized model names and families if not already present
    if 'Base_Model' not in df.columns:
        df['Base_Model'] = df['LLM'].apply(normalize_model_name)
    
    # Group by base model name and calculate average metrics
    grouped_models = df.groupby('Base_Model').agg({
        'F1score': ['mean', 'std', 'count'],
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'Family': 'first',
        'Source': 'first',
        'Is_Vision': lambda x: any(x)  # True if any variant has vision
    }).round(2)
    
    # Flatten column names
    grouped_models.columns = ['F1_Mean', 'F1_Std', 'Count', 'Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'Family', 'Source', 'Has_Vision']
    grouped_models = grouped_models.reset_index()
    
    # Sort by F1 score for visualization
    grouped_models_sorted = grouped_models.sort_values('F1_Mean', ascending=True)
    
    # Create the grouped bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot 1: Grouped models horizontal bar chart
    bars = ax1.barh(range(len(grouped_models_sorted)), grouped_models_sorted['F1_Mean'])
    
    # Color bars by model family
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    family_colors = {
        'Qwen': colors[0],
        'GPT': colors[1], 
        'Gemma/Gemini': colors[2],
        'Llama': colors[3],
        'Granite': colors[4],
        'Mistral': colors[5],
        'Other': colors[6]
    }
    
    for i, (idx, row) in enumerate(grouped_models_sorted.iterrows()):
        bar_color = family_colors.get(row['Family'], colors[6])
        bars[i].set_color(bar_color)
        
        # Add hatching for models with vision capabilities
        if row['Has_Vision']:
            bars[i].set_hatch('///')
            bars[i].set_edgecolor('black')  # Make hatch more visible
            bars[i].set_linewidth(1.5)     # Thicker edge for better visibility
    
    # Customize the plot
    model_labels = []
    for idx, row in grouped_models_sorted.iterrows():
        vision_indicator = '+VISION' if row['Has_Vision'] else ''
        count_indicator = f' (n={int(row["Count"])})'
        label = f"{row['Base_Model']}{vision_indicator}{count_indicator}"
        model_labels.append(label)
    
    ax1.set_yticks(range(len(grouped_models_sorted)))
    ax1.set_yticklabels(model_labels, fontsize=9)
    ax1.set_xlabel('Average F1 Score', fontsize=12)
    ax1.set_title('Overall F1 Score Performance - Models Grouped by Base Name\n(Averaged across hospitals and input types)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(grouped_models_sorted.iterrows()):
        ax1.text(row['F1_Mean'] + 1, i, f"{row['F1_Mean']:.1f}±{row['F1_Std']:.1f}", 
                va='center', fontsize=8)
    
    # Plot 2: Family performance summary
    family_summary = grouped_models.groupby('Family').agg({
        'F1_Mean': ['mean', 'std', 'count']
    }).round(2)
    family_summary.columns = ['Avg_F1', 'Std_F1', 'Model_Count']
    family_summary = family_summary.reset_index().sort_values('Avg_F1', ascending=False)
    
    bars2 = ax2.bar(family_summary['Family'], family_summary['Avg_F1'], 
                    yerr=family_summary['Std_F1'], capsize=5)
    
    # Color bars by family
    for i, family in enumerate(family_summary['Family']):
        bars2[i].set_color(family_colors.get(family, colors[6]))
    
    # Add count labels on bars
    for bar, count in zip(bars2, family_summary['Model_Count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title('Average F1 Score by Model Family', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model Family', fontsize=12)
    ax2.set_ylabel('Average F1 Score', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Create legend for families with improved vision indicator
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=family) 
                      for family, color in family_colors.items() 
                      if family in grouped_models['Family'].values]
    # Add vision indicator to legend with better styling
    legend_elements.append(Patch(facecolor='lightgray', hatch='///', 
                                edgecolor='black', linewidth=1.5,
                                label='Has Vision Capability'))
    
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    save_plot('00_overall_f1_scores_grouped_models')
    
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
    hospitals[['Family', 'Param_Size', 'Is_Vision', 'Base_Model', 'Token_Category']] = pd.DataFrame(
        model_info.tolist(), index=hospitals.index
    )
    
    # Add source category
    hospitals['Source_Category'] = hospitals['Source'].apply(
        lambda x: 'Ollama' if x.lower() == 'ollama' else 'Commercial'
    )
    
    # Create input type column
    hospitals['Input_Type'] = hospitals['Is_Vision'].map({True: 'Vision', False: 'Text-Only'})
    
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
    
    # By source category
    source_stats = df.groupby('Source_Category').agg({
        'Accuracy': ['mean', 'std', 'count'],
        'F1score': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'False Positives': ['mean', 'sum'],
        'False Negatives': ['mean', 'sum']
    }).round(3)
    
    return {
        'overall': overall_stats,
        'family': family_stats,
        'input_type': input_stats,
        'hospital': hospital_stats,
        'source': source_stats
    }

def create_relational_plots(df):
    """Create relational plots comparing performance across models and token sizes"""
    # Filter data with parameter sizes for meaningful analysis
    df_with_params = df[df['Param_Size'].notna()].copy()
    
    if len(df_with_params) > 0:
        # 1. Accuracy vs Parameter Size by Family
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sns.scatterplot(data=df_with_params, x='Param_Size', y='Accuracy', 
                       hue='Family', size='F1score', sizes=(50, 200), alpha=0.7, ax=axes[0,0])
        axes[0,0].set_title('Accuracy vs Parameter Size by Family', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Parameter Size (B)')
        
        sns.scatterplot(data=df_with_params, x='Param_Size', y='F1score', 
                       hue='Input_Type', style='Hospital', s=100, ax=axes[0,1])
        axes[0,1].set_title('F1 Score vs Parameter Size by Input Type', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Parameter Size (B)')
        
        sns.scatterplot(data=df_with_params, x='Param_Size', y='Precision', 
                       hue='Source_Category', s=100, alpha=0.7, ax=axes[1,0])
        axes[1,0].set_title('Precision vs Parameter Size by Source', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Parameter Size (B)')
        
        sns.scatterplot(data=df_with_params, x='Param_Size', y='Recall', 
                       hue='Family', style='Input_Type', s=100, ax=axes[1,1])
        axes[1,1].set_title('Recall vs Parameter Size by Family and Input Type', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Parameter Size (B)')
        
        plt.tight_layout()
        save_plot('01_relational_plots_parameter_analysis')
        ###plt.show()
    
    # 2. Performance correlation matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line plot showing performance trends by family
    family_means = df.groupby('Family')[['Accuracy', 'F1score', 'Precision', 'Recall']].mean()
    family_means.plot(kind='line', marker='o', ax=axes[0])
    axes[0].set_title('Average Performance Metrics by Model Family', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].legend(title='Metrics')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Scatter plot: Precision vs Recall colored by Accuracy
    sns.scatterplot(data=df, x='Recall', y='Precision', hue='Accuracy', 
                   size='F1score', sizes=(30, 200), ax=axes[1])
    axes[1].set_title('Precision vs Recall (colored by Accuracy)', fontsize=14, fontweight='bold')
    axes[1].plot([0, 100], [100, 0], 'k--', alpha=0.3, label='Trade-off line')
    
    plt.tight_layout()
    save_plot('02_relational_performance_trends')
    ###plt.show()

def create_categorical_plots(df):
    """Create categorical plots showing distributions by input type and family"""
    # Remove the detailed family comparisons subplot since it overlaps with the overall_metrics
    # Keep only the main categorical distribution plots
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    sns.boxplot(data=df, x='Family', y='F1score', hue='Input_Type', ax=axes[0,0])
    axes[0,0].set_title('F1 Score Distribution by Family and Input Type', fontsize=12, fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='Family', y='Accuracy', hue='Hospital', ax=axes[0,1])
    axes[0,1].set_title('Accuracy Distribution by Family and Hospital', fontsize=12, fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    sns.violinplot(data=df, x='Source_Category', y='F1score', hue='Input_Type', 
                   split=True, ax=axes[1,0])
    axes[1,0].set_title('F1 Score Distribution by Source and Input Type', fontsize=12, fontweight='bold')
    
    sns.barplot(data=df, x='Token_Category', y='Accuracy', hue='Family', ax=axes[1,1])
    axes[1,1].set_title('Average Accuracy by Token Size and Family', fontsize=12, fontweight='bold')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot('03_categorical_distributions')
    ###plt.show()

def create_distribution_plots(df):
    """Create distribution plots for accuracy, f1, precision, recall"""
    # 1. Overall distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'F1score', 'Precision', 'Recall']
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        # Histogram with KDE
        sns.histplot(data=df, x=metric, hue='Input_Type', kde=True, 
                    alpha=0.6, ax=axes[row, col])
        axes[row, col].set_title(f'{metric} Distribution by Input Type', fontweight='bold')
    
    plt.tight_layout()
    save_plot('05_metric_distributions_by_input_type')
    ###plt.show()
    
    # 2. Distributions by source category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        sns.kdeplot(data=df, x=metric, hue='Source_Category', 
                   fill=True, alpha=0.6, ax=axes[row, col])
        axes[row, col].set_title(f'{metric} KDE by Source Category', fontweight='bold')
    
    plt.tight_layout()
    save_plot('06_metric_distributions_by_source')
    ###plt.show()
    
    # 3. Error distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # False Positives vs False Negatives
    error_data = df.melt(id_vars=['Hospital', 'Family', 'Input_Type'], 
                        value_vars=['False Positives', 'False Negatives'],
                        var_name='Error_Type', value_name='Error_Count')
    
    sns.boxplot(data=error_data, x='Hospital', y='Error_Count', 
               hue='Error_Type', ax=axes[0])
    axes[0].set_title('Error Distribution by Hospital', fontweight='bold')
    
    sns.violinplot(data=error_data, x='Error_Type', y='Error_Count', 
                  hue='Input_Type', split=True, ax=axes[1])
    axes[1].set_title('Error Distribution by Input Type', fontweight='bold')
    
    plt.tight_layout()
    save_plot('07_error_distributions')
    ##plt.show()

def create_matrix_plots(df):
    """Create matrix plots including heatmaps and correlation matrices"""
    # 1. Correlation heatmap
    numeric_cols = ['Accuracy', 'F1score', 'Precision', 'Recall', 
                   'False Positives', 'False Negatives', 'Incorrect Extractions']
    correlation_matrix = df[numeric_cols].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
               square=True, ax=axes[0])
    axes[0].set_title('Performance Metrics Correlation Matrix', fontweight='bold')
    
    # 2. Performance heatmap by Family and Input Type
    pivot_data = df.groupby(['Family', 'Input_Type'])['F1score'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', fmt='.2f', ax=axes[1])
    axes[1].set_title('Average F1 Score: Family vs Input Type', fontweight='bold')
    
    plt.tight_layout()
    save_plot('08_correlation_and_performance_heatmaps')
    ##plt.show()
    
    # 3. Detailed performance matrix
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Family vs Hospital
    pivot1 = df.groupby(['Family', 'Hospital'])['Accuracy'].mean().unstack()
    sns.heatmap(pivot1, annot=True, cmap='viridis', fmt='.2f', ax=axes[0,0])
    axes[0,0].set_title('Average Accuracy: Family vs Hospital', fontweight='bold')
    
    # Source vs Input Type
    pivot2 = df.groupby(['Source_Category', 'Input_Type'])['Precision'].mean().unstack()
    sns.heatmap(pivot2, annot=True, cmap='plasma', fmt='.2f', ax=axes[0,1])
    axes[0,1].set_title('Average Precision: Source vs Input Type', fontweight='bold')
    
    # Token Category vs Family (where available)
    if df['Token_Category'].nunique() > 1:
        pivot3 = df.groupby(['Token_Category', 'Family'])['Recall'].mean().unstack()
        sns.heatmap(pivot3, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1,0])
        axes[1,0].set_title('Average Recall: Token Size vs Family', fontweight='bold')
    else:
        axes[1,0].text(0.5, 0.5, 'Token size data\nnot available', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Token Size Analysis', fontweight='bold')
    
    # Family vs Source with F1 Score
    pivot4 = df.groupby(['Family', 'Source_Category'])['F1score'].mean().unstack()
    sns.heatmap(pivot4, annot=True, cmap='RdYlGn', fmt='.2f', ax=axes[1,1])
    axes[1,1].set_title('Average F1 Score: Family vs Source', fontweight='bold')
    
    plt.tight_layout()
    save_plot('09_detailed_performance_matrices')
    ##plt.show()

def create_pairgrid_analysis(df):
    """Create PairGrid to explore relationships between all numeric metrics"""
    # Select numeric columns and relevant categorical columns
    numeric_cols = ['Accuracy', 'F1score', 'Precision', 'Recall']
    plot_df = df[numeric_cols + ['Family', 'Input_Type']].copy()
    
    # 1. PairGrid colored by Family
    g1 = sns.PairGrid(plot_df, vars=numeric_cols, hue='Family', height=3)
    g1.map_diag(sns.histplot, alpha=0.7)
    g1.map_upper(sns.scatterplot, alpha=0.7)
    g1.map_lower(sns.regplot, scatter_kws={'alpha': 0.5})
    g1.add_legend(title='Model Family')
    
    plt.suptitle('Performance Metrics Relationships by Model Family', 
                y=1.02, fontsize=16, fontweight='bold')
    save_plot('10_pairgrid_by_family', g1.fig)
    ##plt.show()
    
    # 2. PairGrid colored by Input Type
    g2 = sns.PairGrid(plot_df, vars=numeric_cols, hue='Input_Type', height=3)
    g2.map_diag(sns.histplot, alpha=0.7)
    g2.map_upper(sns.scatterplot, alpha=0.7, s=60)
    g2.map_lower(sns.regplot, scatter_kws={'alpha': 0.5})
    g2.add_legend(title='Input Type')
    
    plt.suptitle('Performance Metrics Relationships by Input Type', 
                y=1.02, fontsize=16, fontweight='bold')
    save_plot('11_pairgrid_by_input_type', g2.fig)
    ##plt.show()

def create_hospital_comparison_analysis(df):
    """Detailed comparison between hospital templates"""
    # 1. Side-by-side comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    metrics = ['Accuracy', 'F1score', 'Precision']
    for i, metric in enumerate(metrics):
        # Box plots
        sns.boxplot(data=df, x='Hospital', y=metric, hue='Family', ax=axes[0,i])
        axes[0,i].set_title(f'{metric} by Hospital and Family', fontweight='bold')
        axes[0,i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Violin plots for detailed distribution
        sns.violinplot(data=df, x='Hospital', y=metric, hue='Input_Type', 
                      split=True, ax=axes[1,i])
        axes[1,i].set_title(f'{metric} Distribution by Hospital and Input Type', fontweight='bold')
    
    plt.tight_layout()
    save_plot('12_hospital_comparison_detailed')
    ##plt.show()
    
    # 2. Error analysis by hospital
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # False Positives comparison
    sns.barplot(data=df, x='Hospital', y='False Positives', hue='Family', ax=axes[0])
    axes[0].set_title('Average False Positives by Hospital and Family', fontweight='bold')
    
    # False Negatives comparison
    sns.barplot(data=df, x='Hospital', y='False Negatives', hue='Family', ax=axes[1])
    axes[1].set_title('Average False Negatives by Hospital and Family', fontweight='bold')
    
    # Total errors
    df['Total_Errors'] = df['False Positives'] + df['False Negatives']
    sns.barplot(data=df, x='Hospital', y='Total_Errors', hue='Input_Type', ax=axes[2])
    axes[2].set_title('Total Errors by Hospital and Input Type', fontweight='bold')
    
    plt.tight_layout()
    save_plot('13_hospital_error_analysis')
    ##plt.show()

def create_source_comparison_analysis(df):
    """Compare Ollama vs Commercial models with token size analysis"""
    # 1. Ollama vs Commercial comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Performance metrics comparison
    metrics = ['Accuracy', 'F1score', 'Precision']
    for i, metric in enumerate(metrics):
        if i < 3:
            row = i // 2
            col = i % 2
            sns.boxplot(data=df, x='Source_Category', y=metric, hue='Input_Type', ax=axes[row, col])
            axes[row, col].set_title(f'{metric} by Source Category and Input Type', fontweight='bold')
    
    # Token size analysis (where available)
    df_with_tokens = df[df['Param_Size'].notna()]
    if len(df_with_tokens) > 0:
        sns.scatterplot(data=df_with_tokens, x='Param_Size', y='F1score', 
                       hue='Source_Category', size='Accuracy', sizes=(50, 200), 
                       alpha=0.7, ax=axes[1,1])
        axes[1,1].set_title('F1 Score vs Parameter Size by Source', fontweight='bold')
        axes[1,1].set_xlabel('Parameter Size (B)')
    else:
        axes[1,1].text(0.5, 0.5, 'Parameter size\ndata not available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Parameter Size Analysis', fontweight='bold')
    
    plt.tight_layout()
    save_plot('14_source_comparison_analysis')
    ##plt.show()
    
    # 2. Detailed source statistics - calculate and format properly
    source_stats = df.groupby(['Source_Category', 'Family']).agg({
        'Accuracy': ['mean', 'std', 'count'],
        'F1score': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'False Positives': 'mean',
        'False Negatives': 'mean'
    }).round(3)
    
    return source_stats

def format_source_stats_for_display(source_stats):
    """Format source statistics for clean display in terminal and PDF"""
    formatted_text = ""
    
    for source_category in source_stats.index.get_level_values(0).unique():
        formatted_text += f"\n{source_category.upper()} MODELS:\n"
        source_data = source_stats.loc[source_category]
        
        for family in source_data.index:
            formatted_text += f"  {family}:\n"
            formatted_text += f"    Accuracy: {source_data.loc[family, ('Accuracy', 'mean')]:.1f}±{source_data.loc[family, ('Accuracy', 'std')]:.1f} (n={int(source_data.loc[family, ('Accuracy', 'count')])})\n"
            formatted_text += f"    F1 Score: {source_data.loc[family, ('F1score', 'mean')]:.1f}±{source_data.loc[family, ('F1score', 'std')]:.1f}\n"
            formatted_text += f"    Precision: {source_data.loc[family, ('Precision', 'mean')]:.1f}±{source_data.loc[family, ('Precision', 'std')]:.1f}\n"
            formatted_text += f"    False Positives: {source_data.loc[family, ('False Positives', 'mean')]:.1f}\n"
            formatted_text += f"    False Negatives: {source_data.loc[family, ('False Negatives', 'mean')]:.1f}\n"
    
    return formatted_text

def generate_comprehensive_summary_text(df, stats, grouped_models, source_stats):
    """Generate the comprehensive summary text that can be used by both print and PDF functions"""
    summary_lines = []
    
    # Title and separator
    summary_lines.append("COMPREHENSIVE LLM BENCHMARKING ANALYSIS - SUMMARY REPORT")
    summary_lines.append("="*80)
    summary_lines.append("")
    
    # Dataset Overview
    summary_lines.append("DATASET OVERVIEW:")
    summary_lines.append(f"   • Total Records: {len(df)}")
    summary_lines.append(f"   • Model Families: {', '.join(df['Family'].unique())}")
    summary_lines.append(f"   • Hospitals: {', '.join(df['Hospital'].unique())}")
    summary_lines.append(f"   • Vision-enabled Models: {df['Is_Vision'].sum()}")
    summary_lines.append(f"   • Text-only Models: {(~df['Is_Vision']).sum()}")
    summary_lines.append(f"   • Ollama Models: {(df['Source_Category'] == 'Ollama').sum()}")
    summary_lines.append(f"   • Commercial Models: {(df['Source_Category'] == 'Commercial').sum()}")
    summary_lines.append("")
    
    # Top Performers
    summary_lines.append("TOP PERFORMERS:")
    top_5 = df.nlargest(5, 'F1score')
    for idx, row in top_5.iterrows():
        vision_text = 'Vision' if row['Is_Vision'] else 'Text-only'
        param_text = f"{row['Param_Size']}B" if pd.notna(row['Param_Size']) else 'Unknown'
        summary_lines.append(f"   • {row['LLM']} ({row['Family']}, {param_text}, {vision_text})")
        summary_lines.append(f"     F1: {row['F1score']:.2f}, Accuracy: {row['Accuracy']:.2f}")
    summary_lines.append("")
    
    # Family Performance Ranking
    summary_lines.append("FAMILY PERFORMANCE RANKING (by F1 Score):")
    family_ranking = df.groupby('Family')['F1score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for family, stats_row in family_ranking.iterrows():
        summary_lines.append(f"   • {family}: {stats_row['mean']:.2f} (n={stats_row['count']})")
    summary_lines.append("")
    
    # Key Insights
    summary_lines.append("KEY INSIGHTS:")
    vision_perf = df.groupby('Is_Vision')['F1score'].mean()
    summary_lines.append(f"   • Vision Models Avg F1: {vision_perf[True]:.2f}")
    summary_lines.append(f"   • Text-only Models Avg F1: {vision_perf[False]:.2f}")
    
    hospital_perf = df.groupby('Hospital')['F1score'].mean()
    summary_lines.append(f"   • Hospital 1 Avg F1: {hospital_perf['hospital1']:.2f}")
    summary_lines.append(f"   • Hospital 2 Avg F1: {hospital_perf['hospital2']:.2f}")
    
    source_perf = df.groupby('Source_Category')['F1score'].mean()
    summary_lines.append(f"   • Ollama Models Avg F1: {source_perf['Ollama']:.2f}")
    summary_lines.append(f"   • Commercial Models Avg F1: {source_perf['Commercial']:.2f}")
    summary_lines.append("")
    
    # Grouped Model Statistics
    summary_lines.append("Grouped Model F1 Score Statistics:")
    summary_lines.append(f"   • Unique Base Models: {len(grouped_models)}")
    summary_lines.append(f"   • Total Test Instances: {grouped_models['Count'].sum()}")
    grouped_models_sorted = grouped_models.sort_values('F1_Mean', ascending=True)
    summary_lines.append(f"   • Best Performing Model: {grouped_models_sorted.iloc[-1]['Base_Model']} (F1: {grouped_models_sorted.iloc[-1]['F1_Mean']:.2f})")
    summary_lines.append(f"   • Worst Performing Model: {grouped_models_sorted.iloc[0]['Base_Model']} (F1: {grouped_models_sorted.iloc[0]['F1_Mean']:.2f})")
    summary_lines.append(f"   • Overall Average F1: {grouped_models['F1_Mean'].mean():.2f}")
    summary_lines.append(f"   • Models with Vision: {grouped_models['Has_Vision'].sum()}")
    summary_lines.append("")
    
    # Top 5 and Bottom 5 Performers
    summary_lines.append("Top 5 Performers:")
    for idx, row in grouped_models_sorted.tail(5).iterrows():
        vision_text = 'with Vision' if row['Has_Vision'] else 'Text-only'
        summary_lines.append(f"   • {row['Base_Model']} ({row['Family']}, {vision_text}): F1 = {row['F1_Mean']:.2f} ± {row['F1_Std']:.2f}")
    summary_lines.append("")
    
    summary_lines.append("Bottom 5 Performers:")
    for idx, row in grouped_models_sorted.head(5).iterrows():
        vision_text = 'with Vision' if row['Has_Vision'] else 'Text-only'
        summary_lines.append(f"   • {row['Base_Model']} ({row['Family']}, {vision_text}): F1 = {row['F1_Mean']:.2f} ± {row['F1_Std']:.2f}")
    summary_lines.append("")
    
    # Error Analysis Summary
    summary_lines.append("Error Analysis Summary:")
    summary_lines.append(f"   • Average False Positives: {df['False Positives'].mean():.2f}")
    summary_lines.append(f"   • Average False Negatives: {df['False Negatives'].mean():.2f}")
    summary_lines.append(f"   • Models with more FP than FN: {(df['False Positives'] > df['False Negatives']).sum()}")
    summary_lines.append(f"   • Models with more FN than FP: {(df['False Negatives'] > df['False Positives']).sum()}")
    summary_lines.append("")
    
    # Source Category Statistics
    summary_lines.append("SOURCE CATEGORY DETAILED STATISTICS:")
    formatted_source_stats = format_source_stats_for_display(source_stats)
    summary_lines.extend(formatted_source_stats.split('\n'))
    summary_lines.append("")
    
    # Final separator
    summary_lines.append("="*80)
    
    return '\n'.join(summary_lines)

def create_summary_page(df, grouped_models, stats, source_stats):
    """Create executive summary page using the same text as the comprehensive summary"""
    # Generate the summary text using the centralized function
    summary_text = generate_comprehensive_summary_text(df, stats, grouped_models, source_stats)
    
    # Calculate text dimensions for optimal figure sizing
    lines = summary_text.split('\n')
    num_lines = len(lines)
    max_line_length = max(len(line) for line in lines)
    
    # Adjust figure size based on content
    fig_width = max_line_length * 0.06 + 1
    fig_height = num_lines * 0.1 + 1
    
    # Ensure reasonable bounds
    fig_width = max(min(fig_width, 12), 6)
    fig_height = max(min(fig_height, 16), 8)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Calculate font size based on figure dimensions
    font_size = max(min(fig_width * 0.7, 9), 6)
    
    # Display the text
    ax.text(0.01, 0.99, summary_text, 
           ha='left', va='top', fontsize=font_size, transform=ax.transAxes,
           fontfamily='sans-serif', fontname='Arial')
    
    ax.set_title('LLM Benchmarking Analysis - Executive Summary', 
                fontsize=font_size + 1, fontweight='normal', pad=10,
                fontfamily='sans-serif', fontname='Arial')
    ax.axis('off')
    
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    
    summary_path = save_plot('00_executive_summary')
    return summary_path

def create_normalized_accuracy_distribution(df):
    """Create normalized accuracy distribution plot"""
    # 1. Normalized accuracy by family
    # Calculate family-wise normalized scores (z-score normalization)
    df_norm = df.copy()
    family_stats = df.groupby('Family')['Accuracy'].agg(['mean', 'std'])
    
    for family in df['Family'].unique():
        family_mask = df['Family'] == family
        family_mean = family_stats.loc[family, 'mean']
        family_std = family_stats.loc[family, 'std']
        if family_std > 0:  # Avoid division by zero
            df_norm.loc[family_mask, 'Accuracy_Normalized'] = (
                (df.loc[family_mask, 'Accuracy'] - family_mean) / family_std
            )
        else:
            df_norm.loc[family_mask, 'Accuracy_Normalized'] = 0
    
    # Plot normalized distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.boxplot(data=df_norm, x='Family', y='Accuracy_Normalized', ax=axes[0,0])
    axes[0,0].set_title('Normalized Accuracy Distribution by Family\n(Z-score normalized within family)', 
                       fontsize=12, fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Family Mean')
    axes[0,0].legend()
    
    sns.violinplot(data=df_norm, x='Input_Type', y='Accuracy_Normalized', ax=axes[0,1])
    axes[0,1].set_title('Normalized Accuracy Distribution by Input Type', 
                       fontsize=12, fontweight='bold')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    sns.histplot(data=df_norm, x='Accuracy_Normalized', hue='Hospital', 
                kde=True, alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('Overall Normalized Accuracy Distribution by Hospital', 
                       fontsize=12, fontweight='bold')
    axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Overall Mean')
    axes[1,0].legend()
    
    sns.scatterplot(data=df_norm, x='Accuracy', y='Accuracy_Normalized', 
                   hue='Family', style='Input_Type', s=80, alpha=0.7, ax=axes[1,1])
    axes[1,1].set_title('Raw vs Normalized Accuracy', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Raw Accuracy')
    axes[1,1].set_ylabel('Normalized Accuracy (Z-score)')
    
    plt.tight_layout()
    save_plot('01_accuracy_distribution_normalized')
    
    return df_norm

def create_normalized_f1_hospital_comparison(df):
    """Create normalized F1 score hospital comparison"""
    # Calculate hospital-wise normalized F1 scores
    df_norm = df.copy()
    hospital_stats = df.groupby('Hospital')['F1score'].agg(['mean', 'std'])
    
    for hospital in df['Hospital'].unique():
        hospital_mask = df['Hospital'] == hospital
        hospital_mean = hospital_stats.loc[hospital, 'mean']
        hospital_std = hospital_stats.loc[hospital, 'std']
        if hospital_std > 0:
            df_norm.loc[hospital_mask, 'F1score_Normalized'] = (
                (df.loc[hospital_mask, 'F1score'] - hospital_mean) / hospital_std
            )
        else:
            df_norm.loc[hospital_mask, 'F1score_Normalized'] = 0
    
    # 1. Normalized F1 by family and hospital
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.boxplot(data=df_norm, x='Family', y='F1score_Normalized', hue='Hospital', ax=axes[0,0])
    axes[0,0].set_title('Normalized F1 Score by Family and Hospital\n(Z-score normalized within hospital)', 
                       fontsize=12, fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Hospital Mean')
    axes[0,0].legend()
    
    sns.violinplot(data=df_norm, x='Hospital', y='F1score_Normalized', 
                  hue='Input_Type', split=True, ax=axes[0,1])
    axes[0,1].set_title('Normalized F1 Score Distribution\nby Hospital and Input Type', 
                       fontsize=12, fontweight='bold')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. Side-by-side comparison of raw vs normalized
    hospital_comparison = df.groupby(['Hospital', 'Family']).agg({
        'F1score': 'mean'
    }).reset_index()
    hospital_comparison_norm = df_norm.groupby(['Hospital', 'Family']).agg({
        'F1score_Normalized': 'mean'
    }).reset_index()
    
    sns.barplot(data=hospital_comparison, x='Family', y='F1score', hue='Hospital', ax=axes[1,0])
    axes[1,0].set_title('Raw F1 Scores by Family and Hospital', fontsize=12, fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=hospital_comparison_norm, x='Family', y='F1score_Normalized', 
               hue='Hospital', ax=axes[1,1])
    axes[1,1].set_title('Normalized F1 Scores by Family and Hospital', fontsize=12, fontweight='bold')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_plot('02_f1_score_hospital_comparison_normalized')
    
    return df_norm

def create_false_positives_negatives_comparison(df):
    """Create comprehensive false positives vs false negatives comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot: False Positives vs False Negatives
    sns.scatterplot(data=df, x='False Positives', y='False Negatives', 
                   hue='Family', style='Hospital', s=100, alpha=0.7, ax=axes[0,0])
    axes[0,0].set_title('False Positives vs False Negatives by Family and Hospital', 
                       fontsize=12, fontweight='bold')
    axes[0,0].plot([0, df[['False Positives', 'False Negatives']].max().max()], 
                   [0, df[['False Positives', 'False Negatives']].max().max()], 
                   'k--', alpha=0.5, label='Equal Errors Line')
    axes[0,0].legend()
    
    # 2. Error type distribution by family
    error_data = df.melt(id_vars=['Family', 'Hospital', 'Input_Type'], 
                        value_vars=['False Positives', 'False Negatives'],
                        var_name='Error_Type', value_name='Error_Count')
    
    sns.boxplot(data=error_data, x='Family', y='Error_Count', hue='Error_Type', ax=axes[0,1])
    axes[0,1].set_title('Error Distribution by Model Family', fontsize=12, fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Error ratio analysis
    df['Error_Ratio'] = df['False Positives'] / (df['False Negatives'] + 1e-6)  # Add small epsilon to avoid division by zero
    df['Total_Errors'] = df['False Positives'] + df['False Negatives']
    
    sns.scatterplot(data=df, x='Total_Errors', y='Error_Ratio', 
                   hue='Input_Type', style='Hospital', s=100, alpha=0.7, ax=axes[1,0])
    axes[1,0].set_title('Error Ratio vs Total Errors\n(Ratio = FP/FN)', fontsize=12, fontweight='bold')
    axes[1,0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal FP/FN')
    axes[1,0].set_ylabel('False Positive / False Negative Ratio')
    axes[1,0].legend()
    # Only set log scale if there are positive values
    if df['Error_Ratio'].min() > 0:
        axes[1,0].set_yscale('log')
    
    # 4. Heatmap of average errors by family and hospital
    error_pivot = df.groupby(['Family', 'Hospital'])[['False Positives', 'False Negatives']].mean()
    error_pivot['FP_minus_FN'] = error_pivot['False Positives'] - error_pivot['False Negatives']
    error_heatmap = error_pivot['FP_minus_FN'].unstack()
    
    sns.heatmap(error_heatmap, annot=True, cmap='RdBu_r', center=0, 
               fmt='.1f', ax=axes[1,1])
    axes[1,1].set_title('Error Bias: FP - FN by Family and Hospital\n(+ve = More FP, -ve = More FN)', 
                       fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_plot('03_false_positives_negatives_comparison')
    
    return df

def cleanup_output_directory():
    """Delete the comprehensive_analysis_outputs directory after combining PDFs"""
    import shutil
    
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    except Exception as e:
        pass

def create_combined_pdf():
    """Combine all individual plots into a single comprehensive PDF"""
    if not saved_plots:
        return None
    
    # Sort plots by filename for logical ordering
    sorted_plots = sorted(saved_plots)
    
    # Create combined PDF filename with timestamp - save in main directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = f"comprehensive_llm_analysis_all_plots_{timestamp}.pdf"
    # Save in the main PDF_benchmarking directory instead of output subfolder
    main_dir = os.path.dirname(output_dir)
    combined_filepath = os.path.join(main_dir, combined_filename)
    
    try:
        # Use PdfMerger to combine all PDFs
        merger = PdfMerger()
        
        for pdf_path in sorted_plots:
            if os.path.exists(pdf_path):
                merger.append(pdf_path)
        
        # Write the combined PDF
        merger.write(combined_filepath)
        merger.close()
        
        return combined_filepath
        
    except Exception as e:
        return None

def main():
    """Main execution function"""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Calculate summary statistics
    stats = calculate_summary_statistics(df)
    
    # Create all visualizations - starting with the grouped overview
    grouped_models = overall_metrics(df)
    
    create_normalized_accuracy_distribution(df)
    create_normalized_f1_hospital_comparison(df)
    create_false_positives_negatives_comparison(df)
    
    create_relational_plots(df)
    create_categorical_plots(df)
    create_distribution_plots(df)
    create_matrix_plots(df)
    create_pairgrid_analysis(df)
    create_hospital_comparison_analysis(df)
    source_stats = create_source_comparison_analysis(df)
    create_summary_page(df, grouped_models, stats, source_stats)
    
    # Print comprehensive summary
    
    # Combine all plots into a single PDF
    combined_pdf_path = create_combined_pdf()
    
    # Clean up the temporary output directory after successful PDF creation
    if combined_pdf_path:
        cleanup_output_directory()
    
    if combined_pdf_path:
        print(f"Final combined PDF saved to: {combined_pdf_path}")

if __name__ == "__main__":
    main()