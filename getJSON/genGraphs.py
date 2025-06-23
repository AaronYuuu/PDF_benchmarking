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
    Image_Input = '*imageinput*' in llm_lower or 'vision' in llm_lower or 'vl' in llm_lower
    
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
    else:
        family = 'NuExtract'
    
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
    
    return family, param_size, Image_Input, base_name, token_category

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
        'Image_Input': lambda x: any(x)  # True if any variantvision
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
        'NuExtract': colors[6]
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
        vision_indicator = ' (Image Input)' if row['Has_Vision'] else ''
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
                                label='Given Image Input'))
    
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    save_plot('00_overall_f1_scores_grouped_models')
    
    return grouped_models

def load_and_prepare_data():
    """Load and prepare the hospital data with comprehensive feature engineering"""
    # Load the data
    hospitals = pd.read_csv('Hospital.csv')
    hospitals = hospitals[hospitals['Source'] != "OpenRouter"]
    # Clean unnecessary columns
    cols_to_drop = [col for col in hospitals.columns if 'Unnamed' in col]
    hospitals = hospitals.drop(columns=cols_to_drop, errors='ignore')
    
    # Extract model information
    model_info = hospitals['LLM'].apply(extract_model_info)
    hospitals[['Family', 'Param_Size', 'Image_Input', 'Base_Model', 'Token_Category']] = pd.DataFrame(
        model_info.tolist(), index=hospitals.index
    )
    
    # Add source category
    hospitals['Source_Category'] = hospitals['Source'].apply(
        lambda x: 'Ollama' if x.lower() == 'ollama' else 'Commercial'
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
        # 1. F1 Score and Accuracy vs Parameter Size
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # F1 Score vs Parameter Size by Family
        sns.scatterplot(data=df_with_params, x='Param_Size', y='F1score', 
                       hue='Family', style='Input_Type', s=100, alpha=0.8, ax=axes[0])
        axes[0].set_title('F1 Score vs Parameter Size by Family and Input Type', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Parameter Size (B)')
        axes[0].set_ylabel('F1 Score')
        axes[0].grid(alpha=0.3)
        
        # Accuracy vs Parameter Size by Family
        sns.scatterplot(data=df_with_params, x='Param_Size', y='Accuracy', 
                       hue='Family', style='Input_Type', s=100, alpha=0.8, ax=axes[1])
        axes[1].set_title('Accuracy vs Parameter Size by Family and Input Type', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Parameter Size (B)')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_plot('01_parameter_size_performance_analysis')
    
    # 2. Precision vs Recall Analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Precision vs Recall colored by Accuracy, with Family as hue and Input_Type as style
    sns.scatterplot(data=df, x='Recall', y='Precision', hue='Family', 
                   style='Input_Type', size='Accuracy', sizes=(50, 200), 
                   alpha=0.8, ax=ax)
    ax.set_title('Precision vs Recall by Family and Input Type\n(Size = Accuracy, Color = Family)', 
                fontsize=14, fontweight='bold')
    ax.plot([0, 100], [100, 0], 'k--', alpha=0.3, label='Trade-off line')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot('02_precision_recall_analysis')

def create_categorical_plots(df):
    """Create key categorical performance comparisons"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # F1 Score by Family and Input Type
    sns.boxplot(data=df, x='Family', y='F1score', hue='Input_Type', ax=axes[0,0])
    axes[0,0].set_title('F1 Score Distribution by Family and Input Type', fontsize=12, fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Accuracy by Family and Hospital
    sns.boxplot(data=df, x='Family', y='Accuracy', hue='Hospital', ax=axes[0,1])
    axes[0,1].set_title('Accuracy Distribution by Family and Hospital', fontsize=12, fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Average Performance Metrics by Family
    family_metrics = df.groupby('Family')[['Accuracy', 'F1score', 'Precision', 'Recall']].mean()
    family_metrics.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average Performance Metrics by Family', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(title='Metrics')
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Performance by Input Type (Vision vs Text-only)
    input_metrics = df.groupby('Input_Type')[['Accuracy', 'F1score', 'Precision', 'Recall']].mean()
    input_metrics.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Average Performance Metrics by Input Type', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Score')
    axes[1,1].tick_params(axis='x', rotation=0)
    axes[1,1].legend(title='Metrics')
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot('03_categorical_performance_analysis')

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
    """Create comprehensive error analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. False Positives vs False Negatives scatter
    sns.scatterplot(data=df, x='False Positives', y='False Negatives', 
                   hue='Family', style='Input_Type', s=100, alpha=0.8, ax=axes[0,0])
    axes[0,0].set_title('False Positives vs False Negatives by Family and Input Type', 
                       fontsize=12, fontweight='bold')
    max_error = df[['False Positives', 'False Negatives']].max().max()
    axes[0,0].plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='Equal Errors Line')
    axes[0,0].grid(alpha=0.3)
    axes[0,0].legend()
    
    # 2. Error distribution by family
    error_data = df.melt(id_vars=['Family', 'Hospital', 'Input_Type'], 
                        value_vars=['False Positives', 'False Negatives'],
                        var_name='Error_Type', value_name='Error_Count')
    
    sns.boxplot(data=error_data, x='Family', y='Error_Count', hue='Error_Type', ax=axes[0,1])
    axes[0,1].set_title('Error Distribution by Model Family', fontsize=12, fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # 3. Error ratio analysis
    df['Error_Ratio'] = df['False Positives'] / (df['False Negatives'] + 1e-6)
    df['Total_Errors'] = df['False Positives'] + df['False Negatives']
    
    sns.scatterplot(data=df, x='Total_Errors', y='Error_Ratio', 
                   hue='Family', style='Input_Type', s=100, alpha=0.8, ax=axes[1,0])
    axes[1,0].set_title('Error Ratio vs Total Errors\n(Ratio = FP/FN)', fontsize=12, fontweight='bold')
    axes[1,0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal FP/FN')
    axes[1,0].set_ylabel('False Positive / False Negative Ratio')
    axes[1,0].grid(alpha=0.3)
    axes[1,0].legend()
    
    # 4. Error bias heatmap
    error_pivot = df.groupby(['Family', 'Hospital'])[['False Positives', 'False Negatives']].mean()
    error_pivot['FP_minus_FN'] = error_pivot['False Positives'] - error_pivot['False Negatives']
    error_heatmap = error_pivot['FP_minus_FN'].unstack()
    
    sns.heatmap(error_heatmap, annot=True, cmap='RdBu_r', center=0, 
               fmt='.1f', ax=axes[1,1])
    axes[1,1].set_title('Error Bias: FP - FN by Family and Hospital\n(+ve = More FP, -ve = More FN)', 
                       fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_plot('05_error_analysis_comprehensive')
    
    return df

def create_hospital_comparison_analysis(df):
    """Simplified hospital comparison focusing on key metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # F1 Score comparison by hospital and family
    sns.boxplot(data=df, x='Hospital', y='F1score', hue='Family', ax=axes[0])
    axes[0].set_title('F1 Score Distribution by Hospital and Family', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Accuracy comparison by hospital and input type
    sns.boxplot(data=df, x='Hospital', y='Accuracy', hue='Input_Type', ax=axes[1])
    axes[1].set_title('Accuracy Distribution by Hospital and Input Type', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot('06_hospital_comparison_analysis')

def create_source_comparison_analysis(df):
    """Compare Ollama vs Commercial models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance metrics comparison by source
    source_metrics = df.groupby('Source_Category')[['Accuracy', 'F1score', 'Precision', 'Recall']].mean()
    source_metrics.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Average Performance Metrics by Source Category', fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(title='Metrics')
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1 Score distribution by source and input type
    sns.boxplot(data=df, x='Source_Category', y='F1score', hue='Input_Type', ax=axes[1])
    axes[1].set_title('F1 Score Distribution by Source and Input Type', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot('07_source_comparison_analysis')
    
    # Calculate detailed source statistics
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
    summary_lines.append(f"   • Vision-enabled Models: {df['Image_Input'].sum()}")
    summary_lines.append(f"   • Text-only Models: {(~df['Image_Input']).sum()}")
    summary_lines.append(f"   • Ollama Models: {(df['Source_Category'] == 'Ollama').sum()}")
    summary_lines.append(f"   • Commercial Models: {(df['Source_Category'] == 'Commercial').sum()}")
    summary_lines.append("")
    
    # Top Performers
    summary_lines.append("TOP PERFORMERS:")
    top_5 = df.nlargest(5, 'F1score')
    for idx, row in top_5.iterrows():
        vision_text = 'Vision' if row['Image_Input'] else 'Text-only'
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
    vision_perf = df.groupby('Image_Input')['F1score'].mean()
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
    """Create summary page using the same text as the comprehensive summary"""
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

def cleanup_output_directory():
    """Clean up the temporary output directory after PDF creation"""
    try:
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleaned up temporary directory: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")

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
    # Load and prepare data
    df = load_and_prepare_data() #function added ignore openRouter models, not enough data
    # Calculate summary statistics
    stats = calculate_summary_statistics(df)
    
    # Create key visualizations in logical order
    grouped_models = overall_metrics(df)  # Keep this as the main overview
    
    create_relational_plots(df)           # Parameter size vs performance
    create_categorical_plots(df)          # Family and input type comparisons
    create_matrix_plots(df)              # Performance heatmaps
    create_error_analysis_plots(df)      # Error analysis
    create_hospital_comparison_analysis(df)  # Hospital comparisons
    source_stats = create_source_comparison_analysis(df)  # Source comparisons
    create_summary_page(df, grouped_models, stats, source_stats)  # Executive summary
    
    # Print comprehensive summary to console
    summary_text = generate_comprehensive_summary_text(df, stats, grouped_models, source_stats)
    print(summary_text)
    
    # Combine all plots into a single PDF
    combined_pdf_path = create_combined_pdf()
    
    # Clean up the temporary output directory after successful PDF creation
    if combined_pdf_path:
        cleanup_output_directory()
        print(f"\nFinal combined PDF saved to: {combined_pdf_path}")
    else:
        print("\nError creating combined PDF")

if __name__ == "__main__":
    main()