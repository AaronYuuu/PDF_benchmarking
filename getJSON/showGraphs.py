#new file with everything to show the nessacary graphs and data
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def promptCompare(df):
    df = df.groupby(['Prompt']).F1score.mean().sort_values(ascending=False)
    graph = df.plot(kind='bar', figsize=(12, 6), color='skyblue')
    plt.title('Average F1 Score by LLM')
    plt.xlabel('LLM')
    plt.ylabel('Average F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return graph

def main():
    data = pd.read_csv('Hospital.csv')
    data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()], inplace=True)
    prompts = promptCompare(data)