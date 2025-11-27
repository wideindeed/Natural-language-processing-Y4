import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import logger

def plot_sentiment_performance(results, languages):
    
    for lang in languages:
        data = results[lang]['sentiment_analysis']
        if 'transformer' not in data:
            continue 
            
        labels = ['Accuracy', 'F1-Score']
        baseline_scores = [data['baseline']['accuracy'], data['baseline']['f1_score']]
        transformer_scores = [data['transformer']['accuracy'], data['transformer']['f1_score']]
        
        model_name = "BERT" if lang == 'english' else "AraBERT"
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (Ensemble)', color='#3b82f6')
        rects2 = ax.bar(x + width/2, transformer_scores, width, label=f'{model_name} (Deep Learning)', color='#10b981')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{lang.title()} Model Comparison: Baseline vs {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.show()

def plot_perplexity_comparison(results):
    english_lm = results['english']['language_model']
    arabic_lm = results['arabic']['language_model']
    ngrams = list(english_lm.keys())
    
    en_perplexities = [english_lm[ng]['perplexity'] for ng in ngrams]
    ar_perplexities = [arabic_lm[ng]['perplexity'] for ng in ngrams]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ngrams, en_perplexities, marker='o', linewidth=2, label='English', color='#3b82f6')
    ax.plot(ngrams, ar_perplexities, marker='s', linewidth=2, label='Arabic', color='#10b981')
    
    ax.set_xlabel('N-gram Model')
    ax.set_ylabel('Perplexity')
    ax.set_title('Language Model Perplexity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_pos_frequencies(results):
    fig_pos, axes_pos = plt.subplots(1, 2, figsize=(14, 6))
    fig_pos.suptitle('Top 10 Most Common POS Tags by Language (from Test Set)', fontsize=16)

    try:
        en_tags = results['english']['pos_tagging']['top_10_tags']
        if en_tags:
            labels, counts = zip(*en_tags)
            labels = list(labels)
            counts = list(counts)
            ticks = np.arange(len(labels))

            sns.scatterplot(
                x=counts,
                y=ticks,
                ax=axes_pos[0],
                color='#3b82f6',
                size=counts,
                sizes=(100, 1000),
                legend=False
            )

            axes_pos[0].set_yticks(ticks)
            axes_pos[0].set_yticklabels(labels)
            axes_pos[0].set_title('English POS Tag Frequencies (Scatter Plot)')
            axes_pos[0].set_xlabel('Total Count in Test Set')
            axes_pos[0].grid(True, alpha=0.3)
        else:
            axes_pos[0].text(0.5, 0.5, 'No English POS tags found.', horizontalalignment='center',
                             verticalalignment='center', transform=axes_pos[0].transAxes)
    except Exception as e:
        logger.error(f"Could not plot English POS tags: {e}")
        axes_pos[0].text(0.5, 0.5, 'Error plotting English tags.', horizontalalignment='center', verticalalignment='center',
                         transform=axes_pos[0].transAxes)

    try:
        ar_tags = results['arabic']['pos_tagging']['top_10_tags']
        if ar_tags and ar_tags[0][0] != 'UNK':
            labels, counts = zip(*ar_tags)
            labels = list(labels)
            counts = list(counts)

            plt.rcParams['font.family'] = ['Arial', 'sans-serif']
            ticks = np.arange(len(labels))

            sns.scatterplot(
                x=counts,
                y=ticks,
                ax=axes_pos[1],
                color='#10b981',
                size=counts,
                sizes=(100, 1000),
                legend=False
            )

            axes_pos[1].set_yticks(ticks)
            axes_pos[1].set_yticklabels(labels)
            axes_pos[1].set_title('Arabic POS Tag Frequencies (Scatter Plot)')
            axes_pos[1].set_xlabel('Total Count in Test Set')
            axes_pos[1].grid(True, alpha=0.3)
        else:
            axes_pos[1].text(0.5, 0.5, 'Arabic POS Tagger not run or only found "UNK" tags.', horizontalalignment='center',
                             verticalalignment='center', transform=axes_pos[1].transAxes)
    except Exception as e:
        logger.error(f"Could not plot Arabic POS tags: {e}")
        axes_pos[1].text(0.5, 0.5, 'Error plotting Arabic tags.', horizontalalignment='center', verticalalignment='center',
                         transform=axes_pos[1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()