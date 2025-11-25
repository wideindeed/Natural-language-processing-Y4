import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import logger

def plot_sentiment_performance(results, languages):
    accuracies = [results[lang]['sentiment_analysis']['accuracy'] for lang in languages]
    f1_scores = [results[lang]['sentiment_analysis']['f1_score'] for lang in languages]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(languages))
    width = 0.35
    
    ax.bar(x - width / 2, accuracies, width, label='Accuracy', color='#3b82f6')
    ax.bar(x + width / 2, f1_scores, width, label='F1-Score', color='#10b981')
    
    ax.set_xlabel('Language')
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Analysis Performance by Language')
    ax.set_xticks(x)
    ax.set_xticklabels([lang.title() for lang in languages])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
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