import json
import re
import numpy as np
from datetime import datetime
from collections import Counter

from config import logger, OUTPUT_FILE
from data_loader import MultilingualDataLoader
from preprocessor import MultilingualPreprocessor
from models import NGramLanguageModel, SentimentClassifier
from nlp_tools import BaselineParser, NamedEntityRecognizer
from visualization import plot_sentiment_performance, plot_perplexity_comparison, plot_pos_frequencies
from models import NGramLanguageModel, SentimentClassifier, TransformerClassifier

def run_complete_pipeline():
    logger.info("MULTILINGUAL MOVIE REVIEWS NLP PIPELINE")
    
    data_loader = MultilingualDataLoader()
    preprocessor = MultilingualPreprocessor()
    
    results = {}

    for language in ['english', 'arabic']:
        logger.info(f"\nProcessing {language.upper()} reviews")
        results[language] = {}

        df = data_loader.load_imdb_dataset(language=language, n_samples=1200)
        if df.empty:
            logger.error(f"Failed to load data for {language}. Skipping.")
            continue
            
        train_df, dev_df, test_df = data_loader.create_train_test_split(df)

        logger.info("Preprocessing data")
        train_texts = train_df['text'].tolist()
        train_labels = train_df['sentiment'].tolist()

        dev_texts = dev_df['text'].tolist()
        dev_labels = dev_df['sentiment'].tolist()

        test_texts = test_df['text'].tolist()
        test_labels = test_df['sentiment'].tolist()

        logger.info("Performing POS Tagging and Parsing")
        parser = BaselineParser(language=language)
        sample_text = train_texts[0]
        sample_tokens_full = preprocessor.tokenize(
            preprocessor.clean_text(sample_text, language), language
        )
        sample_pos_tags = preprocessor.pos_tag(sample_tokens_full, language)
        chunk_tree = parser.chunk(sample_pos_tags)

        all_tags = []
        test_tokens_list = [
            preprocessor.tokenize(preprocessor.clean_text(text, language), language)
            for text in test_texts
        ]
        for tokens in test_tokens_list:
            if tokens:
                tags = preprocessor.pos_tag(tokens, language)
                all_tags.extend([tag for word, tag in tags])
        tag_counts = Counter(all_tags)
        top_10_tags = tag_counts.most_common(10)

        results[language]['pos_tagging'] = {
            'sample_tags': sample_pos_tags[:10],
            'top_10_tags': top_10_tags
        }
        results[language]['parsing'] = {'sample_tree_str': str(chunk_tree)}

        logger.info("Training N-Gram language models")
        lm_results = {}
        def normalize_and_tokenize(text, language):
            cleaned = preprocessor.clean_text(text, language)
            tokens = preprocessor.tokenize(cleaned, language)
            if language == 'arabic':
                tokens = [t for t in tokens if re.match(r'^[\u0621-\u064A]+$', t)]
            return tokens

        for n in [2, 3, 4]:
            lm = NGramLanguageModel(n=n, smoothing='kneser_ney')
            train_corpus = [normalize_and_tokenize(text, language) for text in train_texts[:200]]
            test_corpus = [normalize_and_tokenize(text, language) for text in test_texts[:50]]

            lm.train(train_corpus)
            perplexity = lm.calculate_perplexity(test_corpus)
            lm_results[f'{n}-gram'] = {'perplexity': perplexity, 'vocab_size': lm.vocab_size}
            logger.info(f"{n}-gram perplexity: {perplexity:.2f}")

        logger.info("Training BASELINE sentiment classifier (Ensemble)")
        sentiment_clf = SentimentClassifier(language=language)
        sentiment_clf.train(train_texts, train_labels)
        baseline_metrics = sentiment_clf.evaluate(test_texts, test_labels)
        logger.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")

        logger.info(f"Training DEEP LEARNING model ({language})")
        if language == 'english':
            model_name = "bert-base-uncased"
        else:
            model_name = "aubmindlab/bert-base-arabertv2"
        
        dl_classifier = TransformerClassifier(model_name=model_name)
        dl_classifier.train(train_texts, train_labels, val_texts=dev_texts, val_labels=dev_labels)
        dl_metrics = dl_classifier.evaluate(test_texts, test_labels)
        logger.info(f"Deep Learning Accuracy: {dl_metrics['accuracy']:.4f}")

        logger.info("Performing Named Entity Recognition")
        ner = NamedEntityRecognizer(language=language)
        sample_text_ner = test_texts[0]
        sample_tokens_ner = preprocessor.tokenize(sample_text_ner, language)
        entities = ner.extract_entities(sample_text_ner, sample_tokens_ner)

        results[language]['language_model'] = lm_results
        
        results[language]['sentiment_analysis'] = {
            'baseline': {
                'accuracy': baseline_metrics['accuracy'],
                'precision': baseline_metrics['precision'],
                'recall': baseline_metrics['recall'],
                'f1_score': baseline_metrics['f1_score']
            },
            'transformer': {
                'accuracy': dl_metrics['accuracy'],
                'f1_score': dl_metrics['f1_score']
            }
        }
        results[language]['ner'] = {'sample_entities': len(entities)}

    logger.info("\nFINAL RESULTS SUMMARY")
    for lang, metrics in results.items():
        logger.info(f"\n{lang.upper()}:")
        logger.info(f"  Best Perplexity: {min(m['perplexity'] for m in metrics['language_model'].values()):.2f}")
        logger.info(f"  Baseline Acc:    {metrics['sentiment_analysis']['baseline']['accuracy']:.4f}")
        logger.info(f"  Transformer Acc: {metrics['sentiment_analysis']['transformer']['accuracy']:.4f}")

    save_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to '{OUTPUT_FILE}'")

    return results

def analyze_review_demo(text, language='english'):
    logger.info(f"\n--- DEMO: Analyzing {language.upper()} Review ---")
    logger.info(f"Original Review: {text}")

    preprocessor = MultilingualPreprocessor()
    processed = preprocessor.preprocess_pipeline(text, language, do_pos_tag=True)

    logger.info(f"Tokens: {', '.join(processed['tokens'][:15])}...")
    logger.info(f"POS Tags (first 10): {processed['pos_tags'][:10]}")

    parser = BaselineParser(language)
    chunk_tree = parser.chunk(processed['pos_tags'])
    
    np_chunks = []
    for subtree in chunk_tree.subtrees():
        if subtree.label() == 'NP':
            np_chunks.append(" ".join(word for word, tag in subtree.leaves()))

    if np_chunks:
        logger.info(f"Noun Phrases detected: {np_chunks[:3]}...")
    else:
        logger.info("Noun Phrases: None detected")

    ner = NamedEntityRecognizer(language)
    entities = ner.extract_entities(text, processed['tokens'])
    logger.info(f"Named Entities: {entities}")

    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'superb'}
    negative_words = {'bad', 'terrible', 'awful', 'poor', 'horrible', 'disappointing', 'boring', 'confusing'}

    tokens_lower = [t.lower() for t in processed['tokens']]
    pos_count = sum(1 for t in tokens_lower if t in positive_words)
    neg_count = sum(1 for t in tokens_lower if t in negative_words)

    if pos_count > neg_count:
        sentiment = 'POSITIVE'
    elif neg_count > pos_count:
        sentiment = 'NEGATIVE'
    else:
        sentiment = 'NEUTRAL'

    logger.info(f"Rule-Based Sentiment: {sentiment} (Pos: {pos_count}, Neg: {neg_count})")

if __name__ == "__main__":
    results = run_complete_pipeline()

    logger.info("Generating Visualizations...")
    languages = list(results.keys())
    plot_sentiment_performance(results, languages)
    plot_perplexity_comparison(results)
    plot_pos_frequencies(results)

    analyze_review_demo("Christopher Nolan's Inception was absolutely brilliant!", "english")
    analyze_review_demo("The movie was terrible and boring. Even Tom Hanks couldn't save this disaster.", "english")