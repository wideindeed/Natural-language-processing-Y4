# Multilingual Movie Reviews NLP Pipeline

## Overview
This project implements a complete NLP pipeline for analyzing movie reviews in English and Arabic. It includes preprocessing, POS tagging, parsing, NER, language modeling (N-grams), and sentiment classification.

## Project Structure
* `src/`: Source code modules (`main.py`, `data_loader.py`, etc.).
* `notebooks/`: Jupyter notebooks for data exploration.
* `requirements.txt`: List of dependencies.
* `README.md`: Project documentation.

## Installation
1.  Clone the repository.
2.  Create a virtual environment: `python -m venv venv`
3.  Activate the environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    camel_data -i light
    ```

## Usage
Run the main pipeline script:
```bash
python src/main.py