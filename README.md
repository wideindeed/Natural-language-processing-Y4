# Multilingual Movie Reviews NLP Pipeline

## Overview
This project implements a complete NLP pipeline for analyzing movie reviews in English and Arabic. It compares the performance of standard NLP techniques—including POS tagging, parsing, Named Entity Recognition (NER), N-gram language modeling, and sentiment classification—across two distinct languages.

## Project Structure
* `src/`: Source code modules (main pipeline, data loading, preprocessing, models, etc.).
* `Data_Pointers.txt`: Instructions for downloading the required datasets.
* `Launcher.bat`: One-click script to run the entire pipeline on Windows.
* `pipeline_results.json`: The raw output metrics from the latest run.
* `requirements.txt`: List of Python dependencies.
* `LICENSE`: MIT License information.

## Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/wideindeed/Natural-language-processing-Y4.git](https://github.com/wideindeed/Natural-language-processing-Y4.git)
    cd Natural-language-processing-Y4
    ```

2.  **Setup Data:**
    * Read `Data_Pointers.txt`.
    * Download `IMDB Dataset.csv` and `ar_reviews_100k.tsv` from the links provided (inside the Data_Pointers.txt file).
    * Place them in the **root** folder of this project.

3.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    camel_data -i light
    ```

## Usage
There are two ways to run the pipeline:

### Option 1: The Easy Way (Windows)
Double-click the **`Launcher.bat`** file. This will automatically activate the environment and run the code.

### Option 2: Manual Execution
Run the following command from the root directory:
```bash
python src/main.py
