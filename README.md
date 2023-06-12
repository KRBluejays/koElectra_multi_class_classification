# Event Analysis using Fine-tuned KoELECTRA Model

This repository contains a script for fine-tuning a pre-trained Korean ELECTRA model (KoELECTRA) to perform event analysis on text data. It uses transformers and PyTorch for the implementation.

## Requirements
- Python 3.8+
- PyTorch 1.9.0+
- transformers 4.9.1+
- pandas 1.3.3+
- tqdm 4.62.2+
- scikit-learn 0.24.2+

To install all the required libraries, run:

```
pip install -r requirements.txt
```

## Dataset
The script uses a custom dataset with the structure:

```
main_dataset_folder/
├── event_1/
│   ├── file_1.txt
│   ├── file_2.txt
│   ├── ...
├── event_2/
│   ├── file_1.txt
│   ├── file_2.txt
│   ├── ...
├── ...
```
Each subfolder in the main dataset directory represents a type of event and contains text files corresponding to that event.

## Usage
1. To use this script, clone the repository and navigate to it:

```
git clone <repository_url>
cd <repository_name>
```

2. Modify the `main_dataset_dir` variable in the `main` function to point to your dataset directory.

3. Run the script:

```
python koElectra_ver_3.py
```

The script will train and evaluate the model, and save it along with its configuration and the tokenizer used during training. The model's performance over the epochs is logged into a file (`log_history.log`), and a mapping between events and their corresponding labels is also saved (`event_mapping2.json`).

## Model
The script uses a pre-trained KoELECTRA model ("monologg/koelectra-base-v3-discriminator"). It fine-tunes the model for sequence classification with the number of classes equal to the number of event types in the dataset.

## Functionality
The script loads text files from each event subfolder, tokenizes the content of each text file using a tokenizer from the `transformers` library, and maps each event to an integer label. It then splits the resulting dataset into a training set and a test set.

The model is trained over a specified number of epochs (default is 20), and its performance is evaluated at the end of each epoch on the test set. If the model's performance improves, it is saved. The function logs the model's performance over the epochs, evaluates it on the testing set and then saves it.

## Outputs
- The fine-tuned model: `koelectra-base-finetuned-sentiment-analysis_best_table_parse.bin` and `koelectra-base-finetuned-sentiment-analysis_table_parse.bin`
- Training logs: `log_history.log`
- Event to integer mapping: `event_mapping2.json`

## Contribution
This script is open to improvements and bug fixes. Feel free to create an issue or send a pull request.

## Contact
For any questions or issues, please open an issue on this GitHub repository.
