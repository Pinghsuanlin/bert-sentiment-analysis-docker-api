# ================================= Configuration File ==============================
# a central repository for hyperparameters and file paths, ensuring easy adjustments and consistency across the project.

import transformers

DEVICE = "cuda" # or "cpu" depending on availability
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 10
ACCUMULATION_STEPS = 2
# BERT_PATH = "../input/bert-base-uncased"
# MODEL_SAVE_PATH = "./model.bin"
# TRAINING_FILE = "../input/imdb.csv"
# TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)





