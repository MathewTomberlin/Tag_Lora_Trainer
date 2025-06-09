# Tag_Lora_Trainer
A word list-to-Danbooru tag list LoRa trainer. The current version is designed to train a Dolphin3-Qwen2.5:3b model on generated word lists to tag lists.

# Running
python tag_lora_trainer.py

# Required
- Dataset (currently expects local metadata.parquet with Danbooru tag dataset)
- Base Model (currently provides a HuggingFace repo to download from, but may take a local directory)
