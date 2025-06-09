# Tag_Lora_Trainer
A word list-to-Danbooru tag list LoRa trainer. The current version is designed to train a Dolphin3-Qwen2.5:3b model on generated word lists to tag lists.

# Running
From a terminal: `python tag_lora_trainer.py`

# Required
- Dataset (currently expects local metadata.parquet with Danbooru tag dataset)
- Base Model (currently provides a HuggingFace repo to download from, but may take a local directory)

# Using the LoRa
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re

def load_lora(_self, base_model_dir, lora_path):
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        # Load LoRa
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        return tokenizer, model
        
def lora_inference(self, description: str, settings: dict) -> str:
        # Build the inference prompt
        system_prompt = "You are an expert at converting word phrases into precise Danbooru tags. Given a comma-separated list of descriptive words, provide the corresponding Danbooru tags."
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{description}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
        # Use the lora with the prompt
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                top_k=30,
                repetition_penalty=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract the assistant response tag list
        if "<|im_start|>assistant\n" in full_response:
            tags = full_response.split("<|im_start|>assistant\n")[-1].strip()
            tags = re.sub(r'<\|.*?\|>', '', tags).strip()
            return tags
```
