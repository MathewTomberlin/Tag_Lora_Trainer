#!/usr/bin/env python3
"""
Improved Danbooru Tag LoRA Training Script
Trains a LoRA adapter to transform natural language descriptions into Danbooru tags
"""

import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import List, Dict, Tuple
import re
import random
from collections import Counter

#Word generation
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import gensim.downloader as api

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

# Could generate, store and output a seed instead of using a fixed seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Tags to manually remove
UNWANTED_TAGS = {
    'commentary_request', 'translated', '</tool_call>', '<tool_call>',
    'bad_id', 'bad_pixiv_id', 'character_request'
}

# Trie os a character is ascii
def is_ascii(s):
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def clean_and_filter_tags(tag_list):
    """Clean and filter tags for training"""
    cleaned = []
    for tag in tag_list:
        tag = tag.strip().lower()
        # Remove non-ASCII tags
        if not is_ascii(tag):
            continue
        # Remove tags with invalid characters
        if not re.match(r'^[a-z0-9_]+$', tag):
            continue
        # Remove unwanted tags
        if tag in UNWANTED_TAGS:
            continue
        cleaned.append(tag)
    return cleaned
    
class AnimeTermGenerator:
    def __init__(self):
        # Common anime/Japanese term patterns
        self.character_patterns = {
            'girl': ['female', 'woman', 'lady', 'character'],
            'boy': ['male', 'man', 'guy', 'character'],
            'hair': ['hairstyle', 'locks', 'tresses'],
            'eyes': ['gaze', 'look', 'stare'],
            'uniform': ['outfit', 'attire', 'clothing', 'costume'],
            'school': ['academic', 'educational', 'student'],
        }
        
        self.color_terms = {
            'blue': ['azure', 'cerulean', 'sapphire', 'navy'],
            'red': ['crimson', 'scarlet', 'ruby', 'cherry'],
            'green': ['emerald', 'jade', 'forest', 'mint'],
            'purple': ['violet', 'lavender', 'plum', 'indigo'],
            'pink': ['rose', 'coral', 'magenta', 'salmon'],
            'yellow': ['golden', 'amber', 'blonde', 'lemon'],
            'black': ['dark', 'ebony', 'raven', 'obsidian'],
            'white': ['pale', 'ivory', 'snow', 'platinum'],
            'brown': ['brunette', 'chestnut', 'tan', 'coffee'],
            'orange': ['amber', 'copper', 'rust', 'flame'],
        }
        
        self.pose_actions = {
            'sitting': ['seated', 'resting', 'perched'],
            'standing': ['upright', 'erect', 'posed'],
            'walking': ['moving', 'stepping', 'strolling'],
            'running': ['racing', 'dashing', 'sprinting'],
            'lying': ['reclining', 'horizontal', 'resting'],
            'smiling': ['grinning', 'cheerful', 'happy'],
        }
    
    def generate_variants(self, tag: str) -> List[str]:
        """Generate variants for anime-specific terms"""
        clean_tag = tag.replace('_', ' ').lower().strip()
        variants = [clean_tag]
        
        # Check for color + noun combinations
        words = clean_tag.split()
        if len(words) == 2:
            color, noun = words
            if color in self.color_terms and noun in self.character_patterns:
                for c_var in self.color_terms[color][:2]:
                    for n_var in self.character_patterns[noun][:2]:
                        variants.append(f"{c_var} {n_var}")
        
        # Check individual word patterns
        for word in words:
            if word in self.character_patterns:
                variants.extend(self.character_patterns[word][:2])
            elif word in self.color_terms:
                variants.extend(self.color_terms[word][:2])
            elif word in self.pose_actions:
                variants.extend(self.pose_actions[word][:2])
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for variant in variants:
            if variant not in seen and len(variant) > 1:
                result.append(variant)
                seen.add(variant)
        
        return result[:4]  # Return up to 4 variants
    
class EmbeddingGenerator:
    def __init__(self):
        print("Loading word embeddings...")
        # Use pre-trained GloVe vectors (smaller download than word2vec)
        try:
            self.model = api.load('glove-wiki-gigaword-50')  # 50d is faster
            print("Loaded GloVe embeddings")
        except:
            print("Failed to load embeddings, falling back to basic mode")
            self.model = None
    
    def get_similar_words(self, word: str, topn: int = 4) -> List[str]:
        """Get similar words using embeddings"""
        if self.model is None:
            return [word.replace('_', ' ')]
        
        clean_word = word.replace('_', ' ').lower().strip()
        
        try:
            # For compound words, try both full word and parts
            candidates = [clean_word]
            
            # Add individual words from compound terms
            word_parts = clean_word.split()
            if len(word_parts) > 1:
                candidates.extend(word_parts)
            
            similar_words = []
            
            for candidate in candidates:
                if candidate in self.model:
                    similar = self.model.most_similar(candidate, topn=topn)
                    similar_words.extend([word for word, score in similar if score > 0.6])
            
            # Remove duplicates and filter
            filtered = []
            seen = set()
            for word in [clean_word] + similar_words:
                if (word not in seen and 
                    len(word) > 2 and 
                    len(word.split()) <= 2):
                    filtered.append(word)
                    seen.add(word)
            
            return filtered[:5]  # Return up to 5 variants
            
        except KeyError:
            return [clean_word]
        
class WordNetGenerator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms using WordNet"""
        synonyms = set()
        
        # Clean the word
        clean_word = word.replace('_', ' ').lower().strip()
        
        # Skip if it's a stop word or too short
        if clean_word in self.stop_words or len(clean_word) < 3:
            return [clean_word]
        
        # Get synonyms from WordNet
        for syn in wordnet.synsets(clean_word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if (synonym != clean_word and 
                    len(synonym.split()) <= 2 and
                    synonym not in self.stop_words):
                    synonyms.add(synonym)
        
        # Add hypernyms (broader categories)
        for syn in wordnet.synsets(clean_word):
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    hyper = lemma.name().replace('_', ' ').lower()
                    if (hyper != clean_word and 
                        len(hyper.split()) <= 2 and
                        hyper not in self.stop_words):
                        synonyms.add(hyper)
        
        result = [clean_word] + list(synonyms)[:4]  # Original + up to 4 synonyms
        return result

class NLPWordGenerator:
    """Simple wrapper to maintain compatibility with existing code"""
    
    def __init__(self):
        print("🚀 Initializing Conservative Word Generator...")
        self.generator = ConservativeWordGenerator()
        print("✅ Conservative Word Generator ready!")
    
    def generate_word_variants(self, danbooru_tags: List[str]) -> List[str]:
        """Generate word variants using conservative approach"""
        return self.generator.generate_word_variants(danbooru_tags)

class ConservativeWordGenerator:
    """Generates conservative, high-quality word variants that stay close to original tags"""

    def __init__(self):
        print("🚀 Initializing NLP Word Generator...")
        self.wordnet_gen = WordNetGenerator()
        self.anime_gen = AnimeTermGenerator()
        self.embedding_gen = EmbeddingGenerator()
        # Only high-confidence, conservative mappings for common terms
        self.safe_mappings = {
            # Character descriptors - very safe alternatives
            '1girl': ['girl', 'female', 'young woman'],
            '1boy': ['boy', 'male', 'young man'], 
            '2girls': ['two girls', 'pair of girls', 'duo of girls'],
            '2boys': ['two boys', 'pair of boys', 'duo of boys'],
            'multiple_girls': ['several girls', 'multiple females', 'group of girls'],
            'multiple_boys': ['several boys', 'multiple males', 'group of boys'],
            'solo': ['alone', 'single person', 'by herself', 'by himself'],

            # Hair colors - only obvious alternatives
            'blonde_hair': ['blonde hair', 'golden hair', 'yellow hair'],
            'brown_hair': ['brown hair', 'brunette hair', 'dark hair'],
            'black_hair': ['black hair', 'dark hair', 'raven hair'],
            'red_hair': ['red hair', 'ginger hair', 'crimson hair'],
            'blue_hair': ['blue hair', 'azure hair', 'cyan hair'],
            'green_hair': ['green hair', 'emerald hair', 'jade hair'],
            'purple_hair': ['purple hair', 'violet hair', 'lavender hair'],
            'pink_hair': ['pink hair', 'rose hair', 'magenta hair'],
            'white_hair': ['white hair', 'silver hair', 'platinum hair'],
            'grey_hair': ['grey hair', 'gray hair', 'ashen hair'],
            'orange_hair': ['orange hair', 'copper hair', 'amber hair'],

            # Hair length - simple alternatives
            'long_hair': ['long hair', 'lengthy hair', 'flowing hair'],
            'short_hair': ['short hair', 'cropped hair', 'brief hair'],
            'medium_hair': ['medium hair', 'shoulder length hair', 'mid length hair'],
            'very_long_hair': ['very long hair', 'extremely long hair', 'floor length hair'],

            # Eye colors - conservative
            'red_eyes': ['red eyes', 'crimson eyes', 'scarlet eyes'],
            'blue_eyes': ['blue eyes', 'azure eyes', 'cerulean eyes'],
            'green_eyes': ['green eyes', 'emerald eyes', 'jade eyes'],
            'brown_eyes': ['brown eyes', 'hazel eyes', 'chocolate eyes'],
            'purple_eyes': ['purple eyes', 'violet eyes', 'amethyst eyes'],
            'yellow_eyes': ['yellow eyes', 'golden eyes', 'amber eyes'],
            'pink_eyes': ['pink eyes', 'rose eyes', 'coral eyes'],
            'black_eyes': ['black eyes', 'dark eyes', 'obsidian eyes'],
            'grey_eyes': ['grey eyes', 'gray eyes', 'silver eyes'],
            'white_eyes': ['white eyes', 'pale eyes', 'pearl eyes'],

            # Animal features - keep specific and accurate
            'animal_ears': ['animal ears', 'beast ears', 'creature ears'],
            'cat_ears': ['cat ears', 'feline ears', 'kitty ears'],
            'dog_ears': ['dog ears', 'canine ears', 'puppy ears'],
            'fox_ears': ['fox ears', 'vulpine ears', 'kitsune ears'],
            'wolf_ears': ['wolf ears', 'lupine ears', 'wild canine ears'],
            'rabbit_ears': ['rabbit ears', 'bunny ears', 'hare ears'],
            'mouse_ears': ['mouse ears', 'rodent ears', 'tiny ears'],

            'tail': ['tail', 'rear appendage', 'back appendage'],
            'cat_tail': ['cat tail', 'feline tail', 'kitty tail'],
            'dog_tail': ['dog tail', 'canine tail', 'puppy tail'],
            'fox_tail': ['fox tail', 'vulpine tail', 'kitsune tail'],
            'wolf_tail': ['wolf tail', 'lupine tail', 'wild canine tail'],
            'rabbit_tail': ['rabbit tail', 'bunny tail', 'hare tail'],

            # Facial expressions and features
            'blush': ['blush', 'flushed cheeks', 'rosy cheeks'],
            'smile': ['smile', 'grin', 'happy expression'],
            'open_mouth': ['open mouth', 'parted lips', 'agape'],
            'closed_eyes': ['closed eyes', 'shut eyes', 'eyes closed'],
            'wink': ['wink', 'one eye closed', 'playful wink'],

            # Wings and fantasy elements
            'wings': ['wings', 'feathered wings', 'wing appendages'],
            'angel_wings': ['angel wings', 'feathered wings', 'divine wings'],
            'demon_wings': ['demon wings', 'bat wings', 'dark wings'],
            'fairy_wings': ['fairy wings', 'gossamer wings', 'delicate wings'],
            'horns': ['horns', 'horn protrusions', 'head horns'],
            'halo': ['halo', 'ring of light', 'divine circle'],

            # Common poses - simple alternatives
            'sitting': ['sitting', 'seated', 'in sitting position'],
            'standing': ['standing', 'upright', 'in standing position'],
            'lying': ['lying', 'lying down', 'reclined'],
            'walking': ['walking', 'moving', 'in motion'],
            'running': ['running', 'jogging', 'in motion'],
            'kneeling': ['kneeling', 'on knees', 'genuflecting'],
            'crouching': ['crouching', 'squatting', 'hunched down'],

            # Looking directions
            'looking_at_viewer': ['looking at viewer', 'direct gaze', 'eye contact'],
            'looking_away': ['looking away', 'averted gaze', 'distant look'],
            'looking_back': ['looking back', 'glancing back', 'backward glance'],
            'looking_up': ['looking up', 'upward gaze', 'skyward look'],
            'looking_down': ['looking down', 'downward gaze', 'lowered eyes'],

            # Basic clothing - keep specific
            'school_uniform': ['school uniform', 'student outfit', 'academic attire'],
            'dress': ['dress', 'gown', 'one piece garment'],
            'shirt': ['shirt', 'top', 'upper garment'],
            'blouse': ['blouse', 'shirt', 'top'],
            'skirt': ['skirt', 'bottom garment', 'lower wear'],
            'shorts': ['shorts', 'short pants', 'brief bottoms'],
            'pants': ['pants', 'trousers', 'leg wear'],
            'jacket': ['jacket', 'coat', 'outer garment'],
            'sweater': ['sweater', 'pullover', 'knit top'],
            'hoodie': ['hoodie', 'hooded sweater', 'hooded top'],

            # Accessories
            'hat': ['hat', 'headwear', 'head covering'],
            'cap': ['cap', 'hat', 'head covering'],
            'glasses': ['glasses', 'eyewear', 'spectacles'],
            'bow': ['bow', 'ribbon', 'hair ornament'],
            'necklace': ['necklace', 'neck jewelry', 'chain'],
            'earrings': ['earrings', 'ear jewelry', 'ear ornaments'],

            # Settings - simple
            'outdoors': ['outdoors', 'outside', 'exterior setting'],
            'indoors': ['indoors', 'inside', 'interior setting'],
            'school': ['school', 'academy', 'educational building'],
            'classroom': ['classroom', 'school room', 'learning space'],
            'bedroom': ['bedroom', 'sleeping room', 'private room'],
            'kitchen': ['kitchen', 'cooking area', 'food preparation area'],
            'library': ['library', 'book repository', 'reading room'],
            'park': ['park', 'green space', 'recreational area'],
            'beach': ['beach', 'seaside', 'coastal area'],
            'forest': ['forest', 'woods', 'woodland area'],

            # Objects
            'book': ['book', 'tome', 'reading material'],
            'bag': ['bag', 'purse', 'carrying case'],
            'flower': ['flower', 'blossom', 'bloom'],
            'tree': ['tree', 'large plant', 'woody plant'],
            'chair': ['chair', 'seat', 'seating furniture'],
            'table': ['table', 'desk', 'flat surface'],
            'window': ['window', 'glass opening', 'transparent barrier'],
            'door': ['door', 'entrance', 'portal'],
        }

        # Common word transformations that are always safe
        self.safe_transformations = [
            # Remove numbers safely
            ('1girl', 'girl'),
            ('1boy', 'boy'),
            ('2girls', 'two girls'),
            ('2boys', 'two boys'),

            # Simple underscore to space (always safe)
            ('_', ' '),

            # Plural/singular (conservative)
            ('ears', 'ear'),
            ('eyes', 'eye'),
            ('wings', 'wing'),
            ('horns', 'horn'),
        ]

    def generate_word_variants(self, danbooru_tags: List[str]) -> List[str]:
        """Generate 3 conservative, high-quality word variants"""

        # Variant 1: Clean version (remove underscores, minimal changes)
        variant1 = self._create_rephrased_variant(danbooru_tags)

        return [variant1]

    def _create_rephrased_variant(self, tags: List[str]) -> str:
        """Variant 3: Light rephrasing with different word choices"""
        rephrased_words = []

        for i, tag in enumerate(tags):
            # Get all possible variants for this tag
            tag_variant = self._get_variant(tag)
            
            # Select different variant for each pass
            if tag_variant:
                rephrased_words.append(tag_variant)
            else:
                # Fallback
                rephrased_words.append(tag.replace('_', ' '))

        # Different ordering for variety
        reordered = self._reorder_for_variety(rephrased_words, tags)
        return ', '.join(reordered)
    
    def _get_variant(self, tag: str) -> List[str]:
        """Get variants using all NLP approaches in priority order"""
        all_variants = []
        
        # 1. Try word embeddings (semantic similarity)
        embedding_results = self.embedding_gen.get_similar_words(tag)
        for result in embedding_results:
            if result not in all_variants:
                all_variants.append(result)
        
        if len(all_variants) < 2:
            # 2. Try WordNet first (best for common English words)
            wordnet_results = self.wordnet_gen.get_synonyms(tag)
            if len(wordnet_results) > 1:  # Found real synonyms
                all_variants.append(wordnet_results[0])
        
        if len(all_variants) < 2:
            # 3. Try anime-specific patterns
            anime_results = self.anime_gen.generate_variants(tag)
            for result in anime_results:
                if result not in all_variants:
                    all_variants.append(result)
        
        # 4. Safe
        if tag in self.safe_mappings:
            # Use the first (most conservative) alternative
            alternatives = self.safe_mappings[tag]
            all_variants.append(alternatives[0])  # Always use the safest option
        else:
            # No mapping found, just clean it up
            clean_word = tag.replace('_', ' ')
            all_variants.append(clean_word)
        
        random.shuffle(all_variants)  # Shuffle to add variety
        return all_variants[0]  # Return top variant

    def _reorder_naturally(self, words: List[str], original_tags: List[str]) -> List[str]:
        """Reorder words in a natural way (character -> appearance -> pose -> setting)"""
        character_words = []
        appearance_words = []
        pose_words = []
        other_words = []

        for word, original_tag in zip(words, original_tags):
            if any(char in original_tag for char in ['girl', 'boy', 'solo', 'multiple']):
                character_words.append(word)
            elif any(app in original_tag for app in ['hair', 'eyes', 'ears', 'tail', 'wings', 'blush']):
                appearance_words.append(word)
            elif any(pose in original_tag for pose in ['sitting', 'standing', 'lying', 'looking', 'smiling']):
                pose_words.append(word)
            else:
                other_words.append(word)

        # Combine in natural order
        return character_words + appearance_words + pose_words + other_words

    def _reorder_for_variety(self, words: List[str], original_tags: List[str]) -> List[str]:
        """Create variety by slightly shuffling within categories"""
        reordered = self._reorder_naturally(words, original_tags)

        # Very light shuffling - just swap adjacent non-character words sometimes
        if len(reordered) > 3:
            # Find where character words end
            char_end = 0
            for i, (word, tag) in enumerate(zip(reordered, original_tags)):
                if not any(char in tag for char in ['girl', 'boy', 'solo', 'multiple']):
                    char_end = i
                    break
                
            # Light shuffle of non-character words
            if char_end < len(reordered) - 2:
                remaining = reordered[char_end:]
                if len(remaining) >= 2 and random.random() > 0.5:
                    # Swap two adjacent words occasionally
                    idx = random.randint(0, len(remaining) - 2)
                    remaining[idx], remaining[idx + 1] = remaining[idx + 1], remaining[idx]
                reordered = reordered[:char_end] + remaining

        return reordered

# Loads data and creates training set.
# NOTE: You must provide it with the path of a local dataset. Currently it expects a parquet of Danbooru data at the location.
class DanbooruTagProcessor:
    """Processes and prepares Danbooru tag data for training"""
    
    def __init__(self):
        self.tag_frequency = {}
        self.common_tags = set()
        self.tag_categories = {}
        
    def load_danbooru_dataset(self, parquet_path: str = "metadata.parquet", max_samples: int = 100000) -> pd.DataFrame:
        """
        Load Danbooru dataset from a local Parquet file as a pandas DataFrame.
        """
        # Make sure the data file exists
        # NOTE: You must provide it with the path of a local dataset. Currently it expects a parquet of Danbooru data at the location.
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # Read the parquet data to a dataframe
        print(f"Loading Danbooru dataset from {parquet_path} ...")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

        # Select only necessary columns
        tag_columns = ["tag_string_general"]
        available_columns = [col for col in tag_columns if col in df.columns]
        df = df[available_columns].copy()

        # Combine the tag columns into a single tag string (space-separated)
        df["tag_string"] = df[available_columns].fillna("").agg(" ".join, axis=1).str.strip()

        # Remove rows with empty tag_string
        df = df[df["tag_string"] != ""]

        # Add a column for tag count
        df["tag_count"] = df["tag_string"].apply(lambda x: len(x.split()))

        # Limit to max_samples
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        print(f"Returning {len(df)} samples with tags.")
        return df[["tag_string", "tag_count"]]
    
    def analyze_tag_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze tag patterns and build vocabularies"""
        print("Analyzing tag patterns...")
        
        # Ensure tag_count column exists
        if 'tag_count' not in df.columns:
            print("Warning: tag_count column missing, calculating from general_tags...")
            df['tag_count'] = df['tag_string'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        all_tags = []
        for _, row in df.iterrows():
            tags = str(row['tag_string']).split()
            all_tags.extend(tags)
        
        # Count tag frequency
        self.tag_frequency = Counter(all_tags)
        
        # Get common tags (appear in at least 0.1% of images)
        min_frequency = max(10, len(df) * 0.001)
        self.common_tags = {tag.lower() for tag, freq in self.tag_frequency.items() 
                           if freq >= min_frequency}
        
        # Calculate stats with proper error handling
        tag_counts = df['tag_count'].dropna()
        stats = {
            'total_unique_tags': len(self.tag_frequency),
            'common_tags_count': len(self.common_tags),
            'avg_tags_per_image': float(np.mean(tag_counts)) if len(tag_counts) > 0 else 0,
            'median_tags_per_image': float(np.median(tag_counts)) if len(tag_counts) > 0 else 0,
            'max_tags_per_image': int(tag_counts.max()) if len(tag_counts) > 0 else 0
        }
        
        print(f"Tag analysis complete: {stats}")
        return stats
    
def create_tag_to_tag_dataset(processor: DanbooruTagProcessor, df: pd.DataFrame, 
                                num_samples: int = 15000) -> Tuple[List[str], List[str]]:
    """Create tag-to-tag training dataset with word phrase inputs and Danbooru tag outputs"""

    # Filter for quality examples
    df_filtered = df[
        (df['tag_count'] >= 3) &  # At least 3 tags
        (df['tag_count'] <= 20)   # Not too many tags
    ].copy()

    if len(df_filtered) < num_samples:
        print(f"Warning: Only {len(df_filtered)} quality samples available, using all")
    else:
        df_filtered = df_filtered.sample(n=num_samples, random_state=42)

    # Initialize mapping generator
    mapping_generator = NLPWordGenerator()

    training_inputs = []
    training_outputs = []

    print("Generating tag-to-tag training data...")
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        original_tags = row['tag_string'].split()

        # Clean and filter tags
        clean_tags = clean_and_filter_tags(original_tags)

        # Keep only common tags for training stability
        filtered_tags = [tag for tag in clean_tags if tag in processor.common_tags]

        if len(filtered_tags) >= 3:
            # Generate word phrase variants
            word_variants = mapping_generator.generate_word_variants(filtered_tags)

            # Create training pairs
            danbooru_output = ', '.join(filtered_tags)

            for word_variant in word_variants:
                training_inputs.append(word_variant)
                training_outputs.append(danbooru_output)

        if i % 100 == 0 and i > 0:
            print(f"{i}: {word_variants} -> {danbooru_output}")

    print(f"Created {len(training_inputs)} tag-to-tag training examples")
    return training_inputs, training_outputs

def validate_training_pair(word_phrase: str, danbooru_tags: str) -> bool:
    """Validate quality of training pairs"""

    # Check minimum lengths
    if len(word_phrase.split(',')) < 2 or len(danbooru_tags.split(',')) < 2:
        return False

    # Check for reasonable word phrase patterns
    phrase_lower = word_phrase.lower()
    if any(bad in phrase_lower for bad in ['error', 'invalid', 'none', 'tag_string']):
        return False

    # Check that tags look valid
    tags_lower = danbooru_tags.lower()
    if any(bad in tags_lower for bad in ['error', 'invalid', 'none']):
        return False

    return True

# Sets up the training base model and tokenizer, the LoRa training configs, the training prompt, tokenizes training
# and then trains the LoRa.
# NOTE: If you provide this class with a local directory to a model, it should use that instead
# of downloading from HuggingFace. Otherwise, provide a HuggingFace repo.
class DanbooruLoRATrainer:
    """Handles LoRA training for description to tag transformation"""
    
    # NOTE: model_name should either be a HuggingFace repo or a local directory with the model
    def __init__(self, model_name: str = "cognitivecomputations/Dolphin3.0-Qwen2.5-3b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self, output_dir: str = "./danbooru_tag_base"):
        """Initialize model and tokenizer"""
        print("Loading model and tokenizer...")

        # Load model. Will download (or use cached) if provided a HuggingFace repo, or will load from local directory
        print("Loading model for training...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        print("Loading model tokenizer for training...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"  # Important for generation
        )
        
        # Set up special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.eos_token = "<|im_end|>"

        print(f"Saving base model and tokenizer to {output_dir} ...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Base model and tokenizer saved.")
        
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration optimized for this task"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Higher rank for better quality
            lora_alpha=32,  # Higher alpha for stronger adaptation
            lora_dropout=0.1,  # Lower dropout for this specific task
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"      # MLP layers
            ],
            bias="none",
        )
    
    def create_training_prompt(self, word_phrase: str, danbooru_tags: str = None) -> str:
        """Create structured training prompt for tag-to-tag conversion"""
        system_prompt = "You are an expert at converting word phrases into precise Danbooru tags. Given a comma-separated list of descriptive words, provide the corresponding Danbooru tags."
    
        if danbooru_tags:
            # Training format
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{word_phrase}<|im_end|>\n"
                f"<|im_start|>assistant\n{danbooru_tags}<|im_end|>"
            )
        else:
            # Inference format
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{word_phrase}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    
    def prepare_training_data(self, training_inputs: List[str], 
                            training_outputs: List[str]) -> Dataset:
        """Prepare training dataset with optimized tokenization for tag conversion"""
        
        training_examples = []
        max_lengths = []
        
        # Generate training examples and track lengths
        for word_phrase, danbooru_tags in zip(training_inputs, training_outputs):
            # Validate training pair
            if not validate_training_pair(word_phrase, danbooru_tags):
                continue
                
            full_prompt = self.create_training_prompt(word_phrase, danbooru_tags)
            training_examples.append(full_prompt)
            
            # Estimate token length for optimization
            estimated_length = len(full_prompt.split()) * 1.3
            max_lengths.append(min(int(estimated_length), 128))
        
        # Use dynamic max length (95th percentile)
        optimal_max_length = min(int(np.percentile(max_lengths, 95)), 128)
        print(f"Using optimized max_length: {optimal_max_length}")
        
        # Tokenize with optimized settings
        print("Tokenizing training examples...")
        encodings = self.tokenizer(
            training_examples,
            truncation=True,
            padding="max_length",
            max_length=optimal_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Create labels for causal language modeling
        labels = encodings["input_ids"].clone()
        
        # Mask the prompt part in labels (only train on tag generation)
        print("Masking prompt parts of the labels...")
        assistant_prompt = "<|im_start|>assistant\n"
        
        for i, example in enumerate(training_examples):
            assistant_start_index = example.find(assistant_prompt)
            
            if assistant_start_index != -1:
                # Calculate approximate token position for masking
                char_ratio = (assistant_start_index + len(assistant_prompt)) / len(example)
                estimated_token_pos = int(char_ratio * optimal_max_length)
                labels[i, :min(estimated_token_pos, optimal_max_length)] = -100
    
        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })
    
    def train_lora(self, train_dataset: Dataset, output_dir: str = "./danbooru_tag_lora"):
        """Train the LoRA adapter with optimized settings for tag conversion"""
        
        # Setup LoRA
        lora_config = self.create_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Optimized training arguments for tag-to-tag conversion
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Batch size optimization (larger due to shorter sequences)
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            
            # Training schedule
            warmup_steps=50,
            num_train_epochs=2,  # Fewer epochs for direct mapping task
            learning_rate=1e-4,  # Higher learning rate for simpler task
            
            # Performance optimizations
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            group_by_length=True,
            remove_unused_columns=True,
            
            # Logging and saving
            logging_steps=20,
            save_steps=500,
            save_total_limit=2,
            
            # Additional optimizations
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting LoRA training for tag-to-tag conversion...")
        trainer.train()
        
        # Save the trained adapter
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"LoRA adapter saved to {output_dir}")
        
        return trainer

# Sets up the model and tokenizer, saving them
# Loads the dataset up to max_samples
# Analyzes the dataset for information
# Creates the training inputs and outputs
# Prepares and tokenizes the training data
# Trains the model with the training data and config values
def main():
    """Main training pipeline for tag-to-tag LoRA"""
    
    # Initialize components
    processor = DanbooruTagProcessor()
    trainer = DanbooruLoRATrainer()
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Load and process data. The samples here are filtered and used to generate descriptions
    print("Loading Danbooru dataset...")
    df = processor.load_danbooru_dataset(parquet_path="metadata.parquet",max_samples=100000)
    
    print("Analyzing tag patterns...")
    stats = processor.analyze_tag_patterns(df)
    
    # Create tag-to-tag training dataset. This is the max training samples
    training_inputs, training_outputs = create_tag_to_tag_dataset(processor, df, num_samples=30000)
    
    # Prepare training data
    print("Preparing tag-to-tag training dataset...")
    train_dataset = trainer.prepare_training_data(training_inputs, training_outputs)
    
    # Train LoRa
    print("Training tag-to-tag LoRa adapter...")
    trained_model = trainer.train_lora(train_dataset)
    
    print("\nTag-to-tag LoRa training completed!")
    print("\nTo use the trained LoRa model:")
    print("1. Load the tokenizer (AutoTokenizer) and base model (AutoModelForCausalLM) from ./danbooru_tag_base")
    print("2. Load the LoRa model (PeftModel) with the base model and the LoRa path './danbooru_tag_lora'")
    print("3. Prompt format:")
    print(f"     <|im_start|>system\\nYou are an expert at converting word phrases into precise Danbooru tags.")
    print(f"     Given a comma-separated list of descriptive words, provide the corresponding Danbooru tags.<|im_end|>")
    print(f"     <|im_start|>user\\ngirl, blue hair, school uniform, sitting<|im_end|>")
    print(f"     <|im_start|>assistant\\n")
    print(f"From testing, the best inference parameters to use with this LoRa are:")
    print(f"     Temperature: 0.1")
    print(f"     Top P: 0.95")
    print(f"     Top K: 30")
    print(f"     Repetition Penalty: 0.9")
    print(f"     Max New Tokens: 512")
    print("4. Expected Output: '1girl, blue_hair, school_uniform, sitting'")
    
    # Save example usage
    example_usage = {
        "model_name": trainer.model_name,
        "task": "tag-to-tag conversion",
        "input_format": "comma-separated word phrases",
        "output_format": "comma-separated Danbooru tags"
    }
    
    with open("./danbooru_tag_lora/usage_example.json", "w") as f:
        json.dump(example_usage, f, indent=2)
    
    print("Usage example saved to ./danbooru_tag_lora/usage_example.json")

if __name__ == "__main__":
    main()