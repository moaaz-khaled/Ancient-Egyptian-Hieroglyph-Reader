"""
🏺 Hieroglyph NLP Pipeline - Core Models & Functions
Transliteration → Dictionary → spaCy → Arabic Translation → Sentiment
"""

import re
import pandas as pd
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from collections import defaultdict
import os


# ═══════════════════════════════════════════════════════════════════════
# 1️⃣  LOAD GARDINER SIGN LIST
# ═══════════════════════════════════════════════════════════════════════

def load_gardiner_signs(csv_path='Test/Data/Update_gardiner_sign.csv'):
    """Load Gardiner sign database from CSV"""
    try:
        df = pd.read_csv(csv_path)
        gardiner_map = {}
        for _, row in df.iterrows():
            code = str(row['code']).strip().lower()
            if not code or code == 'nan':
                continue
            gardiner_map[code] = {
                'phonetic': str(row['phonetic']).strip() if pd.notna(row['phonetic']) else '',
                'meaning': str(row['meaning']).strip() if pd.notna(row['meaning']) else '',
                'unicode': str(row['unicode']).strip() if pd.notna(row['unicode']) else '',
            }
        print(f'✅ Gardiner Sign List loaded: {len(gardiner_map)} signs')
        return gardiner_map
    except Exception as e:
        print(f'❌ Error loading Gardiner signs: {e}')
        return {}


# ═══════════════════════════════════════════════════════════════════════
# 2️⃣  LOAD EGYPTIAN DICTIONARY
# ═══════════════════════════════════════════════════════════════════════

def load_egyptian_dictionary(csv_path='Test/Data/egyptian_dictionary.csv'):
    """Load Egyptian-English dictionary from CSV"""
    try:
        df_dict = pd.read_csv(csv_path)
        _raw = defaultdict(list)
        for _, row in df_dict.iterrows():
            key = str(row['transliteration']).strip()
            val = str(row['english']).strip()
            if not key or key == 'nan' or not val or val == 'nan':
                continue
            if val not in _raw[key]:
                _raw[key].append(val)
        egyptian_dict = dict(_raw)
        print(f'✅ Egyptian Dictionary loaded: {len(egyptian_dict)} entries')
        return egyptian_dict
    except Exception as e:
        print(f'❌ Error loading dictionary: {e}')
        return {}


# ═══════════════════════════════════════════════════════════════════════
# 3️⃣  LOAD INTENTION DATASET
# ═══════════════════════════════════════════════════════════════════════

def load_intention_dataset(csv_path='Test/Data/intention_dataset.csv'):
    """Load intention dataset from CSV"""
    try:
        df_intent = pd.read_csv(csv_path)
        intention_map = {}
        for _, row in df_intent.iterrows():
            intent_en = str(row['intention_en']).strip()
            intent_ar = str(row['intention_ar']).strip()
            keywords = [kw.strip().lower() for kw in str(row['keywords']).split(',')]
            intention_map[intent_en] = {
                'arabic': intent_ar,
                'keywords': set(keywords),
            }
        print(f'✅ Intention dataset loaded: {len(intention_map)} intentions')
        return intention_map
    except Exception as e:
        print(f'⚠️  Warning: Intention dataset not loaded: {e}')
        return {}


# ═══════════════════════════════════════════════════════════════════════
# 4️⃣  LOAD NLP MODELS
# ═══════════════════════════════════════════════════════════════════════

def load_spacy_model():
    """Load spaCy English model"""
    try:
        nlp = spacy.load('en_core_web_sm')
        print('✅ spaCy model loaded')
        return nlp
    except OSError:
        print('⚠️  spaCy model not found. Installing...')
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        return nlp


def load_translation_model():
    print('🔄 Loading NLLB translation model...')
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        NLLB_MODEL = 'facebook/nllb-200-distilled-600M'
        ar_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        ar_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        ar_model.eval()
        
        print('✅ NLLB model loaded')
        return ar_model, ar_tokenizer, 'arb_Arab'
        
    except Exception as e:
        print(f'❌ Error: {e}')
        return None, None, None


def load_sentiment_model():
    """Load Twitter RoBERTa sentiment model - Optional"""
    try:
        import os
        free_space_mb = 500  # Estimate
        if hasattr(os, 'statvfs'):
            stats = os.statvfs(os.path.expanduser("~/.cache"))
            free_space_mb = stats.f_bavail * stats.f_frsize // (1024*1024)
        
        if free_space_mb < 600:
            print(f'⚠️  Sentiment model skipped (disk space: {free_space_mb}MB < 600MB needed)')
            return None, None
            
        print('🔄 Loading sentiment model...')
        model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, timeout=10)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, timeout=10)
        model.eval()
        print('✅ Sentiment model loaded')
        return model, tokenizer
    except Exception as e:
        print(f'⚠️  Sentiment model not available (optional): {str(e)[:50]}...')
        return None, None


# ═══════════════════════════════════════════════════════════════════════
# 5️⃣  TRANSLITERATION ENGINE + SPACY SENTENCE BUILDER (CELL 4)
# ═══════════════════════════════════════════════════════════════════════

def extract_core_meaning(meanings):
    """Extract the primary meaning from a list"""
    if isinstance(meanings, list):
        text = meanings[0]
    else:
        text = str(meanings)
    first = re.split(r'[/|]', text)[0].strip()
    first = re.sub(r'^be ', '', first).strip()
    return first.lower()


def build_sentence_spacy(core_meanings, nlp_spacy):
    """
    Use spaCy POS tagger to build grammatically correct sentences
    ['name', 'sun'] -> 'my name is sun'
    """
    if not core_meanings:
        return ''
    if len(core_meanings) == 1:
        return core_meanings[0]

    text = ' '.join(core_meanings)
    doc = nlp_spacy(text)
    nouns = [t.text for t in doc if t.pos_ in ('NOUN', 'PROPN')]
    verbs = [t.text for t in doc if t.pos_ == 'VERB']
    adjs = [t.text for t in doc if t.pos_ == 'ADJ']

    possession = {'name', 'son', 'daughter', 'house', 'heart',
                  'brother', 'sister', 'father', 'mother', 'lord'}

    if verbs and nouns:
        return f'{verbs[0]} the {nouns[0]}'
    if len(nouns) >= 2:
        n1, n2 = nouns[0], nouns[1]
        if n1.lower() in possession:
            return f'my {n1} is {n2}'
        return f'{n1} of {n2}'
    if nouns and adjs:
        return f'the {adjs[0]} {nouns[0]}'
    if adjs:
        return f'it is {adjs[0]}'
    return text


def transliterate(gardiner_codes, gardiner_map, egypt_dict, nlp_spacy):
    """
    Full transliteration pipeline:
    1. Map Gardiner codes to phonetics
    2. Dictionary lookup (greedy matching)
    3. spaCy sentence building
    """
    codes = [c.lower().strip() for c in gardiner_codes]
    phonetics = []
    glyphs = []
    sign_meanings = []
    unknown = []

    for code in codes:
        if code in gardiner_map:
            info = gardiner_map[code]
            phonetics.append(info['phonetic'])
            glyphs.append(info['unicode'])
            sign_meanings.append(info['meaning'])
        else:
            phonetics.append('?')
            glyphs.append('□')
            sign_meanings.append('unknown')
            unknown.append(code)

    ph_clean = [p for p in phonetics if p and p != '?']
    full = ''.join(ph_clean)

    # Greedy multi-phoneme matching
    token_results = []
    core_meanings = []
    i = 0
    while i < len(ph_clean):
        matched = False
        for length in range(min(4, len(ph_clean) - i), 0, -1):
            combined = ''.join(ph_clean[i:i+length])
            if combined in egypt_dict:
                meanings_list = egypt_dict[combined]
                meaning_str = ' | '.join(meanings_list)
                core = extract_core_meaning(meanings_list)
                token_results.append({
                    'phonetic': combined,
                    'meaning': meaning_str,
                    'core': core,
                    'found': True
                })
                core_meanings.append(core)
                i += length
                matched = True
                break
        if not matched:
            ph = ph_clean[i]
            token_results.append({
                'phonetic': ph,
                'meaning': f'[{ph}]',
                'core': ph,
                'found': False
            })
            i += 1

    # Sliding window assembly for finding complete words
    assembly_candidates = []
    if full:
        assembly_candidates.append(full)
    for size in range(len(ph_clean), 0, -1):
        for start in range(len(ph_clean) - size + 1):
            window = ''.join(ph_clean[start:start+size])
            if window and window not in assembly_candidates:
                assembly_candidates.append(window)

    found_words = []
    for candidate in assembly_candidates:
        if candidate in egypt_dict:
            meanings_list = egypt_dict[candidate]
            meaning_str = ' | '.join(meanings_list)
            found_words.append({
                'transliteration': candidate,
                'meaning': meaning_str,
                'confidence': 'high' if candidate == full else 'partial',
            })

    high = [w for w in found_words if w['confidence'] == 'high']
    part = [w for w in found_words if w['confidence'] == 'partial']
    best = high[0] if high else (part[0] if part else None)
    
    sentence = build_sentence_spacy(core_meanings, nlp_spacy)

    return {
        'input_codes': codes,
        'glyphs': glyphs,
        'glyph_str': ' '.join(glyphs),
        'per_sign': list(zip(codes, phonetics, sign_meanings)),
        'phonetics': phonetics,
        'phonetic_str': ' '.join(ph_clean),
        'assembled': full,
        'token_results': token_results,
        'found_words': found_words,
        'unknown_codes': unknown,
        'best_meaning': best['meaning'] if best else None,
        'sentence': sentence,
        'core_meanings': core_meanings,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6️⃣  TRANSLATION (CELL 5)
# ═══════════════════════════════════════════════════════════════════════

def translate_to_arabic(english_text, ar_model, ar_tokenizer, lang_code=None):
    if not english_text or english_text.startswith('['):
        return ''
    if ar_model is None or ar_tokenizer is None:
        print(f'⚠️  Translation skipped: model={ar_model is not None}, tokenizer={ar_tokenizer is not None}')
        return english_text
    try:
        inputs = ar_tokenizer(
            english_text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            translated = ar_model.generate(
                **inputs,
                forced_bos_token_id=ar_tokenizer.convert_tokens_to_ids('arb_Arab'),
                max_new_tokens=60,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
            )
        result = ar_tokenizer.decode(translated[0], skip_special_tokens=True)
        print(f'🔄 Translated: "{english_text}" → "{result}"')
        return result
    except Exception as e:
        print(f'❌ Translation error: {e}')
        return f'[error: {e}]'


# ═══════════════════════════════════════════════════════════════════════
# 7️⃣  SENTIMENT ANALYSIS (CELL 6)
# ═══════════════════════════════════════════════════════════════════════

def analyze_sentiment(text, sent_model, sent_tokenizer):
    """Analyze sentiment using Twitter RoBERTa (or default to neutral)"""
    if not text or text.startswith('['):
        return 'neutral', 0.5
    
    # If model not available, return neutral
    if sent_model is None or sent_tokenizer is None:
        return 'neutral', 0.5
    
    try:
        inputs = sent_tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = sent_model(**inputs)
        import numpy as np
        scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
        best_idx = int(np.argmax(scores))
        labels = ['negative', 'neutral', 'positive']
        return labels[best_idx], round(float(scores[best_idx]), 3)
    except:
        return 'neutral', 0.5


# ═══════════════════════════════════════════════════════════════════════
# 8️⃣  INTENTION DETECTION (CELL 6)
# ═══════════════════════════════════════════════════════════════════════

def detect_intention(text, phonetics, intention_map):
    """Detect intention from text"""
    combined = (text + ' ' + phonetics).lower()
    words = set(combined.split())
    scores = {}
    for intent, data in intention_map.items():
        hits = len(words & data['keywords'])
        if hits > 0:
            scores[intent] = hits
    if not scores:
        return 'descriptive', 'وصفي'
    best_intent = max(scores, key=scores.get)
    best_ar = intention_map[best_intent]['arabic']
    return best_intent, best_ar


# ═══════════════════════════════════════════════════════════════════════
# 9️⃣  FULL NLP PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class HieroglyphNLPPipeline:
    """Complete Hieroglyph NLP Pipeline"""
    
    def __init__(self):
        print('🏺 Initializing Hieroglyph NLP Pipeline...')
        
        # Load data
        self.gardiner_map = load_gardiner_signs()
        self.egypt_dict = load_egyptian_dictionary()
        self.intention_map = load_intention_dataset()
        
        # Load models
        self.nlp_spacy = load_spacy_model()
        result = load_translation_model()
        if len(result) == 3:
            self.trans_model, self.trans_tokenizer, self.trans_lang = result
        else:
            self.trans_model, self.trans_tokenizer = result
            self.trans_lang = None
        self.sent_model, self.sent_tokenizer = load_sentiment_model()
        
        print('✅ Pipeline ready!')
    
    def process(self, gardiner_codes):
        """Process Gardiner codes through full pipeline"""
        # Step 1: Transliteration
        trans = transliterate(
            gardiner_codes,
            self.gardiner_map,
            self.egypt_dict,
            self.nlp_spacy
        )
        
        glyph_str = trans['glyph_str']
        phonetic_str = trans['phonetic_str']
        found_words = trans['found_words']
        best_meaning = trans['best_meaning']
        sentence = trans['sentence']
        
        # Determine English meaning - PRIORITY: spaCy sentence > dictionary > unknown
        if sentence and len(sentence) > 0:
            english = sentence
            method = 'spacy-nlp'
        elif best_meaning:
            english = best_meaning
            method = 'dictionary'
        else:
            english = f'[unknown: {trans["assembled"]}]'
            method = 'none'
        
        # Step 2: Translation - Use the final English sentence
        arabic = translate_to_arabic(
            english,
            self.trans_model,
            self.trans_tokenizer,
            self.trans_lang
        )
        
        # Step 3: Sentiment Analysis
        sentiment_text = (sentence + ' ' + (best_meaning or '')).strip()
        sentiment, sent_score = analyze_sentiment(
            sentiment_text,
            self.sent_model,
            self.sent_tokenizer
        )
        
        # Step 4: Intention Detection
        intention_en, intention_ar = detect_intention(
            sentiment_text,
            phonetic_str,
            self.intention_map
        )
        
        return {
            'input': gardiner_codes,
            'glyphs': glyph_str,
            'per_sign': trans['per_sign'],
            'phonetics': phonetic_str,
            'assembled': trans['assembled'],
            'token_results': trans['token_results'],
            'found_words': found_words[:3],  # Top 3 matches
            'sentence': sentence,
            'english': english,
            'trans_method': method,
            'arabic': arabic,
            'sentiment': sentiment,
            'sent_score': sent_score,
            'intention_en': intention_en,
            'intention_ar': intention_ar,
        }