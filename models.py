"""
🏺 Hieroglyph NLP Pipeline - Core Models & Functions
Transliteration → Dictionary → spaCy → Arabic Translation → Sentiment

Translation Model: ByteDance-Seed/Seed-X-PPO-7B
- بيشتغل local على GPU (NVIDIA)
- bfloat16 / float16 — الأكثر استقراراً وجودة
- الـ prompt لازم ينتهي بـ <ar> حسب الـ docs
"""

import re
import os
import pandas as pd
import spacy
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    AutoModelForSequenceClassification,
)
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ═══════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'🖥️  GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)')
    # اختار الـ dtype الأنسب حسب الـ VRAM
    COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    print('⚠️  No GPU found — running on CPU (slow)')
    COMPUTE_DTYPE = torch.float32


# ═══════════════════════════════════════════════════════════════════════
# SEED-X MODEL ID
# ═══════════════════════════════════════════════════════════════════════

SEED_X_MODEL_ID = 'ByteDance-Seed/Seed-X-PPO-7B'


# monkey-patch مش محتاجها local — كانت بس لـ Kaggle


# ═══════════════════════════════════════════════════════════════════════
# 1️⃣  LOAD GARDINER SIGN LIST
# ═══════════════════════════════════════════════════════════════════════

def load_gardiner_signs(csv_path='Test/Data/Update_gardiner_sign.csv'):
    try:
        df = pd.read_csv(csv_path)
        gardiner_map = {}
        for _, row in df.iterrows():
            code = str(row['code']).strip().lower()
            if not code or code == 'nan':
                continue
            gardiner_map[code] = {
                'phonetic': str(row['phonetic']).strip() if pd.notna(row['phonetic']) else '',
                'meaning':  str(row['meaning']).strip()  if pd.notna(row['meaning'])  else '',
                'unicode':  str(row['unicode']).strip()  if pd.notna(row['unicode'])  else '',
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
    try:
        df_intent = pd.read_csv(csv_path)
        intention_map = {}
        for _, row in df_intent.iterrows():
            intent_en = str(row['intention_en']).strip()
            intent_ar = str(row['intention_ar']).strip()
            keywords  = [kw.strip().lower() for kw in str(row['keywords']).split(',')]
            intention_map[intent_en] = {
                'arabic':   intent_ar,
                'keywords': set(keywords),
            }
        print(f'✅ Intention dataset loaded: {len(intention_map)} intentions')
        return intention_map
    except Exception as e:
        print(f'⚠️  Intention dataset not loaded: {e}')
        return {}


# ═══════════════════════════════════════════════════════════════════════
# 4️⃣  LOAD NLP MODELS
# ═══════════════════════════════════════════════════════════════════════

def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
        print('✅ spaCy model loaded')
        return nlp
    except OSError:
        print('⚠️  spaCy model not found. Installing...')
        os.system('python -m spacy download en_core_web_sm')
        return spacy.load('en_core_web_sm')


def load_translation_model():
    """
    تحميل Seed-X-PPO-7B — GPU 6.4GB + CPU RAM offload
    """
    print(f'🔄 Loading {SEED_X_MODEL_ID} ({COMPUTE_DTYPE}) ...')
    print('    GPU: 6.4GB → الموديل هيتوزع على GPU + CPU RAM تلقائي')
    print('    (أول مرة: بياخد 3-5 دقايق — من الـ cache بيبقى ثواني)')

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            SEED_X_MODEL_ID,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # max_memory: حط أكتر حاجة على GPU، الباقي على CPU RAM
        max_memory = {
            0: '5GiB',       # GPU  — نسيب 1.4GB buffer
            'cpu': '20GiB',  # CPU RAM — عندك 32GB
        }

        model = AutoModelForCausalLM.from_pretrained(
            SEED_X_MODEL_ID,
            dtype=COMPUTE_DTYPE,
            device_map='auto',
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()

        # اطبع توزيع الـ layers على GPU/CPU
        if hasattr(model, 'hf_device_map'):
            devices = set(str(v) for v in model.hf_device_map.values())
            print(f'✅ {SEED_X_MODEL_ID} loaded — layers on: {devices}')
        else:
            print(f'✅ {SEED_X_MODEL_ID} loaded')

        return model, tokenizer

    except Exception as e:
        import traceback
        print(f'❌ Translation model failed:')
        traceback.print_exc()
        return None, None


def load_sentiment_model():
    """
    Sentiment model — على CPU عشان يسيب VRAM كله لـ Seed-X
    """
    try:
        print('🔄 Loading sentiment model (CPU)...')
        model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        model      = AutoModelForSequenceClassification.from_pretrained(model_name)
        model      = model.to('cpu')
        model.eval()
        print('✅ Sentiment model loaded (CPU)')
        return model, tokenizer
    except Exception as e:
        print(f'⚠️  Sentiment model failed: {str(e)[:80]}')
        return None, None


# ═══════════════════════════════════════════════════════════════════════
# 5️⃣  TRANSLITERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def extract_core_meaning(meanings):
    text  = meanings[0] if isinstance(meanings, list) else str(meanings)
    first = re.split(r'[/|]', text)[0].strip()
    return re.sub(r'^be ', '', first).strip().lower()


def build_sentence_spacy(core_meanings, nlp_spacy):
    if not core_meanings:
        return ''
    if len(core_meanings) == 1:
        return core_meanings[0]

    doc   = nlp_spacy(' '.join(core_meanings))
    nouns = [t.text for t in doc if t.pos_ in ('NOUN', 'PROPN')]
    verbs = [t.text for t in doc if t.pos_ == 'VERB']
    adjs  = [t.text for t in doc if t.pos_ == 'ADJ']

    possession = {'name', 'son', 'daughter', 'house', 'heart',
                  'brother', 'sister', 'father', 'mother', 'lord'}

    if verbs and nouns:
        return f'{verbs[0]} the {nouns[0]}'
    if len(nouns) >= 2:
        n1, n2 = nouns[0], nouns[1]
        return f'my {n1} is {n2}' if n1.lower() in possession else f'{n1} of {n2}'
    if nouns and adjs:
        return f'the {adjs[0]} {nouns[0]}'
    if adjs:
        return f'it is {adjs[0]}'
    return ' '.join(core_meanings)


def transliterate(gardiner_codes, gardiner_map, egypt_dict, nlp_spacy):
    codes         = [c.lower().strip() for c in gardiner_codes]
    phonetics, glyphs, sign_meanings, unknown = [], [], [], []

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
    full     = ''.join(ph_clean)

    token_results, core_meanings = [], []
    i = 0
    while i < len(ph_clean):
        matched = False
        for length in range(min(4, len(ph_clean) - i), 0, -1):
            combined = ''.join(ph_clean[i:i+length])
            if combined in egypt_dict:
                ml   = egypt_dict[combined]
                core = extract_core_meaning(ml)
                token_results.append({'phonetic': combined, 'meaning': ' | '.join(ml), 'core': core, 'found': True})
                core_meanings.append(core)
                i += length
                matched = True
                break
        if not matched:
            token_results.append({'phonetic': ph_clean[i], 'meaning': f'[{ph_clean[i]}]', 'core': ph_clean[i], 'found': False})
            i += 1

    candidates = [full] if full else []
    for size in range(len(ph_clean), 0, -1):
        for start in range(len(ph_clean) - size + 1):
            w = ''.join(ph_clean[start:start+size])
            if w and w not in candidates:
                candidates.append(w)

    found_words = []
    for c in candidates:
        if c in egypt_dict:
            ml = egypt_dict[c]
            found_words.append({
                'transliteration': c,
                'meaning':         ' | '.join(ml),
                'confidence':      'high' if c == full else 'partial',
            })

    high = [w for w in found_words if w['confidence'] == 'high']
    best = high[0] if high else (found_words[0] if found_words else None)

    return {
        'input_codes':   codes,
        'glyphs':        glyphs,
        'glyph_str':     ' '.join(glyphs),
        'per_sign':      list(zip(codes, phonetics, sign_meanings)),
        'phonetics':     phonetics,
        'phonetic_str':  ' '.join(ph_clean),
        'assembled':     full,
        'token_results': token_results,
        'found_words':   found_words,
        'unknown_codes': unknown,
        'best_meaning':  best['meaning'] if best else None,
        'sentence':      build_sentence_spacy(core_meanings, nlp_spacy),
        'core_meanings': core_meanings,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6️⃣  TRANSLATION — Seed-X-PPO-7B
# ═══════════════════════════════════════════════════════════════════════

def translate_to_arabic(english_text, trans_model, trans_tokenizer, target_lang='ar'):
    """
    ترجمة إنجليزي → عربي بـ Seed-X-PPO-7B

    الـ prompt الصح حسب الـ docs:
      "Translate the following English text into Arabic:\n{text} <ar>"
    """
    if not english_text or english_text.startswith('['):
        return ''
    if trans_model is None or trans_tokenizer is None:
        return ''

    prompt = (
        f'Translate the following English text into Arabic:\n'
        f'{english_text} <{target_lang}>'
    )

    try:
        inputs = trans_tokenizer(prompt, return_tensors='pt')
        inputs.pop('token_type_ids', None)

        device = next(trans_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = trans_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=trans_tokenizer.eos_token_id,
                eos_token_id=trans_tokenizer.eos_token_id,
                use_cache=True,
            )

        input_len = inputs['input_ids'].shape[1]
        arabic    = trans_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return arabic if arabic else english_text

    except Exception as e:
        print(f'❌ Translation failed: {str(e)[:100]}')
        return english_text


# ═══════════════════════════════════════════════════════════════════════
# 7️⃣  SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_sentiment(text, sent_model, sent_tokenizer):
    if not text or text.startswith('['):
        return 'neutral', 0.5
    if sent_model is None or sent_tokenizer is None:
        return 'neutral', 0.5
    try:
        inputs = sent_tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        # CPU دايماً للـ sentiment
        with torch.no_grad():
            scores = torch.softmax(sent_model(**inputs).logits, dim=1).numpy()[0]
        idx    = int(np.argmax(scores))
        labels = ['negative', 'neutral', 'positive']
        return labels[idx], round(float(scores[idx]), 3)
    except:
        return 'neutral', 0.5


# ═══════════════════════════════════════════════════════════════════════
# 8️⃣  INTENTION DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_intention(text, phonetics, intention_map):
    combined = (text + ' ' + phonetics).lower()
    words    = set(combined.split())
    scores   = {i: len(words & d['keywords']) for i, d in intention_map.items() if words & d['keywords']}
    if not scores:
        return 'descriptive', 'وصفي'
    best = max(scores, key=scores.get)
    return best, intention_map[best]['arabic']


# ═══════════════════════════════════════════════════════════════════════
# 9️⃣  FULL PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════

class HieroglyphNLPPipeline:
    """
    Complete Hieroglyph NLP Pipeline — Local GPU version

    التحميل بيحصل مرة واحدة عند الـ init.
    Seed-X-PPO-7B على GPU، Sentiment على CPU.
    """

    def __init__(self):
        print('🏺 Initializing Hieroglyph NLP Pipeline...')
        print(f'   Device : {DEVICE}')
        print(f'   DType  : {COMPUTE_DTYPE}')

        # ── Data ──────────────────────────────────────────────────
        self.gardiner_map  = load_gardiner_signs()
        self.egypt_dict    = load_egyptian_dictionary()
        self.intention_map = load_intention_dataset()

        # ── Models ────────────────────────────────────────────────
        self.nlp_spacy = load_spacy_model()

        self.trans_model, self.trans_tokenizer = load_translation_model()

        self.sent_model, self.sent_tokenizer = load_sentiment_model()

        print('✅ Pipeline ready!')

    def process(self, gardiner_codes):
        # Step 1: Transliteration
        trans        = transliterate(gardiner_codes, self.gardiner_map, self.egypt_dict, self.nlp_spacy)
        best_meaning = trans['best_meaning']
        sentence     = trans['sentence']

        if sentence:
            english, method = sentence, 'spacy-nlp'
        elif best_meaning:
            english, method = best_meaning, 'dictionary'
        else:
            english, method = f'[unknown: {trans["assembled"]}]', 'none'

        # Step 2: Translation → Arabic
        arabic = translate_to_arabic(
            english,
            self.trans_model,
            self.trans_tokenizer,
        )

        # Step 3: Sentiment
        sentiment_text        = (sentence + ' ' + (best_meaning or '')).strip()
        sentiment, sent_score = analyze_sentiment(sentiment_text, self.sent_model, self.sent_tokenizer)

        # Step 4: Intention
        intention_en, intention_ar = detect_intention(
            sentiment_text, trans['phonetic_str'], self.intention_map
        )

        return {
            'input':         gardiner_codes,
            'glyphs':        trans['glyph_str'],
            'per_sign':      trans['per_sign'],
            'phonetics':     trans['phonetic_str'],
            'assembled':     trans['assembled'],
            'token_results': trans['token_results'],
            'found_words':   trans['found_words'][:3],
            'sentence':      sentence,
            'english':       english,
            'trans_method':  method,
            'arabic':        arabic,
            'sentiment':     sentiment,
            'sent_score':    sent_score,
            'intention_en':  intention_en,
            'intention_ar':  intention_ar,
        }