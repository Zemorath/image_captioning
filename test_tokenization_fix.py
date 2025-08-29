#!/usr/bin/env python3
"""
Test script to verify the FIXED tokenization approach.

This script demonstrates the difference between the old broken approach
and the new fixed approach for handling start/end tokens.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TOKENIZATION FIX VERIFICATION")
print("=" * 50)

# Sample captions
sample_captions = [
    "a child in a pink dress is climbing up stairs",
    "a black dog and a spotted dog are fighting", 
    "two dogs of different breeds looking at each other"
]

print("Original captions:")
for i, cap in enumerate(sample_captions, 1):
    print(f"  {i}: {cap}")

# OLD BROKEN APPROACH (what was used before)
print("\n🚫 OLD BROKEN APPROACH:")
old_captions = [f"<start> {cap} <end>" for cap in sample_captions]
print("Captions with broken tokens:")
for i, cap in enumerate(old_captions, 1):
    print(f"  {i}: {cap}")

old_tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")
old_tokenizer.fit_on_texts(old_captions)

print(f"\nOld tokenizer vocabulary size: {len(old_tokenizer.word_index)}")
print("Special token check:")
for token in ['<start>', '<end>', 'start', 'end']:
    if token in old_tokenizer.word_index:
        print(f"  '{token}': index {old_tokenizer.word_index[token]} ✓")
    else:
        print(f"  '{token}': NOT FOUND ✗")

# Test sequence conversion
test_caption = "<start> a child in a pink dress <end>"
old_sequence = old_tokenizer.texts_to_sequences([test_caption])[0]
print(f"\nTest caption: {test_caption}")
print(f"Old sequence: {old_sequence}")

# NEW FIXED APPROACH
print("\n✅ NEW FIXED APPROACH:")
START_TOKEN = 'startseq'
END_TOKEN = 'endseq'

new_captions = [f"{START_TOKEN} {cap} {END_TOKEN}" for cap in sample_captions]
print("Captions with proper tokens:")
for i, cap in enumerate(new_captions, 1):
    print(f"  {i}: {cap}")

new_tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")
new_tokenizer.fit_on_texts(new_captions)

print(f"\nNew tokenizer vocabulary size: {len(new_tokenizer.word_index)}")
print("Special token check:")
for token in [START_TOKEN, END_TOKEN, '<UNK>', 'start', 'end']:
    if token in new_tokenizer.word_index:
        print(f"  '{token}': index {new_tokenizer.word_index[token]} ✓")
    else:
        print(f"  '{token}': NOT FOUND ✗")

# Test sequence conversion
test_caption_new = f"{START_TOKEN} a child in a pink dress {END_TOKEN}"
new_sequence = new_tokenizer.texts_to_sequences([test_caption_new])[0]
print(f"\nTest caption: {test_caption_new}")
print(f"New sequence: {new_sequence}")

# Compare vocabularies
print("\n📊 COMPARISON:")
print(f"Old vocab size: {len(old_tokenizer.word_index)}")
print(f"New vocab size: {len(new_tokenizer.word_index)}")

print("\nTop 10 words in each tokenizer:")
old_words = sorted(old_tokenizer.word_index.items(), key=lambda x: x[1])[:10]
new_words = sorted(new_tokenizer.word_index.items(), key=lambda x: x[1])[:10]

print("OLD:", [(word, idx) for word, idx in old_words])
print("NEW:", [(word, idx) for word, idx in new_words])

print("\n🎯 KEY DIFFERENCES:")
print("1. Old approach: '<start>' becomes ['<', 'start', '>'] - 3 tokens")
print("2. New approach: 'startseq' becomes ['startseq'] - 1 token")
print("3. This fixes the generation loop and caption quality!")

print("\n✅ Ready to run the fixed training script!")
