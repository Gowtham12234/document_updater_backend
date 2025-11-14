# server/processors/summary_service.py
import re
import heapq
from collections import defaultdict

import nltk


def simple_word_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


# Ensure required NLTK resources are available (best-effort)
def ensure_nltk_resource(resource_name, resource_type='corpora'):
    """
    Try to find a resource; if missing, attempt to download it quietly.
    """
    try:
        nltk.data.find(f"{resource_type}/{resource_name}")
        # found
    except LookupError:
        try:
            print(f"[nltk] Resource '{resource_name}' not found. Attempting download...")
            nltk.download(resource_name, quiet=True)
            print(f"[nltk] Downloaded '{resource_name}'.")
        except Exception as e:
            print(f"[nltk] Failed to download '{resource_name}': {e}")

# Ensure punkt tokenizer (sent_tokenize) and stopwords are available (best-effort)
ensure_nltk_resource('punkt', 'tokenizers')
ensure_nltk_resource('stopwords', 'corpora')

# Fallback stopwords if NLTK stopwords are unavailable
FALLBACK_STOPWORDS = {
    'a','an','the','and','or','if','in','on','at','to','from','by','for',
    'is','are','was','were','be','been','of','that','this','these','those',
    'it','its','as','with','not','but','we','you','they','he','she','i','me',
    'my','your','our','their','them','so','do','does','did','have','has','had'
}

def get_stopwords():
    """
    Return a set of English stopwords. Tries NLTK first, then fallback set.
    """
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words('english'))
        if sw:
            return sw
    except Exception as e:
        # Attempted download earlier; if still failing, fallback
        print(f"[nltk] Could not load stopwords from NLTK (using fallback). Error: {e}")
    return FALLBACK_STOPWORDS

def clean_paragraph(text: str) -> str:
    """
    Post-process the joined paragraph:
    - Ensure there is a space after sentence-ending punctuation if missing.
    - Collapse multiple spaces into one.
    - Trim leading/trailing whitespace.
    """
    # Add space after a period, question mark, or exclamation if immediately followed by a letter
    text = re.sub(r'([.?!])(?=[A-Za-z0-9])', r'\1 ', text)
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(text: str, length: str = 'medium') -> str:
    """
    Generates an extractive summary of the text based on word frequency.

    Returns a single-paragraph summary (sentences joined by single spaces).
    """
    if not text or "Error" in text:
        return "Cannot summarize: Text extraction failed or document is empty."

    # Determine sentence count for summary length
    if length == 'short':
        target_sentences = 3
    elif length == 'long':
        target_sentences = 10
    else:  # medium
        target_sentences = 6

    try:
        sentences = simple_sent_tokenize(text)

    except Exception as e:
        # As a fallback, split by period if sent_tokenize fails
        print(f"[summarizer] sent_tokenize failed, falling back to split: {e}")
        sentences = [s.strip() for s in re.split(r'[.?!]\s*', text) if s.strip()]

    if not sentences:
        return "Document is too short or lacks substantive content."

    # 1. Create Frequency Table
    stop_words = get_stopwords()
    word_frequencies = defaultdict(int)

    for word in simple_word_tokenize(text.lower()):

        # consider words with alphanumeric characters, exclude stopwords
        if word not in stop_words and re.match(r'\w', word):
            word_frequencies[word] += 1

    if not word_frequencies:
        # Not enough content/meaningful tokens — return the first few sentences as a paragraph
        first_n = min(target_sentences, len(sentences))
        paragraph = ' '.join(sentences[:first_n])
        return clean_paragraph(paragraph)

    maximum_frequency = max(word_frequencies.values())
    if maximum_frequency == 0:
        maximum_frequency = 1

    # Normalize frequencies
    for word in list(word_frequencies.keys()):
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    # 2. Score Sentences
    sentence_scores = defaultdict(float)
    for idx, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[idx] += word_frequencies[word]

    # 3. Get Top Sentences (by score)
    num_sentences = min(target_sentences, len(sentences))

    if not sentence_scores:
        # fallback to first `num_sentences` if scoring produced nothing
        selected_indices = list(range(num_sentences))
    else:
        selected_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Reassemble summary in original sentence order
    summary_sentences = [sentences[i] for i in sorted(selected_indices)]

    # Join into a single paragraph and clean spacing/punctuation
    paragraph = ' '.join(summary_sentences)
    paragraph = clean_paragraph(paragraph)

    return paragraph
