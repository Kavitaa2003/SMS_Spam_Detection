"""
Rule-based feature extraction for modern spam patterns (links, urgency
language, prize language, etc.) that word-based TF-IDF models tend to miss.
These are combined with the trained ML models' votes as extra input
features for the meta-model.
"""
import re
import re
import unicodedata

# Common cross-script lookalikes used to dodge keyword filters
# (Cyrillic/Greek letters that render almost identically to Latin ones).
HOMOGLYPH_MAP = str.maketrans({
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'і': 'i', 'ѕ': 's', 'ԁ': 'd', 'ℊ': 'g', '℮': 'e', '|': 'l', '0': 'o',
})


def normalize_text(text: str) -> str:
    # NFKC handles fullwidth/compatibility forms (Ｒ -> R, ｅ -> e, etc.)
    text = unicodedata.normalize('NFKC', text)
    # Then collapse known cross-script lookalikes to their ASCII equivalent
    return text.translate(HOMOGLYPH_MAP)

SHORTENER_DOMAINS = [
    'bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly',
    'is.gd', 'buff.ly', 'rebrand.ly', 'cutt.ly'
]

URGENCY_WORDS = [
    'urgent', 'immediately', 'verify', 'suspended', 'act now',
    'click here', 'claim', 'expire', 'limited time', 'confirm now'
]

PRIZE_WORDS = [
    'winner', 'won', 'prize', 'free', 'congratulations',
    'reward', 'gift', 'cash', 'selected'
]

# Matches explicit http(s)/www links, AND bare shortened-style links like
# "bit.ly/xyz123" that have no protocol prefix.
URL_PATTERN = re.compile(r'(https?://|www\.)\S+|\b\w+\.\w{2,}/\S+')
PHONE_PATTERN = re.compile(r'\b\d{5,}\b')  # 5+ consecutive digits: numbers/shortcodes

FEATURE_NAMES = [
    'has_url', 'has_shortened_link', 'has_urgency_words', 'has_prize_words',
    'has_phone_number', 'has_currency_symbol', 'has_excessive_punctuation',
    'has_excessive_caps'
]


def extract_features(message: str) -> list:
    text = normalize_text(message.lower())

    has_url = 1 if URL_PATTERN.search(text) else 0
    has_shortened_link = 1 if any(domain in text for domain in SHORTENER_DOMAINS) else 0
    has_urgency_words = 1 if any(word in text for word in URGENCY_WORDS) else 0
    has_prize_words = 1 if any(word in text for word in PRIZE_WORDS) else 0
    has_phone_number = 1 if PHONE_PATTERN.search(text) else 0
    has_currency_symbol = 1 if re.search(r'[$£€]', message) else 0

    exclamation_count = message.count('!')
    has_excessive_punctuation = 1 if exclamation_count >= 2 else 0

    letters = [c for c in message if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    has_excessive_caps = 1 if caps_ratio > 0.5 and len(letters) > 5 else 0

    return [
        has_url,
        has_shortened_link,
        has_urgency_words,
        has_prize_words,
        has_phone_number,
        has_currency_symbol,
        has_excessive_punctuation,
        has_excessive_caps,
    ]


def extract_features_dict(message: str) -> dict:
    return dict(zip(FEATURE_NAMES, extract_features(message)))
