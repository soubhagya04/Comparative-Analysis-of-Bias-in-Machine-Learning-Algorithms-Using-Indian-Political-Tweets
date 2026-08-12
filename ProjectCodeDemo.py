# ============================================================
#  COMPARATIVE ANALYSIS OF BIAS IN ML ALGORITHMS
#  Dataset : BJP vs Congress Tweets (Parth)
#  Kaggle  : kaggle.com/datasets/parth1902/bjp-vs-congress-tweet-dataset
#  Author  : Soubhagya Rout  |  Research Project 2026
# ============================================================

# ──────────────────────────────────────────────────────────────
# SECTION 0 : INSTALL DEPENDENCIES
# Run this cell once in Google Colab or your terminal
# ──────────────────────────────────────────────────────────────
# !pip install kaggle pandas numpy scikit-learn matplotlib seaborn
# !pip install nltk wordcloud textblob imbalanced-learn

# ──────────────────────────────────────────────────────────────
# SECTION 1 : UPLOAD DATASET  (TWO FILES — BJP + Congress)
# Upload both CSV files when prompted in Google Colab
# ──────────────────────────────────────────────────────────────

import os
import shutil

os.makedirs("data", exist_ok=True)

try:
    from google.colab import files

    print("📂 Step 1/2 — Upload the BJP tweets CSV file:")
    up1 = files.upload()
    bjp_file = list(up1.keys())[0]
    shutil.move(bjp_file, f"data/{bjp_file}")
    BJP_PATH = f"data/{bjp_file}"
    print(f"✅ BJP file saved : {BJP_PATH}\n")

    print("📂 Step 2/2 — Upload the Congress tweets CSV file:")
    up2 = files.upload()
    cong_file = list(up2.keys())[0]
    shutil.move(cong_file, f"data/{cong_file}")
    CONGRESS_PATH = f"data/{cong_file}"
    print(f"✅ Congress file saved : {CONGRESS_PATH}")

except Exception as e:
    # ── Fallback for local / non-Colab runs ──
    BJP_PATH     = "data/bjp_tweets.csv"      # ← change if needed
    CONGRESS_PATH = "data/congress_tweets.csv" # ← change if needed
    print(f"Not in Colab or upload failed: {e}")
    print(f"Using fallback paths:\n  BJP     : {BJP_PATH}\n  Congress: {CONGRESS_PATH}")


# ──────────────────────────────────────────────────────────────
# SECTION 2 : IMPORTS
# ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import re
import string
from collections import Counter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

# Imbalanced data
from imblearn.over_sampling import SMOTE

# Visualisation extras
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)          # needed by wordnet in newer NLTK
nltk.download("punkt_tab", quiet=True)         # needed by word_tokenize in Python 3.12

print("✅ All imports successful.")

# ──────────────────────────────────────────────────────────────
# SECTION 2.5 : GPU DETECTION & SETUP
# ──────────────────────────────────────────────────────────────
import subprocess, numpy as _np_gpu

def _check_gpu():
    """
    Detect GPU for LightGBM and XGBoost.
    LightGBM 4.x  → device='cuda'  (CUDA-native, best on Colab T4)
    LightGBM 3.x  → device='gpu'   (OpenCL fallback)
    XGBoost 2.0+  → device='cuda', tree_method='hist'
    Returns (lgb_device, xgb_device) — 'cuda'/'gpu'/'cpu'
    """
    _X = _np_gpu.random.rand(50, 5).astype("float32")
    _y = _np_gpu.random.randint(0, 2, 50)

    # ── Step 1: is a GPU physically present? ────────────────────
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print("   ⚠️  nvidia-smi failed → no GPU present")
            return "cpu", "cpu"
        # show first line (GPU name + memory)
        print(f"   🖥️  {r.stdout.splitlines()[8].strip()}")
    except Exception as e:
        print(f"   ⚠️  nvidia-smi error: {e} → no GPU present")
        return "cpu", "cpu"

    # ── Step 2: LightGBM — try CUDA first (LGB 4.x), then OpenCL ─
    lgb_device = "cpu"
    import lightgbm as _lgb
    for _dev in ("cuda", "gpu"):          # cuda = CUDA-native (T4 friendly)
        try:
            _lgb.train(
                {"device": _dev, "verbose": -1, "num_leaves": 4},
                _lgb.Dataset(_X, _y),
                num_boost_round=3,
            )
            lgb_device = _dev
            print(f"   ✅ LightGBM GPU: device='{_dev}' works")
            break
        except Exception as e:
            print(f"   ⚠️  LightGBM device='{_dev}' failed: {e}")

    # ── Step 3: XGBoost — CUDA + hist ───────────────────────────
    xgb_device = "cpu"
    import xgboost as _xgb
    try:
        _t = _xgb.XGBClassifier(device="cuda", tree_method="hist",
                                 n_estimators=2, verbosity=0)
        _t.fit(_X, _y)
        xgb_device = "cuda"
        print("   ✅ XGBoost  GPU: device='cuda' works")
    except Exception as e:
        print(f"   ⚠️  XGBoost device='cuda' failed: {e}")

    return lgb_device, xgb_device

print("\n🔍 Detecting GPU …")
_LGB_DEVICE, _XGB_DEVICE = _check_gpu()
_XGB_TREE = "hist"   # works for both CPU and GPU in XGBoost 2.0+

GPU_AVAILABLE = _LGB_DEVICE != "cpu" or _XGB_DEVICE != "cpu"
print(f"\n   ⚙️  LGB device={_LGB_DEVICE!r}  |  XGB device={_XGB_DEVICE!r}")
if not GPU_AVAILABLE:
    print("   💡 Tip: Runtime → Change runtime type → T4 GPU, then restart & run all")

# ──────────────────────────────────────────────────────────────
# SECTION 3 : LOAD & MERGE BOTH FILES
# ──────────────────────────────────────────────────────────────

def load_and_tag(path: str, party_name: str) -> pd.DataFrame:
    """Load a CSV and stamp every row with the party label."""
    tmp = pd.read_csv(path)
    tmp.columns = tmp.columns.str.strip().str.lower()

    # Normalise tweet-text column name
    for alt in ["tweet", "tweets", "content", "message"]:
        if alt in tmp.columns and "text" not in tmp.columns:
            tmp = tmp.rename(columns={alt: "text"})
            break

    # Normalise sentiment column name
    for alt in ["label", "polarity", "class", "target"]:
        if alt in tmp.columns and "sentiment" not in tmp.columns:
            tmp = tmp.rename(columns={alt: "sentiment"})
            break

    tmp["party"] = party_name
    print(f"  {party_name:10s} → {len(tmp):,} rows | columns: {tmp.columns.tolist()}")
    return tmp

print("Loading datasets …")
df_bjp     = load_and_tag(BJP_PATH,      "BJP")
df_congress = load_and_tag(CONGRESS_PATH, "Congress")

df = pd.concat([df_bjp, df_congress], ignore_index=True)

# Standardise sentiment to Title Case (Positive / Negative)
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()

# Drop rows missing essential columns
df = df.dropna(subset=["text", "sentiment", "party"]).reset_index(drop=True)

print(f"\nCombined Shape : {df.shape}")
print("\nColumns :", df.columns.tolist())
print("\nFirst 5 rows:")
df.head()


# ──────────────────────────────────────────────────────────────
# SECTION 4 : INITIAL DATA EXPLORATION (EDA)
# ──────────────────────────────────────────────────────────────

# ── 4.1  Basic info ──────────────────────────────────────────
print("=== Dataset Info ===")
print(df.info())
print("\n=== Missing Values ===")
print(df.isnull().sum())

# ── 4.2  Column check (already standardised in Section 3) ────
print(f"\nFinal columns : {df.columns.tolist()}")
print(f"Total rows    : {len(df):,}")

# ── 4.3  Party + Sentiment distribution ──────────────────────
print("\n=== Sentiment Counts ===")
print(df["sentiment"].value_counts())
print("\n=== Party Counts ===")
print(df["party"].value_counts())
print("\n=== Sentiment by Party ===")
print(df.groupby(["party", "sentiment"]).size().unstack(fill_value=0))

os.makedirs("outputs", exist_ok=True)

# ── Sentiment Label Mapping: 0 = Positive, 1 = Negative ───────────
# Dataset confirmed: labels are numeric 0 (Positive) and 1 (Negative)
LABEL_MAP = {
    0:   "Positive",
    1:   "Negative",
    "0": "Positive",
    "1": "Negative",
}
df["sentiment"] = df["sentiment"].map(LABEL_MAP).fillna(df["sentiment"].astype(str))
print(f"✅ Sentiment labels mapped: 0 → Positive  |  1 → Negative")
print(f"   Label distribution after mapping:")
print(df["sentiment"].value_counts().to_string())

# Side-by-side sentiment per party
fig, ax = plt.subplots(figsize=(8, 4))
sentiment_party = df.groupby(["party", "sentiment"]).size().unstack(fill_value=0)
sentiment_party.plot(kind="bar", ax=ax,
    color=["#d62728", "#2ca02c"])  # Negative=red, Positive=green
ax.set_title("Sentiment Distribution — BJP vs Congress", fontsize=14, fontweight="bold")
ax.set_xlabel("Party")
ax.set_ylabel("Number of Tweets")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("outputs/01_sentiment_by_party.png", dpi=150)
plt.show()
print("⚠️  NOTE: Class imbalance is itself a source of model bias!")




# ──────────────────────────────────────────────────────────────
# SECTION 4.4 : PARTY COLUMN ALREADY SET — BJP vs Congress
# ──────────────────────────────────────────────────────────────
# The 'party' column was stamped in Section 3 when files were loaded.
# No keyword detection needed.

print("=== Party Distribution ===")
print(df["party"].value_counts())

# Visualise party tweet counts
plt.figure(figsize=(6, 4))
ax = df["party"].value_counts().plot(
    kind="bar",
    color=["#FF6600", "#138808"]   # saffron=BJP, green=Congress
)
ax.set_title("Tweet Count by Party", fontsize=14, fontweight="bold")
ax.set_xlabel("Party")
ax.set_ylabel("Number of Tweets")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/01b_party_distribution.png", dpi=150)
plt.show()


# ──────────────────────────────────────────────────────────────
# SECTION 5 : FULL TWEET PREPROCESSING PIPELINE
# Pipeline:
#   Raw Tweet → Lowercase → Remove URLs/Mentions → Handle Hashtags
#   → Expand Contractions → Handle Emojis → Tokenize
#   → Remove Stopwords (EXCEPT negations) → Lemmatize
#   → Negation Handling → Final Clean Text
# ──────────────────────────────────────────────────────────────
import html


# ── STEP A: Contraction Expansion Dictionary ─────────────────────
CONTRACTIONS = {
    "don't": "do not",      "doesn't": "does not",  "didn't": "did not",
    "won't": "will not",    "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not","can't": "cannot",       "isn't": "is not",
    "aren't": "are not",    "wasn't": "was not",     "weren't": "were not",
    "haven't": "have not",  "hasn't": "has not",     "hadn't": "had not",
    "I'm": "i am",          "i'm": "i am",           "you're": "you are",
    "they're": "they are",  "we're": "we are",       "he's": "he is",
    "she's": "she is",      "it's": "it is",         "that's": "that is",
    "there's": "there is",  "here's": "here is",     "who's": "who is",
    "what's": "what is",    "i've": "i have",        "you've": "you have",
    "we've": "we have",     "they've": "they have",  "i'd": "i would",
    "you'd": "you would",   "he'd": "he would",      "she'd": "she would",
    "we'd": "we would",     "they'd": "they would",  "i'll": "i will",
    "you'll": "you will",   "he'll": "he will",      "she'll": "she will",
    "we'll": "we will",     "they'll": "they will",  "let's": "let us",
    "it'd": "it would",     "that'd": "that would",  "mustn't": "must not",
    "mightn't": "might not","needn't": "need not",   "shan't": "shall not",
    "daren't": "dare not",  "mayn't": "may not",     "oughtn't": "ought not",
    "'re": " are",          "'ve": " have",          "'ll": " will",
    "'d": " would",         "'m": " am",
}

# Build case-insensitive regex for contraction matching
_CONTRACTION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

def expand_contractions(text: str) -> str:
    """Expand English contractions (don't → do not, won't → will not)."""
    def _replace(match):
        token = match.group(0)
        return CONTRACTIONS.get(token.lower(), CONTRACTIONS.get(token, token))
    return _CONTRACTION_RE.sub(_replace, text)

# ── STEP B: Stopwords — PRESERVE negations ───────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Negation words that must NOT be removed from stopwords
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "none", "cannot", "without", "hardly", "barely", "scarcely",
}
stop_words -= NEGATION_WORDS   # ← ensure negations survive stopword removal

# Extended Hinglish / political noise stopwords
CUSTOM_STOPWORDS = {
    "hai", "hain", "ki", "ke", "ka", "se", "ko", "ne", "par",
    "aur", "yeh", "ye", "nahi", "nahin", "kya", "bhi", "jo",
    "woh", "toh", "mein", "ek", "ho", "koi", "ab", "tha", "thi",
    "hum", "aap", "unka", "unke", "yahi", "kuch", "apna", "apne",
    "agar", "lekin", "phir", "sab", "bas", "kaisa", "bahut",
    "via", "amp", "rt", "gt", "lt",          # Twitter artifacts
    "said", "say", "like", "just", "get", "one", "will", "new",
}
stop_words.update(CUSTOM_STOPWORDS)


# Punctuation boundary set for negation scope already removed —
# tweets rarely have punctuation so we use a fixed WINDOW instead.

def apply_negation_handling(tokens: list) -> list:
    """
    Window-based Negation Handling (tweet-optimised):
    Tweets rarely contain punctuation, so boundary-based negation
    spreads too far. Instead, mark only the next 3 tokens after a
    negation word with the '_NEG' suffix.
    Example: ["not", "good", "idea", "today"] → ["not", "good_NEG", "idea_NEG", "today"]
    """
    result = []
    negating = False
    window = 0

    for token in tokens:
        if token in NEGATION_WORDS:
            negating = True
            window = 3              # affect only next 3 tokens
            result.append(token)
        elif negating and window > 0:
            result.append(token + "_NEG")
            window -= 1
            if window == 0:
                negating = False
        else:
            negating = False
            result.append(token)

    return result


def clean_tweet(text: str, apply_negation: bool = True) -> str:
    """
    Full tweet preprocessing pipeline:

      Raw Tweet
        ↓ Lowercase
        ↓ Remove URLs, mentions
        ↓ Handle hashtags (keep words)
        ↓ Expand contractions (don't → do not)
        ↓ Tokenize
        ↓ Negation handling  ← BEFORE stopword removal so negated
        ↓ Stopword removal      words (not_good) are seen whole
        ↓ Lemmatization
        ↓ Final Clean Text
    """
    # ── Pre-step: HTML entity decoding (&amp; → &) ───────────────
    text = html.unescape(str(text))

    # ── Step 1: Lowercase ────────────────────────────────────────
    text = text.lower()

    # ── Step 2: Remove URLs and @mentions ───────────────────────
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)                        # @mentions
    text = re.sub(r"^rt[\s:]+", "", text)                   # RT prefix

    # ── Step 3: Handle hashtags (keep the word, strip #) ────────
    text = re.sub(r"#(\w+)", r"\1", text)

    # ── Step 4: Expand contractions (don't → do not) ────────────
    text = expand_contractions(text)

    # ── Step 5: Remove non-ASCII characters (Devanagari, Arabic etc.) ─
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Squish elongated words (soooo → soo, greaaaat → greaat)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove punctuation
    punct_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punct_to_remove))
    text = text.replace("'", "")

    # Remove standalone digits
    text = re.sub(r"\b\d+\b", "", text)

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # ── Step 6: Tokenize ─────────────────────────────────────────
    tokens = word_tokenize(text)

    # ── Step 7: Negation handling (window=3, tweet-optimised) ────
    # Run BEFORE stopword removal so negation words are still present
    if apply_negation:
        tokens = apply_negation_handling(tokens)

    # ── Step 8: Remove stopwords (EXCEPT negation words) ────────
    # len > 1 to keep short but meaningful words like "no", "ok", "go"
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # ── Step 9: Lemmatization ────────────────────────────────────
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # ── Step 10: Final clean text ────────────────────────────────
    return " ".join(tokens)


print("Cleaning tweets … (may take 1-2 minutes for large datasets)")
df["clean_text"] = df["text"].apply(clean_tweet)

# Drop rows where cleaning left an empty string
before = len(df)
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
print(f"✅ Preprocessing complete. Dropped {before - len(df)} empty rows.")
print(f"   Remaining tweets : {len(df):,}")

print("\nSample cleaning comparison (3 tweets):")
for i in range(min(3, len(df))):
    print(f"  [{i+1}] ORIGINAL : {df['text'].iloc[i][:80]}")
    print(f"      CLEANED  : {df['clean_text'].iloc[i][:80]}\n")

# ── Pipeline trace: show each step on a single example tweet ────
EXAMPLE = df["text"].iloc[0] if len(df) > 0 else "I don't think BJP won't win! #Elections2024 @user https://t.co/abc 😊"
print("\n" + "─"*60)
print("📋 PREPROCESSING PIPELINE TRACE (on one example tweet):")
print("─"*60)
_ex = str(EXAMPLE)
print(f" Raw          : {_ex[:100]}")
_ex = html.unescape(_ex)
_ex = _ex.lower()
print(f" Lowercase    : {_ex[:100]}")
_ex = re.sub(r"http\S+|www\S+|https\S+", "", _ex)
_ex = re.sub(r"@\w+", "", _ex)
_ex = re.sub(r"^rt[\s:]+", "", _ex)
print(f" No URLs/Ment : {_ex[:100]}")
_ex = re.sub(r"#(\w+)", r"\1", _ex)
print(f" Hashtags     : {_ex[:100]}")
_ex = expand_contractions(_ex)
print(f" Contractions : {_ex[:100]}")

_ex = re.sub(r"[^\x00-\x7F]+", " ", _ex)
_ex = re.sub(r"(.)\1{{2,}}", r"\1\1", _ex)
_punct = string.punctuation.replace("'", "")
_ex = _ex.translate(str.maketrans("", "", _punct)).replace("'", "")
_ex = re.sub(r"\b\d+\b", "", _ex)
_ex = re.sub(r"\s+", " ", _ex).strip()
_toks = word_tokenize(_ex)
print(f" Tokenized    : {_toks[:12]} ...")
_toks_sw = [t for t in _toks if t not in stop_words and len(t) > 2]
print(f" No Stopwords : {_toks_sw[:12]} ...")
_toks_lem = [lemmatizer.lemmatize(t) for t in _toks_sw]
print(f" Lemmatized   : {_toks_lem[:12]} ...")
_toks_neg = apply_negation_handling(_toks_lem)
print(f" Neg Handled  : {_toks_neg[:12]} ...")
print(f" Final Text   : {' '.join(_toks_neg)[:100]}")
print("─"*60)



# ──────────────────────────────────────────────────────────────
# SECTION 6 : SENTIMENT LABELS (already present in BJP vs Congress Tweets dataset)
# ──────────────────────────────────────────────────────────────
# The dataset already has a 'sentiment' column — no TextBlob needed.

print("\n=== Sentiment Distribution ===")
print(df["sentiment"].value_counts())

# Visualise overall sentiment distribution
fig, ax = plt.subplots(figsize=(7, 4))
df["sentiment"].value_counts().plot(
    kind="bar", ax=ax,
    color=["#d62728", "#2ca02c"]   # Negative=red, Positive=green
)
ax.set_title("Overall Sentiment Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Tweet Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/02_sentiment_distribution.png", dpi=150)
plt.show()


# ──────────────────────────────────────────────────────────────
# SECTION 7 : WORD CLOUDS PER SENTIMENT CLASS
# ──────────────────────────────────────────────────────────────

sentiment_classes = df["sentiment"].unique()
fig, axes = plt.subplots(1, len(sentiment_classes), figsize=(6 * len(sentiment_classes), 5))
if len(sentiment_classes) == 1:
    axes = [axes]

for ax, sent in zip(axes, sentiment_classes):
    corpus = " ".join(df[df["sentiment"] == sent]["clean_text"])
    if not corpus.strip():
        ax.axis("off")
        continue
    wc = WordCloud(width=500, height=350, background_color="white",
                   max_words=80, colormap="Set2").generate(corpus)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(sent, fontsize=13, fontweight="bold")

plt.suptitle("Top Words per Sentiment Class", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/03_wordclouds.png", dpi=150, bbox_inches="tight")
plt.show()


# ──────────────────────────────────────────────────────────────
# SECTION 8 : FEATURE ENGINEERING — TF-IDF + CLASS IMBALANCE FIX
# ──────────────────────────────────────────────────────────────

# ── 8.1  Encode target label ──────────────────────────────────
le_sentiment = LabelEncoder()
df["label"] = le_sentiment.fit_transform(df["sentiment"])

X = df["clean_text"]
y = df["label"]

# ── 8.2  Check class balance BEFORE fix ───────────────────────
from collections import Counter
class_counts = Counter(y)
print("\n=== Class Distribution BEFORE balancing ===")
for cls_id, count in sorted(class_counts.items()):
    cls_name = le_sentiment.inverse_transform([cls_id])[0]
    print(f"  {cls_name:12s}: {count:,} tweets")

imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
print(f"  Imbalance ratio : {imbalance_ratio:.2f}x")
if imbalance_ratio > 1.5:
    print("  ⚠️  Significant imbalance detected — SMOTE will be applied.")
else:
    print("  ✅  Classes are reasonably balanced.")

# Visualise sentiment class balance
plt.figure(figsize=(6, 3))
class_names = [le_sentiment.inverse_transform([i])[0] for i in sorted(class_counts)]
class_vals  = [class_counts[i] for i in sorted(class_counts)]
plt.bar(class_names, class_vals, color=["#d62728", "#2ca02c"])  # Negative=red, Positive=green
plt.title("Sentiment Class Distribution Before Balancing", fontsize=12, fontweight="bold")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ── 8.2b  PARTY IMBALANCE CHECK ───────────────────────────────
# ⚠️  If BJP has more tweets than Congress (or vice versa), the model
# learns more BJP language patterns → performs better on BJP tweets
# → inflated cross-party bias scores. We diagnose and report this.
print("\n=== Party Tweet Count (Imbalance Check) ===")
party_counts = df["party"].value_counts()
print(party_counts.to_string())

party_max = party_counts.max()
party_min = party_counts.min()
party_ratio = party_max / party_min
print(f"\n  Majority party : {party_counts.idxmax()}  ({party_max:,} tweets)")
print(f"  Minority party : {party_counts.idxmin()}  ({party_min:,} tweets)")
print(f"  Party imbalance ratio : {party_ratio:.2f}x")

if party_ratio > 1.5:
    print()
    print("  ⚠️  WARNING: Party imbalance detected!")
    print(f"  The model will see {party_ratio:.1f}x more {party_counts.idxmax()} tweets during training.")
    print("  This can cause the model to learn BJP/Congress vocabulary unevenly,")
    print("  leading to biased cross-party accuracy scores in Section 10.4.")
    print("  ✅ FIX applied: train/test split is stratified on party+sentiment combined.")
else:
    print("  ✅ Party counts are reasonably balanced — no extra fix needed.")

# Visualise party imbalance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: raw counts
party_counts.plot(kind="bar", ax=axes[0], color=["#FF6600", "#138808"], edgecolor="white")
axes[0].set_title("Tweet Count per Party", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Number of Tweets")
axes[0].set_xlabel("Party")
axes[0].tick_params(axis="x", rotation=0)
for i, v in enumerate(party_counts):
    axes[0].text(i, v + 20, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

# Right: sentiment split per party (stacked bar)
party_sent = df.groupby(["party", "sentiment"]).size().unstack(fill_value=0)
party_sent.plot(kind="bar", stacked=True, ax=axes[1],
                color=["#d62728", "#2ca02c"], edgecolor="white")
axes[1].set_title("Sentiment Split per Party", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Number of Tweets")
axes[1].set_xlabel("Party")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Sentiment")

plt.suptitle("Party Imbalance Diagnostic", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/00_party_imbalance_check.png", dpi=150)
plt.show()

# ── 8.3  Train / test split (stratified on party + sentiment) ────
# Using a combined strata key ensures BOTH parties AND both sentiment
# classes are proportionally represented in train and test sets.
# This is critical when BJP has more tweets than Congress — without
# combined stratification, the test set could be BJP-heavy, making
# cross-party bias numbers unreliable.
party_col = df["party"].values
strata = pd.Series(
    [f"{p}|{s}" for p, s in zip(party_col, y)],
    index=df.index
)

X_train, X_test, y_train, y_test, strata_train, strata_test = train_test_split(
    X, y, strata, test_size=0.20, random_state=42, stratify=strata
)

print("\n=== Train/Test Split — Party Distribution ===")
print("Train:")
print(df.loc[X_train.index, "party"].value_counts().to_string())
print("Test:")
print(df.loc[X_test.index, "party"].value_counts().to_string())

# ── 8.4  Word + Character TF-IDF (FeatureUnion) ──────────────
# Word TF-IDF captures full-word patterns (unigrams + bigrams)
# Char TF-IDF captures spelling variation, abbreviations, informal
# text — very high impact for noisy tweet data (+3-5% accuracy)
from sklearn.pipeline import FeatureUnion

word_tfidf = TfidfVectorizer(
    max_features=8000,        # top word features
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
)

char_tfidf = TfidfVectorizer(
    analyzer="char",          # character-level
    ngram_range=(3, 5),       # 3-to-5 char n-grams
    max_features=3000,        # top char features
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
)

tfidf = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf),
])

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"\nFeatureUnion TF-IDF — Train: {X_train_vec.shape}  |  Test: {X_test_vec.shape}")
print(f"  Word features : {word_tfidf.max_features:,}")
print(f"  Char features : {char_tfidf.max_features:,}")

# ── 8.5  SMOTE to fix class imbalance ────────────────────────
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("\nApplying SMOTE to balance training classes …")
smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts.values()) - 1))
X_train_bal, y_train_bal = smote.fit_resample(X_train_vec, y_train)

bal_counts = Counter(y_train_bal)
print("=== Class Distribution AFTER SMOTE ===")
for cls_id, count in sorted(bal_counts.items()):
    cls_name = le_sentiment.inverse_transform([cls_id])[0]
    print(f"  {cls_name:12s}: {count:,} tweets")
print("✅ SMOTE complete — training set is now balanced.")

# Visualise after balancing
plt.figure(figsize=(6, 3))
bal_names = [le_sentiment.inverse_transform([i])[0] for i in sorted(bal_counts)]
bal_vals  = [bal_counts[i] for i in sorted(bal_counts)]
plt.bar(bal_names, bal_vals, color=["#d62728", "#2ca02c"])  # Negative=red, Positive=green
plt.title("Class Distribution After SMOTE", fontsize=12, fontweight="bold")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



# ──────────────────────────────────────────────────────────────
# SECTION 9 : TRAIN MULTIPLE ML MODELS
# ──────────────────────────────────────────────────────────────

# ── Best / recommended params (no tuning) ───────────────────────
BEST_SVM_C    = 0.5
BEST_LR_C     = 5
BEST_NB_ALPHA = 0.01

BEST_LGB_PARAMS = {
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "num_leaves"       : 60,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "bagging_freq"     : 1,
}

print("✅ Using recommended params — no tuning step.")


# ── MODELS dict ──────────────────────────────────────────────────
MODELS = {
    "Linear SVM": LinearSVC(
        C=BEST_SVM_C,
        max_iter=5000,
        dual=True,
        class_weight="balanced",
        random_state=42,
    ),

    "Logistic Regression": LogisticRegression(
        C=BEST_LR_C,
        max_iter=3000,
        class_weight="balanced",
        solver="saga",
        random_state=42,
        n_jobs=-1,
    ),

    "Naive Bayes": MultinomialNB(
        alpha=BEST_NB_ALPHA,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

MODELS["LightGBM"] = LGBMClassifier(
    **BEST_LGB_PARAMS,
    class_weight="balanced",
    device=_LGB_DEVICE,
    random_state=42,
    verbose=-1,
)
print(f"✅ LightGBM added to MODELS  (device={_LGB_DEVICE!r})")

# ── Training & Evaluation loop ───────────────────────────────────
results      = {}
per_party_f1 = {}

print("\n" + "="*65)
print("         MODEL TRAINING & EVALUATION")
print("="*65)

for name, model in MODELS.items():
    print(f"\n▶  Training : {name}")
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_vec)

    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")

    results[name] = {
        "Accuracy"          : round(acc,  4),
        "Weighted F1"       : round(f1_w, 4),
        "Weighted Precision": round(prec, 4),
        "Weighted Recall"   : round(rec,  4),
    }

    report_dict  = classification_report(y_test, y_pred,
                                         target_names=le_sentiment.classes_,
                                         output_dict=True)
    per_party_f1[name] = {cls: report_dict[cls]["f1-score"]
                          for cls in le_sentiment.classes_}

    print(f"   Accuracy : {acc:.4f}  |  Weighted F1 : {f1_w:.4f}")
    print(classification_report(y_test, y_pred, target_names=le_sentiment.classes_))


# ──────────────────────────────────────────────────────────────
# SECTION 10 : BIAS ANALYSIS — PER-PARTY F1 GAPS
# ──────────────────────────────────────────────────────────────
"""
BIAS DEFINITION used in this research:
  A model is biased if it performs significantly better for one
  sentiment class (or political-party subset) compared to another.
  We measure this as:
    - Per-class F1 gap  = max(F1) − min(F1) across classes per model
    - Cross-party F1    = train/test on each party separately
"""

print("\n" + "="*65)
print("         BIAS ANALYSIS : PER-SENTIMENT F1 SCORES")
print("="*65)

# ── 10.1  Heatmap of per-class F1 ─────────────────────────────
f1_df = pd.DataFrame(per_party_f1).T  # rows=models, cols=sentiment classes
print(f1_df.round(4))

plt.figure(figsize=(8, 4))
sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, linewidths=0.5, cbar_kws={"label": "F1 Score"})
plt.title("Per-Class F1 Score by Model\n(Bias visible as uneven rows)", fontsize=13, fontweight="bold")
plt.xlabel("Sentiment Class")
plt.ylabel("ML Model")
plt.tight_layout()
plt.savefig("outputs/04_bias_heatmap.png", dpi=150)
plt.show()

# ── 10.2  F1 gap (bias magnitude) per model ───────────────────
f1_df["F1_Gap"] = f1_df[le_sentiment.classes_].max(axis=1) - f1_df[le_sentiment.classes_].min(axis=1)
print("\n=== F1 Gap (Bias Magnitude — lower is fairer) ===")
print(f1_df[["F1_Gap"]].sort_values("F1_Gap"))

plt.figure(figsize=(7, 4))
f1_df["F1_Gap"].sort_values().plot(kind="barh", color="#e07b39")
plt.axvline(0.05, color="red", linestyle="--", linewidth=1, label="5% threshold")
plt.title("F1 Gap per Model (Bias Indicator)", fontsize=13, fontweight="bold")
plt.xlabel("Max F1 − Min F1  (lower = less bias)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/05_f1_gap.png", dpi=150)
plt.show()


# ── 10.3  Cross-class bias: accuracy per sentiment class per model ─
"""
For each sentiment class, we test how well each model predicts
that class.  A large accuracy gap across classes = model bias.
"""
print("\n=== Cross-Class Bias (Accuracy per Sentiment Class per Model) ===")

class_bias = {}   # {sentiment_class: {model: accuracy}}

for cls in le_sentiment.classes_:
    cls_label  = le_sentiment.transform([cls])[0]
    mask       = df["label"] == cls_label
    X_c        = tfidf.transform(df.loc[mask, "clean_text"])
    y_c        = df.loc[mask, "label"].values
    class_bias[cls] = {}
    for name, model in MODELS.items():
        y_pred_c = model.predict(X_c)
        class_bias[cls][name] = round(accuracy_score(y_c, y_pred_c), 4)

class_bias_df = pd.DataFrame(class_bias).T   # rows=classes, cols=models
print(class_bias_df)

plt.figure(figsize=(9, 4))
class_bias_df.plot(kind="bar", figsize=(9, 4), edgecolor="white")
plt.title("Model Accuracy per Sentiment Class\n(Uneven bars = class-level bias)",
          fontsize=13, fontweight="bold")
plt.xlabel("Sentiment Class")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("outputs/06_cross_class_bias.png", dpi=150)
plt.show()


# ── 10.4  Cross-PARTY bias (if dataset has a 'party' column) ──
"""
For each political party, we test how well models recognise
sentiment in THAT party's tweets. A large accuracy gap across
parties = the model is biased towards/against certain parties.
"""

if "party" in df.columns:
    parties = df["party"].unique()
    print(f"\n=== Cross-Party Bias — found {len(parties)} parties: {list(parties)} ===")

    party_bias = {}   # {party: {model_name: accuracy}}

    for party in parties:
        mask  = df["party"] == party
        X_p   = tfidf.transform(df.loc[mask, "clean_text"])
        y_p   = df.loc[mask, "label"].values
        party_bias[party] = {}
        for name, model in MODELS.items():
            y_pred_p = model.predict(X_p)
            party_bias[party][name] = round(accuracy_score(y_p, y_pred_p), 4)

    party_bias_df = pd.DataFrame(party_bias).T   # rows=parties, cols=models
    print(party_bias_df)

    # ── Heatmap: easier to spot which party is most disadvantaged ──
    plt.figure(figsize=(10, max(4, len(parties) * 0.6)))
    sns.heatmap(party_bias_df, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.4, vmax=1.0, linewidths=0.5,
                cbar_kws={"label": "Accuracy"})
    plt.title("Model Accuracy per Political Party\n(Low score = model is biased AGAINST that party)",
              fontsize=13, fontweight="bold")
    plt.xlabel("ML Model")
    plt.ylabel("Political Party")
    plt.tight_layout()
    plt.savefig("outputs/06b_party_bias_heatmap.png", dpi=150)
    plt.show()

    # ── Bar chart: side-by-side per party ─────────────────────────
    party_bias_df.plot(kind="bar", figsize=(11, 5), edgecolor="white")
    plt.title("Model Accuracy per Political Party\n(Uneven bars = cross-party bias)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Political Party")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/06c_party_bias_bar.png", dpi=150)
    plt.show()

    # ── Bias magnitude per party ──────────────────────────────────
    party_bias_df["Accuracy_Range"] = (
        party_bias_df[list(MODELS.keys())].max(axis=1) -
        party_bias_df[list(MODELS.keys())].min(axis=1)
    )
    print("\n=== Party Bias Magnitude (Max − Min accuracy across models) ===")
    print("Higher = models disagree MORE on that party → stronger bias signal")
    print(party_bias_df[["Accuracy_Range"]].sort_values("Accuracy_Range", ascending=False))

    party_bias_df.to_csv("outputs/cross_party_bias.csv")
    print("\n✅ Saved: outputs/cross_party_bias.csv")

else:
    print("\n⚠️  No 'party' column found in the dataset.")
    print("   To enable cross-party bias analysis, add a 'party' column to your CSV,")
    print("   e.g.:  df['party'] = df['username'].map({'BJP_handle': 'BJP', ...})")
    print("   Then re-run this section.")


# ──────────────────────────────────────────────────────────────
# SECTION 11 : CONFUSION MATRICES
# ──────────────────────────────────────────────────────────────

_n_models = len(MODELS)
_ncols = 3
_nrows = (_n_models + _ncols - 1) // _ncols   # enough rows for all models
fig, axes = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 5 * _nrows))
axes = axes.flatten()

for ax, (name, model) in zip(axes, MODELS.items()):
    y_pred = model.predict(X_test_vec)
    cm     = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le_sentiment.classes_,
                yticklabels=le_sentiment.classes_)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Hide any spare axes
for ax in axes[_n_models:]:
    ax.set_visible(False)

plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/07_confusion_matrices.png", dpi=150)
plt.show()


# ──────────────────────────────────────────────────────────────
# SECTION 12 : OVERALL COMPARISON TABLE
# ──────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results).T
results_df["F1_Gap"] = f1_df["F1_Gap"]

print("\n" + "="*70)
print("         FINAL COMPARISON TABLE")
print("="*70)
print(results_df.to_string())

# Bar chart — weighted F1 comparison
_palette = ["#d62728","#ff7f0e","#2ca02c","#1f77b4",
            "#9467bd","#8c564b","#e377c2","#17becf"]
_f1_sorted = results_df["Weighted F1"].sort_values()
_bar_colors = _palette[:len(_f1_sorted)]
ax = _f1_sorted.plot(
    kind="barh",
    figsize=(9, max(4, len(_f1_sorted) * 0.7)),
    color=_bar_colors,
)
plt.title("Model Comparison — Weighted F1 Score", fontsize=13, fontweight="bold")
plt.xlabel("Weighted F1 Score")
plt.xlim(0, 1)
for i, v in enumerate(_f1_sorted):
    ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/08_model_comparison_f1.png", dpi=150)
plt.show()


# ──────────────────────────────────────────────────────────────
# SECTION 13 : BIAS MITIGATION EXPERIMENT — CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────
"""
One simple bias mitigation strategy for imbalanced classes:
  → Use class_weight='balanced' in Logistic Regression & SVM
  → Compare F1 gap before and after
"""
print("\n=== Bias Mitigation: Balanced Class Weights ===")

mitigated_models = {
    "LR (balanced)"  : LogisticRegression(
        C=2, max_iter=2000, class_weight="balanced",
        solver="liblinear", random_state=42
    ),
    "SVM (balanced)" : LinearSVC(
        C=BEST_SVM_C, max_iter=5000, class_weight="balanced", random_state=42
    ),
}

for name, model in mitigated_models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    rep    = classification_report(y_test, y_pred,
                                   target_names=le_sentiment.classes_, output_dict=True)
    class_f1s = [rep[cls]["f1-score"] for cls in le_sentiment.classes_]
    gap = max(class_f1s) - min(class_f1s)
    print(f"{name:20s}  |  Weighted F1 = {f1_score(y_test, y_pred, average='weighted'):.4f}"
          f"  |  F1 Gap = {gap:.4f}")

print("\n✅ Lower F1 Gap after mitigation = less bias!")


# ──────────────────────────────────────────────────────────────
# SECTION 14 : SAVE RESULTS TO CSV
# ──────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

results_df.to_csv("outputs/model_comparison_results.csv")
class_bias_df.to_csv("outputs/cross_class_bias.csv")
f1_df.to_csv("outputs/per_class_f1_and_gap.csv")

print("\n✅ All results saved to outputs/ folder.")
print("   Files:")
print("   - model_comparison_results.csv")
print("   - cross_class_bias.csv")
print("   - per_class_f1_and_gap.csv")
print("   - 01_sentiment_distribution.png")
print("   - 02_sentiment_distribution.png")
print("   - 03_wordclouds.png")
print("   - 04_bias_heatmap.png")
print("   - 05_f1_gap.png")
print("   - 06_cross_party_bias.png")
print("   - 07_confusion_matrices.png")
print("   - 08_model_comparison_f1.png")


# ──────────────────────────────────────────────────────────────
# SECTION 15 : RESEARCH PAPER METRICS SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  RESEARCH PAPER — KEY FINDINGS SUMMARY")
print("="*70)
print(f"  Total tweets analysed   : {len(df):,}")
print(f"  Sentiment classes       : {', '.join(le_sentiment.classes_)}")
print(f"  Train set size          : {X_train_vec.shape[0]:,}")
print(f"  Test set size           : {X_test_vec.shape[0]:,}")
print(f"  TF-IDF features         : {X_train_vec.shape[1]:,}")
print()

best_model = results_df["Weighted F1"].idxmax()
least_bias = results_df["F1_Gap"].idxmin()
most_bias  = results_df["F1_Gap"].idxmax()

print(f"  Best overall performer  : {best_model}  (F1={results_df.loc[best_model,'Weighted F1']:.4f})")
print(f"  Least biased model      : {least_bias}  (F1 gap={results_df.loc[least_bias,'F1_Gap']:.4f})")
print(f"  Most biased model       : {most_bias}  (F1 gap={results_df.loc[most_bias,'F1_Gap']:.4f})")
print()
print("  → Use these numbers directly in your Results section!")
print("="*70)