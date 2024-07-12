import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Additional downloads for handling contractions
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    # For simplicity, this example manually handles one contraction type
    # a library like contractions will be used moving ahead to handle this more comprehensively
    text = text.replace("we'll", "we will")
    
    return text

def extract_keywords(text):
    text = preprocess_text(text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    
    # Perform part-of-speech tagging
    tagged_tokens = pos_tag(filtered_tokens)
    
    # Focus on nouns and adjectives
    focused_tokens = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('JJ')]
    
    # Calculate frequency distribution
    freq_dist = FreqDist(focused_tokens)
    
    # Select keywords based on frequency
    keywords = [word for word, freq in freq_dist.most_common(5)]
    
    return keywords

# testing
text = "Let's go for hiking this weekend and we'll have dinner outside."
keywords = extract_keywords(text)
print(keywords)