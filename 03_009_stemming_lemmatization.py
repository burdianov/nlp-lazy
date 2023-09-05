import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# call only once
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

porter = PorterStemmer()

porter.stem("walking")
porter.stem("walked")
porter.stem("walks")
porter.stem("ran")
porter.stem("running")
porter.stem("bosses")
porter.stem("replacement")

sentence = "Lemmatization is more sophisticated than stemming".split()

for token in sentence:
    print(porter.stem(token))

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("mice")
lemmatizer.lemmatize("walking")
lemmatizer.lemmatize("walking", pos=wordnet.VERB)

lemmatizer.lemmatize("going")
lemmatizer.lemmatize("going", pos=wordnet.VERB)

lemmatizer.lemmatize("mice")


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


sentence = "Donald Trump has a devoted following".split()

words_and_tags = nltk.pos_tag(sentence)

for word, tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
    print(lemma, end=" ")

sentence = "The cat was following the bird as it flew by".split()

words_and_tags = nltk.pos_tag(sentence)

for word, tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
    print(lemma, end=" ")
