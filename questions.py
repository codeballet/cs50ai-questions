import math
import nltk
import os
import sys
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = {}
    
    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            contents[file] = f.read()

    return contents

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens_list = word_tokenize(document)

    filtered_tokens_list = [
        word.lower() for word in tokens_list if 
        word not in string.punctuation and 
        word not in stopwords.words('english')
        ]

    return filtered_tokens_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    print(documents)

    # count how many documents there are
    documents_count = len(documents)
    print(f'length of documents: {documents_count}')

    # create a set of the words
    all_words = set()
    for words in documents.values():
        all_words.update(words)
    print(f'all_words: {all_words}')

    # find out how many documents each word appears in
    word_doc_count = {}
    for word in all_words:
        count = 0
        for document in documents:
            if word in documents[document]:
                count += 1
        word_doc_count[word] = count
    print(f'word_doc_count: {word_doc_count}')

    # calculate the inverse document frequency of each word as
    # the ln(number of documents / number of documents in which the word appears)
    idfs = {}
    for word, count in word_doc_count.items():
        idf = math.log(documents_count / count)
        idfs[word] = idf
    print(f'difs: {idfs}')

    # Return a dictionary mapping the words to their idf values


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
