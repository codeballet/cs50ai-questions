# Questions

## Introduction

Questions is an AI that can answer questions.

Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. Among the more famous question answering systems is Watson, the IBM computer that competed (and won) on Jeopardy! A question answering system of Watson’s accuracy requires enormous complexity and vast amounts of data. For this project, I am designing a simpler question answering system based on an inverse document frequency algorithm.

The question answering system performs two tasks: document retrieval and passage retrieval. The system must have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval first identifies which document(s) are most relevant to the query. Once the top documents are found, the top document(s) are be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

How to find the most relevant documents and passages? To find the most relevant documents, the AI is using tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once the AI has found the most relevant documents, it is using a combination of inverse document frequency and a query term density measure in order to find the most appropriate passage.

More sophisticated question answering systems might employ other strategies (analyzing the type of question word used, looking for synonyms of query words, lemmatizing to handle different forms of the same word, etc.). Such improvements may be added to the project later.

## How to run the AI

In order to run the AI, make sure you have a corpus of text documents available. For testing purposes, there are two corpora included in the repository: `corpus`, which contains a collection of text files on artificial intelligence gathered from wikipedia; and `my_corpus`, which contains two very short nonsense text files, used in the early stages of the development of the AI.

Once a corpus is made available, start the AI by entering the command:

```
python questions.py <path_to_corpus>
```

For instance, if you want to use the included `corpus`, enter:

```
python questions.py corpus
```

## Expanding or changing the working knowledge of the AI

You may create your own corpus of `.txt` documents as you like.

## Intellectual Property Rights

MIT

## Acknowledgements

The project was created as part of Harvard's CS50 Introduction to Artificial Intelligence with Python.
