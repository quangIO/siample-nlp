#!/usr/bin/python

from typing import List, Dict, TextIO, Set
from collections import Counter
from math import log
from numpy import dot
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


closed_class_stop_words = {'a', 'the', 'an', 'and', 'or', 'but', 'about', 'above', 'after', 'along', 'amid', 'among',
                           'as', 'at', 'by', 'for', 'from', 'in', 'into', 'like', 'minus', 'near', 'of', 'off', 'on',
                           'onto', 'out', 'over', 'past', 'per', 'plus', 'since', 'till', 'to', 'under', 'until', 'up',
                           'via', 'vs', 'with', 'that', 'can', 'cannot', 'could', 'may', 'might', 'must', 'need',
                           'ought', 'shall', 'should', 'will', 'would', 'have', 'had', 'has', 'having', 'be', 'is',
                           'am', 'are', 'was', 'were', 'being', 'been', 'get', 'gets', 'got', 'gotten', 'getting',
                           'seem', 'seeming', 'seems', 'seemed', 'enough', 'both', 'all', 'your' 'those', 'this',
                           'these', 'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my', 'its', 'his' 'her',
                           'every', 'either', 'each', 'any', 'another', 'an', 'a', 'just', 'mere', 'such',
                           'merely' 'right', 'no', 'not', 'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite', 'rather', 'somewhat',
                           'sufficiently' 'same', 'different', 'such', 'when', 'why', 'where', 'how', 'what', 'who',
                           'whom', 'which', 'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace',
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday', 'everyone', 'everyplace',
                           'everything' 'everywhere', 'whatever', 'whenever', 'whereever', 'whichever', 'whoever',
                           'whomever' 'he', 'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their', 'theirs',
                           'you', 'your', 'yours', 'me', 'my', 'mine', 'I', 'we', 'us', 'much', 'and/or'}

N_QUERY: int = 225
queries: Dict[str, List[float]] = {}
abstracts: Dict[str, List[float]] = {}
q_text: List[List[str]] = [[]] * N_QUERY
ps = PorterStemmer()


def useless(word: str):
    return word in closed_class_stop_words or word.isdigit() or not word.isalnum()


def create_list(t: List[str], d: Dict[str, List[float]], idx: int, store_list=None):
    text = list(nltk.bigrams([w for w in t if not useless(w)]))
    counter = Counter(text)
    for word, count in counter.items():
        if word not in d:
            d[word] = [0] * (1400 + 1)
        d[word][idx] = log(count / len(text))
        d[word][-1] += 1
    if store_list is not None:
        store_list[idx] = counter.keys()


def process_file(f: TextIO, d: Dict, store_list=None):
    current: List[str] = []
    idx: int = -1
    for line in f:
        s = [ps.stem(w) for w in word_tokenize(line)]
        if s[0].startswith('.W'):
            continue
        if s[0].startswith('.I'):
            if idx > -1:
                create_list(current, d, idx, store_list)
                current = []
            idx += 1
        else:
            current.extend(s)
    create_list(current, d, idx, store_list)


def calculate_tf_idf(d: Dict, w: str, idx: int, no_term: int) -> float:
    if w not in d:
        return 0
    tf = d[w][idx]
    idf = log(no_term / (d[w][-1]))
    return tf * idf


def main():
    with open('cran.qry', 'r') as f1, open('cran.all.1400', 'r') as f2:
        process_file(f1, queries, q_text)
        process_file(f2, abstracts)

    query_scores: List[List[float]] = [[]] * N_QUERY
    for i in range(N_QUERY):
        query_scores[i] = [calculate_tf_idf(queries, w, i, N_QUERY) for w in q_text[i]]

    for i in range(N_QUERY):
        similarities: List[float] = [0] * 1400
        for j in range(1400):
            score = [calculate_tf_idf(abstracts, w, j, 1400) for w in q_text[i]]
            similarities[j] = dot(query_scores[i], score)
            if similarities[j]:
                similarities[j] /= norm(query_scores[i]) * norm(score)
        indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)
        similarities.sort(reverse=True)
        for j in range(1400):
            if similarities[j]:
                print(' '.join([str(i + 1), str(indices[j] + 1), '{0:.32f}'.format(similarities[j])]))


if __name__ == "__main__":
    main()
