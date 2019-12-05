__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"

from random import random, choice
import string
from typing import List, Union, NamedTuple, Tuple


# GramFreqTuple = NamedTuple("GramFreqTuple", [('ngram', str), ('freq', float)])


class NGram:
    # TODO doc me!

    def __init__(self, n: int, stops: List[str]=None, text: List[str]=None):
        # TODO doc me!
        self.stops: List[str] = stops if stops is not None \
            else self.default_stops()
        self.corpus: str = self.stringify_docs(text, self.stops)
        self.ngrams: dict = {}
        self.order: int = n - 1

    # def parse_ngrams(self, n: int) -> None:
    #     ngrams = []
    #     processed: List[str] = self.corpus.split(' ')
    #     num_words: int = len(processed)
    #
    #     # set for unique ngrams and the total number of ngrams in the corpus
    #     unique: set = set()
    #     num_grams: int = num_words - n + 1
    #
    #     for i in range(n-1, num_words):
    #         # build current ngram as [i-n+1, i-n+2, ..., i+1]
    #         # i+1 is necessitated due to the exclusive upper bound on range
    #         ngram = [processed[j] for j in range(i-n+1, i+1)]
    #         joined = ' '.join(ngram)
    #
    #         # if ngram has been encountered before, skip
    #         if joined not in unique:
    #             # else, new ngram to calculate the probability of
    #             freq = self.corpus.count(joined) / num_grams
    #             unique.add(joined)
    #             ngrams.append((joined, freq))
    #             # word is element of interest (c)
    #             # hist is list of preceding words [c-1, c-2, ..., c-n+1]
    #             word, hist = ngram[-1], list(reversed(ngram[:-1]))
    #
    #     self.ngrams[n] = sorted(ngrams, key=lambda x: -x[1])

    def parse_ngrams(self, n: int) -> None:
        """Updates ngram dict with probabilities for the given n.

        :param n: 1 = unigram, 2 = bigram, 3 = trigram, ...
        :return:
        """
        # create table entry for this n (eg entries in ngrams[1] are unigrams)
        self.ngrams[n]: dict = {}
        processed: List[str] = self.corpus.split(' ')
        num_words: int = len(processed)

        # set for unique ngrams and the total number of ngrams in the corpus
        unique: set = set()
        num_grams: int = num_words - n + 1

        # iterate through every group of n consecutive words
        for i in range(n-1, num_words):
            # build current ngram as [i-n+1, i-n+2, ..., i+1]
            # i+1 is necessitated due to the exclusive upper bound on range
            ngram = [processed[j] for j in range(i-n+1, i+1)]
            joined = ' '.join(ngram)

            # if ngram has been encountered before, skip
            if joined not in unique:
                # else, new ngram to calculate the probability of
                unique.add(joined)
                freq = self.corpus.count(joined) / num_grams
                # word is element of interest (c)
                # hist is list of preceding words [c-1, c-2, ..., c-n+1]
                word, hist = ngram[-1], list(reversed(ngram[:-1]))

                # build table s.t. ngrams[n][word][c-1][...] has value freq
                self.ngrams[n][word] = self._nest_dict(
                    self.ngrams[n][word], hist, freq
                )

    def generate(self, numwd: int=20, numsq: int=10) -> List[str]:
        """Generates numsq sentences of numwd words.

        :param numwd: number of words to generate per sentence
        :param numsq: number of sentences to generate
        :return: list of generated sentences
        """
        sequences: List[str] = []   # list storing each generated sentence

        # generate the desired number of sentences
        while len(sequences) < numsq:
            # create a list to store the working sequence and select a unigram
            sequence: List[str] = []
            unigram = self.roulette_selector(sequence)
            sequence.append(unigram)

            # roulette select a bigram based on the proceeding unigram
            bigram = self.roulette_selector(sequence)
            sequence.append(bigram)

            # generate trigrams until the sentence is finished
            while len(sequence) < numwd:
                trigram = self.roulette_selector(sequence)
                sequence.append(trigram)

            sequences.append(' '.join(sequence))

        return sequences

    def roulette_selector(self, seq: List[str]=None) -> str:
        """Randomly selects the next ngram conditioned on proceeding sequence.

        :param seq: the last 2 words (at most) in the generated sequence
        :return: a semi-randomly generated next word
        """
        seq = seq[-2:]  # reduce to last two elements (or less if shorter)
        # if empty list, pick a pseudorandom unigram
        if not seq:
            return choice(list(self.ngrams[1]))

        # only one element; roulette select a likely applicable bigram
        elif len(seq) == 1:
            options = sorted([tuple([k,v]) for k,v in
                              self.ngrams[2][seq[0]].items()],
                             key=lambda x: -x[1])
            next_word = self._roulette_selector(options)
            return next_word

        # else, consider applicable trigrams and roulette select one
        else:
            options = sorted([tuple([k,v]) for k,v in
                              self.ngrams[2][seq[0]][seq[1]].items()],
                             key=lambda x: -x[1])
            next_word = self._roulette_selector(options)
            return next_word

    def _roulette_selector(self, options: List[tuple]) -> str:
        """Helper that semi-randomly selects from a list of potential words.

        :param options: list of potential next words and their probabilities
        :return: first word exhausting the random value
        """
        r = random()
        for tup in options:
            r -= tup[1]
            if r <= 0:
                return tup[0]
        return options[0][0]

    def word_prob(self, word: str, cond: List[str]) -> float:
        """Calculates the probability of word conditioned on cond.

        :param word: word to calculate probability for
        :param cond: ordered list of preceding words
        :return: probability of word appearing after cond
        """
        pass

    def load_text(self, text: List[str]) -> None:
        """Sets model's corpus; if not empty, extends corpus with new docs.

        :param text: list of documents to set as the model's corpus
        :return:
        """
        # gets list of joined processed documents
        processed = self.stringify_docs(text, self.stops)

        # if corpa isn't None, we extend it to preserve the existing corpa
        if self.corpus is not None:
            # TODO recalculate ALL stats
            processed = ' '.join([self.corpus, processed])
        self.corpus = processed

    def extend_corpus(self, ext: str):
        # TODO recalculate all stats
        pass

    @classmethod
    def default_stops(cls) -> List[str]:
        """Loads default stopwords from provided EnglishStopwords.txt file"""
        with open('EnglishStopwords.txt', 'r') as f:
            stops = [l for l in f.readlines()]
        return stops

    @classmethod
    def _nest_dict(cls, d: dict, keys: List[str], val: float) \
            -> Union[dict, float]:
        """Recursively constructs nested dicts with each key a word in an ngram.

        :param d: the current working dict
        :param keys: list of preceding words in ngram [c-1, ..., c-n+1]
        :param val: the probability of a given ngram
        :return: base case returns val; else, the dict constructed on unwinding
        """
        # if no more keys, at the end (beginning) of ngram; assign prob.
        if not keys:
            return val
        # word has been encountered before; pass its dict through to preserve
        elif keys[0] in d.keys():
            d[keys[0]] = cls._nest_dict(d[keys[0]], keys[1:], val)
            return d[keys[0]]
        # word has not been seen in this sequence; pass through blank dict
        else:
            d[keys[0]] = cls._nest_dict({}, keys[1:], val)
            return d[keys[0]]

    @classmethod
    def stringify_docs(cls, docs: List[str], stops: List[str]) -> str:
        """Removes punctuation and stopwords from docs list before concatenation

        Given a list of documents, this first splits each individual document
        into its constituent words. These are then lower-cased and filtered to
        remove all stopwords and punctuation, then joined together with spaces.
        After each document is processed in this manner, they are all joined
        together into one large string comprising the corpus for this model.

        Stopwords are defined via self.stops, which are provided on object
        instantiation. Punctuation is defined by the Python standard library
        `string` module by `string.punctuation`.

        :param docs: list of documents to process
        :param stops: list of stopwords to remove
        :return: string of all processed docs concatenated together
        """
        processed = ' '.join([
            ' '.join(
                w.lower() for w in t.split(' ') if w.lower() not in stops
                and w.lower() not in string.punctuation
            ) for t in docs
        ])
        return processed
