# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computes rouge scores between two text blobs.

Implementation replicates the functionality in the original ROUGE package. See:

Lin, Chin-Yew. ROUGE: a Package for Automatic Evaluation of Summaries. In
Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004),
Barcelona, Spain, July 25 - 26, 2004.

Default options are equivalent to running:
ROUGE-1.5.5.pl -e data -n 2 -a settings.xml

Or with use_stemmer=True:
ROUGE-1.5.5.pl -m -e data -n 2 -a settings.xml

In these examples settings.xml lists input files and formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from absl import logging
import nltk
import numpy as np
import six
from six.moves import map, range
from nltk.stem import porter


import abc


# Pre-compile regexes that are use often
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
punctuation = re.compile(
    "[\u0000-\u002F\u003A-\u0040\u00B0"
    + "\u2000-\u206F\u0021-\u002F"
    + "\u005B-\u0060\u007B-\u007F\u00A1\u00A6\u00AB-\u00AF"
    + "\u00B7\u00B8\u00BB\u00BF\u2E00-\u2E4E\u3000-\u301F\uFE30-\uFE4F\uFF01-\uFF0F"
    + "\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF64\uFE50-\uFE6B\uFE10-\uFE19"
    + "\u0609\u060A\u060C\u060D\u061B\u061E\u061F\u066A"  # Arabic specific
    + "]+"
)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)

ArabicDiacritics = re.compile(
    "[\u0640\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670]+"
)
ArabicSuperfluous = re.compile(
    "[\u06DC\u06DF\u06E0\u06E2\u06E3\u06E5\u06E6\u06E8\u06EA\u06EB\u06EC\u06ED\u0653]+"
)


def removeDiacriticsArabic(text):
    return ArabicDiacritics.sub(r"", text)


def removeSuperfluousArabic(text):
    return ArabicSuperfluous.sub(r" ", text)


def tokenize(text, stemmer):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
      text: A text blob to tokenize.
      stemmer: An optional stemmer.

    Returns:
      A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    text = removeDiacriticsArabic(removeSuperfluousArabic(text))
    # Replace any non-alpha-numeric characters with spaces.
    # text = six.ensure_str(text) # NON_ALPHANUM_RE.sub(" ", six.ensure_str(text))
    text = punctuation.sub(" ", six.ensure_str(text))

    tokens = SPACES_RE.split(text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x for x in tokens]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens]  # if VALID_TOKEN_RE.match(x)]
    print(tokens)

    return tokens


class Tokenizer(abc.ABC):
    """Abstract base class for a tokenizer.

    Subclasses of Tokenizer must implement the tokenize() method.
    """

    @abc.abstractmethod
    def tokenize(self, text):
        raise NotImplementedError("Tokenizer must override tokenize() method")


class DefaultTokenizer(Tokenizer):
    """Default tokenizer which tokenizes on whitespace."""

    def __init__(self, use_stemmer=False):
        """Constructor for DefaultTokenizer.

        Args:
          use_stemmer: boolean, indicating whether Porter stemmer should be used to
          strip word suffixes to improve matching.
        """
        self._stemmer = porter.PorterStemmer() if use_stemmer else None

    def tokenize(self, text):
        return tokenize(text, self._stemmer)


class Score(collections.namedtuple("Score", ["precision", "recall", "fmeasure"])):
    """Tuple containing precision, recall, and f-measure values."""


class BaseScorer(object, metaclass=abc.ABCMeta):
    """Base class for Scorer objects."""

    @abc.abstractmethod
    def score(self, target, prediction):
        """Calculates score between the target and prediction.

        Args:
          target: Text containing the target (ground truth) text.
          prediction: Text containing the predicted text.

        Returns:
          A dict mapping each score_type (string) to Score object.
        """


class AggregateScore(collections.namedtuple("AggregateScore", ["low", "mid", "high"])):
    """Tuple containing confidence intervals for scores."""


class BootstrapAggregator(object):
    """Aggregates scores to provide confidence intervals.

    Sample usage:
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
      aggregator = Aggregator()
      aggregator.add_scores(scorer.score("one two three", "one two"))
      aggregator.add_scores(scorer.score("one two five six", "seven eight"))
      result = aggregator.aggregate()
      print result
      {'rougeL': AggregateScore(
           low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
           mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
           high=Score(precision=1.0, recall=0.66, fmeasure=0.80)),
       'rouge1': AggregateScore(
           low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
           mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
           high=Score(precision=1.0, recall=0.66, fmeasure=0.80))}
    """

    def __init__(self, confidence_interval=0.95, n_samples=1000):
        """Initializes a BootstrapAggregator object.

        Args:
          confidence_interval: Confidence interval to compute on the mean as a
            decimal.
          n_samples: Number of samples to use for bootstrap resampling.

        Raises:
          ValueError: If invalid argument is given.
        """

        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError("confidence_interval must be in range [0, 1]")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self._n_samples = n_samples
        self._confidence_interval = confidence_interval
        self._scores = collections.defaultdict(list)

    def add_scores(self, scores):
        """Adds a sample for future aggregation.

        Args:
          scores: Dict mapping score_type strings to a namedtuple object/class
            representing a score.
        """

        for score_type, score in six.iteritems(scores):
            self._scores[score_type].append(score)

    def aggregate(self):
        """Aggregates scores previously added using add_scores.

        Returns:
          A dict mapping score_type to AggregateScore objects.
        """

        result = {}
        for score_type, scores in six.iteritems(self._scores):
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple(
                (scores[0].__class__(*percentiles[j, :]) for j in range(3))
            )
            result[score_type] = AggregateScore(
                low=intervals[0], mid=intervals[1], high=intervals[2]
            )
        return result

    def _bootstrap_resample(self, matrix):
        """Performs bootstrap resampling on a matrix of scores.

        Args:
          matrix: A 2-d matrix of (sample, measure).

        Returns:
          A 2-d matrix of (bounds, measure). There are three bounds: low (row 0),
          mid (row 1) and high (row 2). Mid is always the mean, while low and high
          bounds are specified by self._confidence_interval (which defaults to 0.95
          meaning it will return the 2.5th and 97.5th percentiles for a 95%
          confidence interval on the mean).
        """

        # Matrix of (bootstrap sample, measure).
        sample_mean = np.zeros((self._n_samples, matrix.shape[1]))
        for i in range(self._n_samples):
            sample_idx = np.random.choice(
                np.arange(matrix.shape[0]), size=matrix.shape[0]
            )
            sample = matrix[sample_idx, :]
            sample_mean[i, :] = np.mean(sample, axis=0)

        # Take percentiles on the estimate of the mean using bootstrap samples.
        # Final result is a (bounds, measure) matrix.
        percentile_delta = (1 - self._confidence_interval) / 2
        q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
        return np.percentile(sample_mean, q, axis=0)


def fmeasure(precision, recall):
    """Computes f-measure given precision and recall values."""

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0


class RougeScorer(BaseScorer):
    """Calculate rouges scores between two blobs of text.

    Sample usage:
      scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
      scores = scorer.score('The quick brown fox jumps over the lazy dog',
                            'The quick brown dog jumps on the log.')
    """

    def __init__(
        self, rouge_types, use_stemmer=False, split_summaries=False, tokenizer=None
    ):
        """Initializes a new RougeScorer.

        Valid rouge types that can be computed are:
          rougen (e.g. rouge1, rouge2): n-gram based scoring.
          rougeL: Longest common subsequence based scoring.

        Args:
          rouge_types: A list of rouge types to calculate.
          use_stemmer: Bool indicating whether Porter stemmer should be used to
            strip word suffixes to improve matching. This arg is used in the
            DefaultTokenizer, but other tokenizers might or might not choose to
            use this.
          split_summaries: whether to add newlines between sentences for rougeLsum
          tokenizer: Tokenizer object which has a tokenize() method.
        Returns:
          A dict mapping rouge types to Score tuples.
        """

        self.rouge_types = rouge_types
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = DefaultTokenizer(use_stemmer)
            logging.debug("Using default tokenizer.")

        self._split_summaries = split_summaries

    def score_multi(self, targets, prediction):
        """Calculates rouge scores between targets and prediction.

        The target with the maximum f-measure is used for the final score for
        each score type..

        Args:
          targets: list of texts containing the targets
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """
        score_dicts = [self.score(t, prediction) for t in targets]
        max_score = {}
        for k in self.rouge_types:
            index = np.argmax([s[k].fmeasure for s in score_dicts])
            max_score[k] = score_dicts[index][k]

        return max_score

    def score(self, target, prediction):
        """Calculates rouge scores between the target and prediction.

        Args:
          target: Text containing the target (ground truth) text,
          or if a list
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """

        # Pre-compute target tokens and prediction tokens for use by different
        # types, except if only "rougeLsum" is requested.
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self._tokenizer.tokenize(target)
            prediction_tokens = self._tokenizer.tokenize(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _score_lcs(target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    if self._split_summaries:
                        sents = nltk.sent_tokenize(text)
                    else:
                        # Assume sentences are separated by newline.
                        sents = six.ensure_str(text).split("\n")
                    sents = [x for x in sents if len(x)]
                    return sents

                target_tokens_list = [
                    self._tokenizer.tokenize(s) for s in get_sents(target)
                ]
                prediction_tokens_list = [
                    self._tokenizer.tokenize(s) for s in get_sents(prediction)
                ]

                scores = _summary_level_lcs(target_tokens_list, prediction_tokens_list)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                target_ngrams = _create_ngrams(target_tokens, n)
                prediction_ngrams = _create_ngrams(prediction_tokens, n)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result


def _create_ngrams(tokens, n):
    """Creates ngrams from the given list of tokens.

    Args:
      tokens: A list of tokens from which ngrams are created.
      n: Number of tokens to use, e.g. 2 for bigrams.
    Returns:
      A dictionary mapping each bigram to the number of occurrences.
    """

    ngrams = collections.Counter()
    for ngram in (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def _score_lcs(target_tokens, prediction_tokens):
    """Computes LCS (Longest Common Subsequence) rouge scores.

    Args:
      target_tokens: Tokens from the target text.
      prediction_tokens: Tokens from the predicted text.
    Returns:
      A Score object containing computed scores.
    """

    if not target_tokens or not prediction_tokens:
        return Score(precision=0, recall=0, fmeasure=0)

    # Compute length of LCS from the bottom up in a table (DP appproach).
    lcs_table = _lcs_table(target_tokens, prediction_tokens)
    lcs_length = lcs_table[-1][-1]

    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    _fmeasure = fmeasure(precision, recall)

    return Score(precision=precision, recall=recall, fmeasure=_fmeasure)


def _lcs_table(ref, can):
    """Create 2-d LCS score table."""
    rows = len(ref)
    cols = len(can)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def _backtrack_norec(t, ref, can):
    """Read out LCS."""
    i = len(ref)
    j = len(can)
    lcs = []
    while i > 0 and j > 0:
        if ref[i - 1] == can[j - 1]:
            lcs.insert(0, i - 1)
            i -= 1
            j -= 1
        elif t[i][j - 1] > t[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return lcs


def _summary_level_lcs(ref_sent, can_sent):
    """ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.

    Args:
      ref_sent: list of tokenized reference sentences
      can_sent: list of tokenized candidate sentences

    Returns:
      summary level ROUGE score
    """
    if not ref_sent or not can_sent:
        return Score(precision=0, recall=0, fmeasure=0)

    m = sum(map(len, ref_sent))
    n = sum(map(len, can_sent))
    if not n or not m:
        return Score(precision=0, recall=0, fmeasure=0)

    # get token counts to prevent double counting
    token_cnts_r = collections.Counter()
    token_cnts_c = collections.Counter()
    for s in ref_sent:
        # s is a list of tokens
        token_cnts_r.update(s)
    for s in can_sent:
        token_cnts_c.update(s)

    hits = 0
    for r in ref_sent:
        lcs = _union_lcs(r, can_sent)
        # Prevent double-counting:
        # The paper describes just computing hits += len(_union_lcs()),
        # but the implementation prevents double counting. We also
        # implement this as in version 1.5.5.
        for t in lcs:
            if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
                hits += 1
                token_cnts_c[t] -= 1
                token_cnts_r[t] -= 1

    recall = hits / m
    precision = hits / n
    _fmeasure = fmeasure(precision, recall)
    return Score(precision=precision, recall=recall, fmeasure=_fmeasure)


def _union_lcs(ref, c_list):
    """Find union LCS between a ref sentence and list of candidate sentences.

    Args:
      ref: list of tokens
      c_list: list of list of indices for LCS into reference summary

    Returns:
      List of tokens in ref representing union LCS.
    """
    lcs_list = [lcs_ind(ref, c) for c in c_list]
    return [ref[i] for i in _find_union(lcs_list)]


def _find_union(lcs_list):
    """Finds union LCS given a list of LCS."""
    return sorted(list(set().union(*lcs_list)))


def lcs_ind(ref, can):
    """Returns one of the longest lcs."""
    t = _lcs_table(ref, can)
    return _backtrack_norec(t, ref, can)


def _score_ngrams(target_ngrams, prediction_ngrams):
    """Compute n-gram based rouge scores.

    Args:
      target_ngrams: A Counter object mapping each ngram to number of
        occurrences for the target text.
      prediction_ngrams: A Counter object mapping each ngram to number of
        occurrences for the prediction text.
    Returns:
      A Score object containing computed scores.
    """

    intersection_ngrams_count = 0
    for ngram in six.iterkeys(target_ngrams):
        intersection_ngrams_count += min(target_ngrams[ngram], prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())

    precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    recall = intersection_ngrams_count / max(target_ngrams_count, 1)
    _fmeasure = fmeasure(precision, recall)

    return Score(precision=precision, recall=recall, fmeasure=_fmeasure)
