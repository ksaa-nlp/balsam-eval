"""Autonomous evaluation for decision models whose output shapes differ.

A decision model takes a multiple-choice question and returns the option it
picked, usually with a calibrated probability for every option. Every vendor
returns that in a different envelope. This package calls any of them behind one
interface, and derives the mapping itself when it meets a format it has not
seen before.

Invariants, inherited from the benchmark this was extracted from:

1. Probabilities are stored exactly as the model returned them. No
   renormalising, no second softmax, no vector synthesised from a scalar
   confidence. A model without a distribution gets null probability metrics.
2. Decision confidence for scoring is always `max(probabilities)`. Provider
   confidence is stored and displayed, never scored.
3. Gold matching is literal string equality after `strip()`. Invalid output
   counts as wrong and stays in the accuracy denominator.
4. Every stored number is re-derivable from the run's own raw records.
5. Secrets never reach a run file.
"""
