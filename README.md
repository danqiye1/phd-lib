# Created with PyTorch

## Description

I am consolidating my PyTorch work for PhD into one single repository of installable packages. I want my work to be easily reproducible and transparent (plus easy) to review. You may find that some of these are rehash of tried and tested techniques, but I subscribe to the idea of "what I cannot create, I don't understand". Some of the (upcoming) work packages here include:
1. Continual Machine Learning
2. Spiking Neural Network

Over the years, I have also found several utilities and design patterns to be ubiquitous across different PyTorch projects. I want to put them into a utilities library for easy use in future, instead of always looking up such patterns in documentation, tutorials, or stackoverflow. The `patterns` package is my attempt to do so.

Tests may be a little overkill for non-production code, but it may be a good idea to add them in future.

## Getting Started

Installation:
```bash
git clone https://github.com/danqiye1/phd-lib
cd phd-lib
pip install -e .
```

There are currently 2 packages:
```
continual: Tools and experiments on continual learning
patterns: Tools for normal experiments
```

## Running experiments

Experiments can be rerun as package modules:
```bash
# Example for running rehearsal
python -m continual.rehearsal
```
