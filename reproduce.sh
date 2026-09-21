#!/bin/sh
# Everything in README.md's table, from nothing. ~4,200 calls, ~$1.63 at $0.042/Mtok.
set -e
mkdir -p data && cd data
[ -d scifact ] || { curl -sLO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip && unzip -oq scifact.zip; }
cd ..
python -m pip install bm25s pytrec_eval-terrier
python jev_wide.py                       # self-check, free, no API key needed
TYPESAFE_API_KEY=${TYPESAFE_API_KEY:?set it} python bench_scifact.py 300 50
