#!/bin/bash
# paper.md -> paper.tex -> paper.pdf, with the numbers checked first.
#
# The check is not optional and not a courtesy: four published figures in this project
# drifted from their run records before it existed. If it fails, nothing is built.
set -eu
cd "$(dirname "$0")/../.."

python experiments/paper/claims.py > paper/claims.txt
python experiments/paper/check_paper.py
python experiments/paper/figures.py > /dev/null

pandoc paper/paper.md \
  --from=markdown+pipe_tables+tex_math_dollars \
  --to=latex \
  --bibliography=paper/refs.bib \
  --citeproc \
  --standalone \
  --metadata title="Interface metrics overstate what survives end-to-end training" \
  --metadata author="Andres Hernandez" \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=1in \
  -o paper/paper.tex

pdflatex -interaction=nonstopmode -output-directory=paper paper/paper.tex > paper/latex.log 2>&1 || {
  echo "pdflatex failed, see paper/latex.log"; tail -20 paper/latex.log; exit 1; }
pdflatex -interaction=nonstopmode -output-directory=paper paper/paper.tex >> paper/latex.log 2>&1 || true
ls -la paper/paper.pdf
