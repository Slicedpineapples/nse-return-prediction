PYTHON ?= python3

.PHONY: paper assets clean

paper: assets
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

assets:
	cd ../.. && $(PYTHON) research/publication/scripts/generate_assets.py

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out
