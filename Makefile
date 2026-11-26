.PHONY: install test data pretrain baselines tune main ablate experiments report
install:
	pip install -r requirements-dev.txt
test:
	python -m pytest -q
data:
	python scripts/build_data.py
pretrain:
	python -m modellab.pretrain --minutes 35
baselines:
	python -m modellab.experiments baselines
tune:
	python -m modellab.experiments tune
main:
	python -m modellab.experiments main
ablate:
	python -m modellab.experiments ablate
experiments:
	python -m modellab.experiments all
report:
	python -m modellab.analysis && python scripts/make_report.py
