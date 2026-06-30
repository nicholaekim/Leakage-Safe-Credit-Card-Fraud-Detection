.PHONY: setup data quick bench clean

setup:           ## install dependencies
	pip install -r requirements.txt

data:            ## download the ULB dataset into data/raw/
	bash scripts/download_data.sh

quick:           ## fast smoke test (subsample, 1 seed, 2 models)
	python -m experiments.run_benchmark --quick

bench:           ## full leaky-vs-safe benchmark
	python -m experiments.run_benchmark

clean:           ## remove generated tables/figures and caches
	rm -f results/tables/*.csv results/figures/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
