.PHONY: setup data quick bench suite clean distclean

setup:           ## install dependencies
	pip install -r requirements.txt

data:            ## download the ULB dataset into data/raw/
	bash scripts/download_data.sh

quick:           ## fast smoke test (subsample, 1 seed, 2 models)
	python -m experiments.run_benchmark --quick

bench:           ## full leaky-vs-safe benchmark
	python -m experiments.run_benchmark

suite:           ## the whole experiment suite, E1-E8 + fee sensitivity, in order
	python -m experiments.run_benchmark
	python -m experiments.run_policy_ladder
	python -m experiments.run_leakage_forensics
	python -m experiments.run_bootstrap
	python -m experiments.run_bootstrap --seeds 3 4 --tag confirm \
		--pairs class_weight__random_forest:smote__random_forest
	python -m experiments.run_benchmark --split temporal
	python -m experiments.run_explain
	python -m experiments.run_tuning
	python -m experiments.run_fee_sensitivity
# ordered so dependencies hold: run_bootstrap and run_fee_sensitivity
# reprice the per-seed scores that run_benchmark caches in results/scores/.
# the second run_bootstrap call is E5's confirmatory arm: the money-gap pair
# was pre-registered on the discovery seeds (0-2) and re-tested on the held
# back seeds 3-4

clean:           ## remove generated tables/figures (keeps results/scores cache)
	rm -f results/tables/*.csv
	find results/figures -name '*.png' \
		! -name pipeline_overview.png ! -name bootstrap_forest_slide.png \
		-delete
# pipeline_overview + bootstrap_forest_slide are hand-made (no generating
# script) and git-tracked, so clean must not delete them
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean ## also drop the per-seed score cache run_bootstrap reads
	rm -f results/scores/*.npz
# use distclean after any code/data change: otherwise a rerun of
# run_bootstrap would silently analyse scores produced by the old code
