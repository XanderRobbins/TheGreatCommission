.PHONY: install generate-data train-baseline fine-tune-lora translate evaluate demo clean help

help:
	@echo "Scripture Translation System - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install              - Install dependencies"
	@echo "  make generate-data        - Generate sample Bible verse data"
	@echo ""
	@echo "Training:"
	@echo "  make train-baseline       - Train baseline model (EN-ES)"
	@echo "  make fine-tune-lora       - Fine-tune with LoRA on rare language"
	@echo ""
	@echo "Translation & Evaluation:"
	@echo "  make demo                 - Run complete system demo"
	@echo "  make translate            - Translate a book"
	@echo "  make evaluate             - Evaluate translations"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean                - Remove generated files"
	@echo "  make logs                 - View training logs"

install:
	pip install -r requirements.txt
	python -m nltk.downloader punkt

generate-data:
	python data/generate_sample_data.py

train-baseline:
	python scripts/train_baseline.py \
		--data_path ./data/en_es_verses.jsonl \
		--source_lang eng_Latn \
		--target_lang spa_Latn \
		--batch_size 32 \
		--num_epochs 3

fine-tune-lora:
	python scripts/fine_tune_lora.py \
		--pretrained_model_path ./models/checkpoints/final_model \
		--data_path ./data/en_sw_verses.jsonl \
		--target_lang swh_Latn \
		--batch_size 8 \
		--num_epochs 5

translate:
	python scripts/translate_book.py \
		--input_path ./data/en_verses.json \
		--output_path ./results/translation.json \
		--target_lang spa_Latn

demo:
	python demo.py

evaluate:
	python evaluation.py

logs:
	tail -f logs/*.log

clean:
	rm -rf logs/*.log
	rm -rf results/*.json
	rm -rf models/checkpoints/*
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

.SILENT: help
