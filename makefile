# Makefile for setting up a Python virtual environment, installing dependencies, and managing the project

PROJECT_DIR := $(shell pwd)
BIN_DIR := $(PROJECT_DIR)/venv/bin
SRC_DIR := $(PROJECT_DIR)/invest
NOTEBOOKS_DIR := $(PROJECT_DIR)/notebooks

# Use PYTHONPATH and PATH from .env file if it exists
-include .env

.PHONY: install clean test lint format run help

venv:
	python3.12 -m venv venv
	$(BIN_DIR)/python -m pip install --upgrade pip


.env: venv
	@echo "PYTHONPATH=$(shell pwd)" > .env
	@echo "PATH=$(shell pwd)/$(BIN_DIR):$$PATH" >> .env

venv/dependencies: venv .env requirements.lock
	$(BIN_DIR)/pip install -r requirements.lock
	@touch venv/dependencies

venv/test-dependencies: venv/dependencies requirements-test.txt
	$(BIN_DIR)/pip install -r requirements-test.txt
	@touch venv/test-dependencies

install: venv/dependencies

clean:
	rm -rf venv
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

requirements.lock: venv .env
	$(BIN_DIR)/pip install -r requirements.txt
	$(BIN_DIR)/pip freeze > requirements.lock
	touch requirements.lock

lock: requirements.lock

format: venv/test-dependencies
	$(BIN_DIR)/autoflake --recursive $(SRC_DIR)
	$(BIN_DIR)/isort $(SRC_DIR)
	$(BIN_DIR)/black $(SRC_DIR)
	$(BIN_DIR)/ruff check --fix $(SRC_DIR)

lint: venv/test-dependencies
	$(BIN_DIR)/flake8 $(SRC_DIR)
	$(BIN_DIR)/pylint $(SRC_DIR)
	$(BIN_DIR)/mypy $(SRC_DIR)

test: venv/test-dependencies
	$(BIN_DIR)/pytest

run:
	$(BIN_DIR)/python -m $(SRC_DIR)

help:
	@echo "Available commands:"
	@echo "  make install          : Set up virtual environment and install dependencies"
	@echo "  make clean            : Remove virtual environment and cached files"
	@echo "  make lock      : Create or update requirements.lock file"
	@echo "  make format           : Format the code"
	@echo "  make lint             : Run linters"
	@echo "  make test             : Run tests"
	@echo "  make run              : Run the application"
	@echo "  make help             : Show this help message"
