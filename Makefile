# Python interpreter
PYTHON = python

# Main script
MAIN = myrpal.py

# Default target
.PHONY: all
all: check

# Declare all targets as .PHONY to prevent file lookup
.PHONY: run ast clean check help %

# Run the RPAL interpreter
run:
	@if "$(word 2,$(MAKECMDGOALS))" == "" ( \
		echo Usage: make run filename \
	) else ( \
		$(PYTHON) $(MAIN) $(word 2,$(MAKECMDGOALS)) \
	)

# Catch-all target to prevent Make from looking for files
%:
	@:

# Run with AST output only
ast:
	@if "$(word 2,$(MAKECMDGOALS))" == "" ( \
		echo Usage: make ast filename \
	) else ( \
		$(PYTHON) $(MAIN) -ast $(word 2,$(MAKECMDGOALS)) \
	)

# Clean up compiled Python files
clean:
	@del /S /Q *.pyc 2>nul
	@rmdir /S /Q __pycache__ 2>nul

# Check Python installation
check:
	@$(PYTHON) --version

# Help message
help:
	@echo Main script: myrpal.py
	@echo.
	@echo Direct script usage:
	@echo   python myrpal.py [options] filename
	@echo.
	@echo Available script options:
	@echo   -ast    Show the Abstract Syntax Tree for the RPAL file
	@echo   -st     Show the Standardized Tree for the RPAL file
	@echo.
	@echo RPAL Interpreter Makefile
	@echo Usage:
	@echo   make run filename    - Run the interpreter with an RPAL file
	@echo   make ast filename    - Show only the AST for an RPAL file
	@echo   make clean          - Clean up compiled files
	@echo   make help           - Show this help message
	@echo.	@echo Example:
	@echo   make run test.rpal
	@echo   make ast test.rpal
	@echo.
	

