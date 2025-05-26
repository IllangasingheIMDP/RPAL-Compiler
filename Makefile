all:
	@echo "RPAL Compiler is ready to use"
	@echo "Usage: python myrpal.py [-ast] file_name"

clean:
	rm -f *.pyc
	rm -rf __pycache__
