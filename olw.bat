@echo off
python -c "import sys; from obsidian_llm_wiki.cli import cli; sys.exit(cli())" %*
