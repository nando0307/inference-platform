# Modal requires registry images to expose `python`; reuse vLLM's existing `python3`.
PYTHON_ALIAS_SETUP = 'RUN ln -s "$(command -v python3)" "$(dirname "$(command -v python3)")/python"'
