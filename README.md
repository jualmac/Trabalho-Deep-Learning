# Trabalho-Deep-Learning

O MVP do trabalho de neuroevolucao esta em [`src/README.md`](src/README.md).

Para executar a aplicacao:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m neuroevolution_mvp.cli
PYTHONPATH=src streamlit run src/app.py
```
