portuguese_wsc
==============================

**Currently under development**

Solver for Winograd Schema Challenge in Portuguese. Portuguese translations for original Winograd Schema Challenge are also being proposed here.

- Code for Language Model based on [Pytorch's Word-level language modeling RNN example](https://github.com/pytorch/examples/tree/master/word_language_model)
- Code for parallelization of PyTorch model based on [PyTorch-Encoding package](https://github.com/zhanghang1989/PyTorch-Encoding) with help from [this medium post](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255).
- Idea of using language model for solving Winograd Schema Challenge based on paper "A Simple Method for Commonsense Reasoning":
```
@article{DBLP:journals/corr/abs-1806-02847,
  author    = {Trieu H. Trinh and
               Quoc V. Le},
  title     = {A Simple Method for Commonsense Reasoning},
  journal   = {CoRR},
  volume    = {abs/1806.02847},
  year      = {2018},
  url       = {http://arxiv.org/abs/1806.02847},
  archivePrefix = {arXiv},
  eprint    = {1806.02847},
  timestamp = {Mon, 13 Aug 2018 16:46:22 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1806-02847},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

----

Makefile contains some of the commands used to run the code. When project is finalized reproduction instructions will be made available here.

----

### Project Setup

- To create the environment for running the project: `conda env create -f environment.yml`

- Run tests with `pytest --cov=src tests/`. Use `pytest --cov=src --cov-report=html tests/` for generation of HTML test report. Needs pytest and pytest-cov packages. If there are import errors, should run `pip install -e .` to locally install package from source code.

----


### Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`.
    ├── README.md          <- The top-level README for developers using this project.
    ├── environment.yml    <- Contains project's requirements, generated from Anaconda environment.
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── githooks           <- Contains githooks scripts being used for development. Git hook directory for repo needs to be set to this folder.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   └── scripts           
    └── tests              <- Tests module, using Pytest.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
