portuguese_wsc
==============================

**Currently under development**

Solver for Winograd Schema Challenge in Portuguese. Portuguese translations for original Winograd Schema Challenge are also being proposed here.

Preliminary results were presented on a conference paper: [Melo, Gabriela Souza de; Imaizumi, Vinicius A. ; Cozman, Fabio Gagliardi . Winograd Schemas in Portuguese. In: Encontro Nacional de Inteligência Artificial e Computacional, 2019](http://www.bracis2019.ufba.br/Camera_Ready/199152_1.pdf).

----

## Project Setup

![Python 3.7](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Blue_Python_3.7_Shield_Badge.svg/76px-Blue_Python_3.7_Shield_Badge.svg.png)

- This project has not been tested in machines without CUDA GPUs available.

- A Dockerfile is available, and may be used with `docker build -t wsc_port  .` followed by `nvidia-docker run -it -v $PWD/models:/code/models wsc_port <desired_command>` (ie `nvidia-docker run -it -v $PWD/models:/code/models wsc_port python -m src.main`).

- The docker-compose file contains a few different options for running the code, which can be run with commands such as: `docker-compose run <service_name>` (ie `docker-compose run train`). For the jupyter-server, run with `docker-compose run --service-ports jupyter-server`.

- For running outside of the Docker container, Conda is required.

    - To create the conda environment: `conda env create -f environment.yml`

- Makefile contains some of the commands used to run the code. These commands must be run from inside the environment.

    - to setup the environment for running the project: `make dev-init`. This command also makes sure `make processed-data` is run, which prepares data needed to train model
        - The data corresponding to the Corpus being used is organized as follows:
            - Raw data: files used to generate the final Winograd Schema Challenge schema collection JSONs
            - External data: the compressed XML file, as downloaded from Wikipedia's dump archive
            - Interim data: TXT files extracted from the above. May or may not be split between different, smalle files
            - Processed data: TXT files, containing text split between train, test and validation splits. It also contains the generated Winograd Schema Challenge schema collection JSONs.
                - Additionally, `make recuced-processed-data` reduces size of each of these splits
    - running `make corpus` will speed up first run of code (but is not necessary)
    - `make train` trains a model
    - `make winograd-test` runs evaluation of Winograd Schema Challenge
    - `make generate` runs language model for generation of text

- Code runs for both English and Portuguese cases, and this setting is controlled by the variable `PORTUGUESE` in `src.consts`.

- Run tests with `make tests`, which is equivalent to `pytest --cov=src tests/`. Use `pytest --cov=src --cov-report=html tests/` for generation of HTML test report. Needs pytest and pytest-cov packages. If there are import errors, should run `pip install -e .` to locally install package from source code.


### Winograd Collection Generation

There is also code in this repository for generation the Winograd Schema Collection JSON, from the original HTML file, to be ready to be used by the solver. This generation happens by executing `python -m src.winograd_collection_manipulation.wsc_subsets_generation`. To generate the version with translated names, after that first command, simply run `python -m src.winograd_collection_manipulation.name_replacer`. These commands don't need to be called to be able to run the solver, given that the JSON file is already present in this repository. However, this code is being made available, in case it can help with translations for the Challenge to other languages.

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
    ├── models             <- Trained and serialized models, model predictions, or model summaries. Gitignored due to their size.
    │
    ├── notebooks          <- Jupyter notebooks, used during experimentation and testing.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    └── tests              <- Tests module, using Pytest.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

## References

- Code for Language Model based on [Pytorch's Word-level language modeling RNN example](https://github.com/pytorch/examples/tree/master/word_language_model)
- Code for parallelization of PyTorch model based on [PyTorch-Encoding package](https://github.com/zhanghang1989/PyTorch-Encoding) with help from [this medium post](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255).
- Idea of using language model for solving Winograd Schema Challenge based on the paper ["A Simple Method for Commonsense Reasoning"](https://arxiv.org/abs/1806.02847), by Trieu H. Trinh and Quoc V. Le, 2018.
