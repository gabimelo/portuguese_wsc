FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion make

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

COPY ./environment.yml /code/environment.yml

# name of environment in environment.yml needs to be "wsc_port", as this gets refered to later on in this file
RUN if [ "$(awk '/name/ {print $2}' /code/environment.yml)" != "wsc_port" ]; then exit 1; fi

RUN conda env create -f /code/environment.yml
RUN echo "source activate wsc_port" > ~/.bashrc
ENV PATH /opt/conda/envs/wsc_port/bin:$PATH

RUN python -c "import nltk; nltk.download('punkt')"

RUN python -m ipykernel install --user --name jupyter_env --display-name "Python (jupyter_env)"
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

COPY . /code

WORKDIR "/code"

ENTRYPOINT ["/bin/bash", "-c"]
