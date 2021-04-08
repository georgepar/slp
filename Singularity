Bootstrap: docker
From: ubuntu:20.04

%setup
    mkdir ${SINGULARITY_ROOTFS}/project

%environment
    export PYTHONPATH=$PYTHONPATH:/opt/cmusdk/

%post
    apt-get -y update
    DEBIAN_FRONTEND=noninteractive apt-get -y install python3 python3-pip ffmpeg git
    ln -s /usr/bin/python3 /usr/bin/python
    pip3 install --no-cache-dir poetry cookiecutter

    cd /project
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    pip3 install --no-cache-dir -r requirements.txt
    cd /

    python -m  spacy download en_core_web_sm
    python -m  spacy download en_core_web_md
    python -m  spacy download el_core_news_sm
    python -m  spacy download el_core_news_md

    git clone https://github.com/A2Zadeh/CMU-MultimodalSDK /opt/cmusdk/

    apt-get -y remove --purge git python3-pip
    apt-get -y install python3-setuptools
    apt-get clean
    apt-get autoclean
    apt-get -y autoremove

%files
    pyproject.toml /project
    poetry.lock /project
