Bootstrap: docker
From: ubuntu:20.04

%setup
    mkdir ${SINGULARITY_ROOTFS}/project

%post
    apt-get -y update
    apt-get -y install python3 python3-pip
    ln -s /usr/bin/python3 /usr/bin/python
    pip3 install poetry cookiecutter
    cd /project && poetry export -f requirements.txt --output requirements.txt --without-hashes && pip3 install -r requirements.txt

%files
    pyproject.toml /project
    poetry.lock /project
