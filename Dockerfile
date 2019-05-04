FROM chainer/chainer:latest-python3

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

WORKDIR /pfn2019
ADD . /pfn2019
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt