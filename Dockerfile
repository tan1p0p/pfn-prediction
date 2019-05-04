FROM chainer/chainer:latest-python3

WORKDIR /pfn2019
ADD . /pfn2019
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt