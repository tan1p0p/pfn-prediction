FROM chainer/chainer

WORKDIR /pfn2019

ADD . /pfn2019

RUN pip install -r requirements.txt