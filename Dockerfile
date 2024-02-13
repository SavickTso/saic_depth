FROM ubuntu:22.04

RUN apt update --fix-missing
RUN apt install vim git python3-pip ffmpeg libsm6 libxext6 -y
WORKDIR /root/
RUN apt remove -y python3-blinker

COPY . /root/saic_depth_completion/.
# RUN git clone https://github.com/alaflaquiere-sony/saic_depth_completion.git
RUN cd saic_depth_completion && pip3 install -r requirements.txt && python3 setup.py install
WORKDIR /root/saic_depth_completion