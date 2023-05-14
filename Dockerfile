FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y sudo 
#RUN apt-get install -y terminator --fix-missing
#RUN apt-get install -y iproute2 
#RUN apt-get install -y gedit 
#RUN apt-get install -y lsb-release 
#RUN apt-get install -y lsb-core 
RUN apt-get install -y build-essential --fix-missing
RUN apt-get install -y wget 
RUN apt-get install -y nano
RUN apt-get install -y python3-pip

#opencv
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0
RUN pip install pytest

RUN pip install -r ./requirement.txt

RUN adduser user
RUN adduser user sudo
#remove password
RUN passwd -d user
#RUN echo "user:user" |chpasswd
#RUN echo "user:''" |chpasswd


USER user