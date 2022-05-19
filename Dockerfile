FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

RUN apt-get update

RUN apt-get install python3

RUN apt-get install python3-pip