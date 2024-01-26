# Linux environment
#FROM ubuntu:latest

# Python environment with equivalent version to venv
FROM python:3.7
LABEL MAINTAINER Navya Annapareddy <na3au@virginia.edu>

RUN apt-get update && apt-get -y update
#RUN apt-get install -y build-essential python3.7 python3-pip python3-dev
#RUN pip3 -q install pip --upgrade

RUN mkdir src
WORKDIR src/
COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

WORKDIR /src/images

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Use local ip to contain notebook/python to container
CMD [ "./images/Collaboration_Container.py"]

#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]