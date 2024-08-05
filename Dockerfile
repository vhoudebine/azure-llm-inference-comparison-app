FROM ubuntu:22.04
USER root
COPY . /usr/app/
RUN chmod -R 777 /usr/app/
WORKDIR /usr/app/
RUN apt-get update && apt-get install -y \
python3 \
python3-pip 
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY run.sh run.sh
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]