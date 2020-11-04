FROM ubuntu:20.04
WORKDIR /home/rim/chatbot/IT-Chatbot 
COPY app.py .
COPY requirement.txt .
RUN apt-get update && \
apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r requirement.txt
EXPOSE 5000
CMD ["python3","./app.py"]
