FROM achiaracapuano/nlp-python-base:nlp-3.7.6-buster
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
RUN pip install pandas
RUN pip install requests
RUN pip install numpy
RUN pip install nltk
RUN python -m nltk.downloader stopwords
EXPOSE 5000
CMD python main.py


#docker build -f ./docker_files/Dockerfile -t achiaracapuano/indiegogo:flask  ./docker_files/
#docker push achiaracapuano/indiegogo:flask