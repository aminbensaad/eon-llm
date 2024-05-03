# app/Dockerfile

FROM python:3.10

WORKDIR /app

# prevent 'Hash Sum mismatch' error
# more here: https://stackoverflow.com/questions/67732260/how-to-fix-hash-sum-mismatch-in-docker-on-mac
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy app files to container
COPY . .

# install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# container listens on Streamlit’s default port at runtime
EXPOSE 8501

# check that container still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# configure executable container 
ENTRYPOINT ["streamlit", "run", "Chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]