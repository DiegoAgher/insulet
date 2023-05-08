FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY vae_insulet.ckpt /app/vae_insulet.ckpt
COPY xgbmodel.bin /app/xgbmodel.bin
COPY test.csv /app/test.csv
COPY test-img/ /app/test-img/


# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the command to start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
