FROM python:3.11

# Create a non-root user
RUN useradd -m -u 1000 user

# Create the application directory and ensure it has the correct permissions
RUN mkdir -p /home/user/app && chown user:user /home/user/app

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch to the non-root user
USER user

# Set the initial working directory
WORKDIR $HOME/app

# Clone the Git repository
RUN git clone https://github.com/brlrb/responsible-ai-assistant.git .

# Change working directory to the 'app/' directory within the cloned repository
WORKDIR $HOME/app/app

# Install requirements
RUN pip install --user -r requirements.txt

# Copy the rest of your application (if needed) into the 'app/' directory
COPY --chown=user . ./

# Command to run your application
CMD ["chainlit", "run", "app.py", "--port", "7860"]