FROM --platform=linux/amd64 pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
	
# Set working directory
WORKDIR /workspace

# Copy inference script and requirements
COPY predict.py /workspace/predict.py
COPY predict.sh /workspace/predict.sh
COPY Dockerfile /workspace/Dockerfile
COPY checkpoint /workspace/checkpoint

ENV PATH="/opt/conda/bin:$PATH"


# Install Python dependencies
RUN pip3 install nninteractive

# Copy the prediction script
RUN chmod +x /workspace/predict.sh
