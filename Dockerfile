FROM python:3.7
  
RUN pip3 install --no-cache-dir \
    imageio \
    scipy \
    jupyter \
    matplotlib

COPY *.py /python-scripts/
COPY *.ipynb /notebooks/

ENV PYTHONPATH $PYTHONPATH:/python-scripts
