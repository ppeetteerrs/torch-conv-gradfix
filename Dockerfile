FROM ghcr.io/ppeetteerrs/pytorch:latest

RUN pip install "poetry>=1.2.*" poetry-dynamic-versioning

CMD ["zsh"]