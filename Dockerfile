FROM ghcr.io/open-webui/open-webui

RUN pip install anthropic

CMD [ "bash", "start.sh"]