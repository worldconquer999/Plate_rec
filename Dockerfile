FROM ai-env

USER root

ENV PYTHONPATH=/ai-engine
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /ai-engine

COPY ./ /ai-engine

EXPOSE 8003

CMD ["python3", "/ai-engine/src/app.py"]