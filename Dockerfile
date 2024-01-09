FROM python:3.11 AS build

WORKDIR /assets/solr

COPY ./assets/solr .
RUN find ./*/ -name '*.sh' -exec rm '{}' ';'

RUN pip install -r requirements.txt

RUN find /assets/solr -name '*.yaml' -exec python build_schema.py -t localhost -p 8983 -s '{}' -i ';'

WORKDIR /assets/solr/bash
RUN find /assets/solr -name '*.sh' -exec cp '{}' . ';'


FROM solr:9.3

COPY --from=build /assets/solr/bash /etc/solr/conf
COPY --from=build /assets/solr/data /etc/solr/data

CMD [ "bash", "/etc/solr/conf/solr_entry.sh" ]
EXPOSE 8983


