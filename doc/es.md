# ElasicSearch

```

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.15.1-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/kibana/kibana-7.15.1-linux-x86_64.tar.gz
tar -xvf elasticsearch-7.15.1-linux-x86_64.tar.gz
tar -xvf kibana-7.15.1-linux-x86_64.tar.gz
vi config/jvm.options
https://blog.csdn.net/csdn_20150804/article/details/107917560
https://blog.csdn.net/weixin_39168541/article/details/120352944

./elasticsearch -d
sudo nohup bin/kibana --allow-root &
curl -X DELETE http://elastic:ABCabc123@10.2.0.55:9200/_index_template/qa_template?pretty

curl -XPUT http://elastic:ABCabc123@172.28.29.249:9200/_template/qa_template/?pretty  -H 'content-Type:application/json'  -d  '
{
  "order": 0,
  "index_patterns": [
    "qa_*"
  ],
  "settings": {
    "index": {
      "analysis": {
        "analyzer": {
          "blank": {
            "pattern": " ",
            "type": "pattern"
          }
        }
      },
      "number_of_shards": "5",
      "number_of_replicas": "1"
    }
  },
  "mappings": {
    "numeric_detection": true,
    "_source": {
      "enabled": true
    },
    "dynamic": "true",
    "date_detection": false,
    "properties": {
      "answer": {
        "type": "text"
      },
      "updatetime": {
        "format": "epoch_millis",
        "type": "date"
      },
      "question_fine_cut": {
        "index": "true",
        "type": "text",
        "fields": {
          "analyzed": {
            "search_analyzer": "blank",
            "analyzer": "blank",
            "type": "text"
          },
          "raw": {
            "type": "keyword"
          }
        }
      },
      "question_rough_cut": {
        "index": "true",
        "type": "text",
        "fields": {
          "analyzed": {
            "search_analyzer": "blank",
            "analyzer": "blank",
            "type": "text"
          },
          "raw": {
            "type": "keyword"
          }
        }
      }
    }
  }
}
'

curl -XGET http://elastic:ABCabc123@10.2.0.55:9200/qa_v1/_search?pretty -H 'content-Type:application/json'  -d '
{
    "query" : { 
        "match" : {
          "question": "狗"
        }
    }
}'

curl -XPUT http://elastic:ABCabc123@172.28.29.249:9200/qa_v1?pretty
curl -XPUT http://elastic:ABCabc123@10.2.0.55:9200/qa_v1?pretty
curl http://elastic:ABCabc123@10.2.0.55:9200/_cat/indices?v

./bin/elasticsearch-setup-passwords interactive
curl -XPOST -H 'Content-type: application/json' -u elastic:ABCabc123 'http://172.28.29.249:9200/_xpack/security/user/qa?pretty' -d '{
   "password" : "ABCabc123",
   "full_name" : "qa",
   "roles" : ["admin"],
   "email" : "qa@qa.com"
}'

```