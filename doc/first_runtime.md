# Pet Robot
# Work Flow
## Preprocess
note: all detail config need set in `config/defaults.py`. It better to run `make clean` before any action.
TODO: fellow steps will induce to make file
TODO: try to rebuild the code in c++

1.  run `python qa/queryUnderstanding/preprocess/unionData.py` to get all data. Processed data will be save in mongo. If first running, run `python qa/queryUnderstanding/preprocess/tsc_process.py` to save related data to mongo.

2.  run `python qa/queryUnderstanding/querySegmentation/newWordDiscovery.py`, then add new word into `custom.txt`. 

3.  run `python qa/queryUnderstanding/representation/word2vec.py` to train word2vec embedding.

4.  run `python qa/retrieval/semantic/hnsw.py` to train hnsw model.
TODO: HNSW do not load data, get data via mongo

5.  run `python qa/queryUnderstanding/queryReformat/queryNormalization/synonym_detection.py` to get synonym word. Before running, it's better to set which words need to find synonym word in `data/synonym/synonym.txt`. Because this is unsupervised method. You need filter one more time.
TODO: NEED to run again

6. run `python qa/queryUnderstanding/querySuggest/query_during_guide.py` to generate profix data

7. run `python qa/contentUnderstanding/content_profile.py` to tag question.

## content profile
1. TODO: try to build profile for question&answer data.

## search
1.  run `python qa/server/create_index.py` to build inverted index then save on mongo. Now you can retrieve data with lexical way.  

2.  TODO: use char word2vec-cbow model to detect whether two question is similarity is better.

3.  run `qa/queryUnderstanding/queryReformat/queryCorrection/correct.py` to generate correct model.  This code contains pycorrect, so you need run it to install model before start service.

4.  TODO: manual retrieval, maybe support `!`, `|`, `&` search language

5.  TODO: deeplearning matching model

6.  TODO: add one route recall for profile


## Intent
1. Visit `https://github.com/codemayq/chinese_chatbot_corpus`, and get all chitchat data. 