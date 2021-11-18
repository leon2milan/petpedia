# Pet Robot
Note: `pip` version should be `20.3`， otherwise, you cannot install rasa-x.
# Work Flow


## RASA
TODO: deprecate rasa.
1. run `mkdir rasa & cd rasa & rasa init`

2. run `make train` to train intent model. Before this, you need fill `data/NLU.yml, rule.yml, stories.yml`, `domain.yml`. You can run `make test` to get the evaluation of the model of intention.

3. run `make run` to lunch rasa-x application. You need run `rasa run actions& ` and `./ngrok http 5002` in other terminal.

## Preprocess
note: all detail config need set in `config/defaults.py`. It better to run `make clean` before any action.
TODO: fellow steps will induce to make file
TODO: try to rebuild the code in c++

1.  run `python core/queryUnderstanding/preprocess/unionData.py` to get all data. Processed data will be save in mongo.

2.  run `python core/queryUnderstanding/querySegmentation/newWordDiscovery.py`, then add new word into `custom.txt`. 

3.  run `python core/queryUnderstanding/representation/word2vec.py` to train word2vec embedding.

4.  run `python core/queryUnderstanding/representation/sif.py` train pca model.
NOTE: after crawling large data. This model become worse. 
TODO: Need optimize.

5.  run `python core/retrieval/semantic/hnsw.py` to train hnsw model.
TODO: HNSW do not load data, get data via mongo
6.  run `python core/queryUnderstanding/queryReformat/queryNormalization/synonym_detection.py` to get synonym word. Before running, it's better to set which words need to find synonym word in `data/synonym/synonym.txt`. Because this is unsupervised method. You need filter one more time.
TODO: NEED to run again

## content profile
1. TODO: try to build profile for question&answer data.

## search
1.  run `python core/server/create_index.py` to build inverted index then save on mongo. Now you can retrieve data with lexical way.  

2.  TODO: use char word2vec-cbow model to detect whether two question is similarity is better.

3.  run `core/queryUnderstanding/queryReformat/queryCorrection/correct.py` to generate correct model. 

4.  TODO: manual retrieval, maybe support `!`, `|`, `&` search language

5.  TODO: deeplearning matching model

6.  TODO: add one route recall for profile


## Intent
1. Visit `https://github.com/codemayq/chinese_chatbot_corpus`, and get all chitchat data. 