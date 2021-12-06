#Finetune

This is continue train `bert`.
```
# Continue train Bert
python preprocess.py --corpus_path corpora/petpedia.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target mlm
```

```
python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_tiny.bin \
                    --vocab_path models/google_zh_vocab.txt --config_path models/bert/tiny_config.json \
                    --output_model_path models/pet_roberta_tiny.bin --world_size 2 --gpu_ranks 0 1 \
                    --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm


python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_small.bin \
                    --vocab_path models/google_zh_vocab.txt --config_path models/bert/small_config.json \
                    --output_model_path models/pet_roberta_small.bin --world_size 2 --gpu_ranks 0 1 \
                    --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm


python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_medium.bin \
                    --vocab_path models/google_zh_vocab.txt --config_path models/bert/medium_config.json \
                    --output_model_path models/pet_roberta_medium.bin --world_size 2 --gpu_ranks 0 1 \
                    --total_steps 200000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm


python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_base.bin \
                    --vocab_path models/google_zh_vocab.txt --config_path models/bert/base_config.json \
                    --output_model_path models/pet_roberta_base.bin --world_size 2 --gpu_ranks 0 1 \
                    --total_steps 200000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```
This is continue train whole-word-mask bert
```
python preprocess.py --corpus_path corpora/petpedia.txt --spm_model_path models/cluecorpussmall_spm.model \
                      --dataset_path dataset.pt --processes_num 8 --target mlm
python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_word_tiny.bin \
                    --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/tiny_config.json \
                    --output_model_path models/pet_roberta_word_tiny.bin --world_size 1 --gpu_ranks 0\
                    --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
                    
python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_word_small.bin \
                    --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/small_config.json \
                    --output_model_path models/pet_roberta_word_small.bin --world_size 1 --gpu_ranks 0\
                    --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
                    
python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_word_medium.bin \
                    --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/medium_config.json \
                    --output_model_path models/pet_roberta_word_medium.bin --world_size 1 --gpu_ranks 0\
                    --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
  
python pretrain.py --dataset_path dataset.pt --pretrained_model_path models/chinese_roberta_word_base.bin \
                    --spm_model_path models/cluecorpussmall_spm.model --config_path models/bert/base_config.json \
                    --output_model_path models/pet_roberta_word_base.bin --world_size 1 --gpu_ranks 0\
                    --total_steps 200000 --save_checkpoint_steps 10000 --batch_size 64 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm
```
This is finetune simCSE.
```
python finetune/run_simcse.py --pretrained_model_path ../pretrained_model/pet_roberta_base/pytorch_model.bin \
                               --vocab_path models/google_zh_vocab.txt \
                               --config_path ../pretrained_model/pet_roberta_base/config.json \
                               --train_path ../ai-petpedia/data/similarity/pet_unsup.tsv \
                               --dev_path ../ai-petpedia/data/similarity/dev.tsv \
                               --seq_length 64 --learning_rate 1e-5 --batch_size 64 --epochs_num 3 \
                               --pooling first --temperature 0.05
                               

python scripts/convert_bert_from_uer_to_huggingface.py \
      --input_model_path models/finetuned_model.bin \
      --output_model_path ../pretrained_model/simCSE/pytorch_model.bin  \
      --layers_num 12 
````