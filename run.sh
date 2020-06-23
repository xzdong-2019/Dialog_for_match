#train
cd  /dfsdata2/dongxz1_data/research/BERT-ResSel

pip install filelock -i https://pypi.tuna.tsinghua.edu.cn/simple
#
python main.py --model bert_base_ft --train_type fine_tuning --bert_pretrained bert-base-uncased


# test_base
#python main.py --model bert_base_ft --train_type fine_tuning --bert_pretrained bert-base-uncased --evaluate results/bert_base_ft/fine_tuning/20200515-173025/checkpoints/checkpoint_2.pth

#
#python main.py --model bert_base_ft --train_type fine_tuning --bert_pretrained bert-base-uncased --evaluate results_v1/bert_base_ft/fine_tuning/20200519-234339/checkpoints/checkpoint_3.pth

#python main.py --model bert_base_ft --train_type fine_tuning --bert_pretrained bert-base-uncased --evaluate results_v2/bert_base_ft/fine_tuning/20200521-171347/checkpoints/checkpoint_3.pth
