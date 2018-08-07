import os
import json
from gensim.models import word2vec

homedir=os.environ['HOME']

sentences=word2vec.LineSentence(homedir+"/results/w2v_training_data.txt")

f=open(homedir+"/results/tf_all.json",'r',encoding='utf-8')
tf_all=json.load(f)
f.close()

tf_all_com={}
for k in tf_all.keys():
	tf_all_com[k]=tf_all[k]+1e-10

model=word2vec.Word2Vec(sg=1,size=256,window=3,min_count=0,sample=1e-3,hs=0,negative=5,workers=4,sorted_vocab=1,compute_loss=True)
model.build_vocab_from_freq(tf_all_com)
model.train(sentences,total_examples=24748,epochs=200)
path=homedir+"/results/models/w2v_sg.model"
model.save(path)

model=word2vec.Word2Vec(sg=0,size=256,window=3,min_count=0,sample=1e-3,hs=0,negative=5,workers=4,sorted_vocab=1,compute_loss=True)
model.build_vocab_from_freq(tf_all_com)
model.train(sentences,total_examples=24748,epochs=200)
path=homedir+"/results/models/w2v_cbow.model"
model.save(path)