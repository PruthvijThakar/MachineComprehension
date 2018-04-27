from flask import Flask,render_template, session, request,redirect,url_for,jsonify
from functools import wraps
import pandas as pd
from logger import logger
import datetime
from keras.models import model_from_json
import json
import random
import pickle
from sklearn.externals import joblib
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot,merge, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import ast

app = Flask(__name__)
def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
	X = []
	Xq = []
	for story, query in data:
		x=[word_idx[w] for w in story]
		X.append(x)
		xq=[word_idx[w] for w in query]
		Xq.append(xq)
	return (pad_sequences(X, maxlen=story_maxlen),pad_sequences(Xq, maxlen=query_maxlen))

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '?' in line:
            q=line
            q = tokenize(q)
            substory = None
            substory = [x for x in story if x]
            data.append((substory, q))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
        #print(data)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q) for story, q in data if not max_length or len(flatten(story)) < max_length]
    return data





@app.route('/FinalProjectCognitive-single-supporting-fact',methods=['GET','POST'])
def FinalProjectCognitivesinglesupportingfact():
	return render_template('FinalProjectCognitive-single-supporting-fact.html')

@app.route('/FinalProjectCognitive-qa2_two-supporting-facts',methods=['GET','POST'])
def FinalProjectCognitiveqa2_twosupportingfacts():
	return render_template('FinalProjectCognitive-qa2_two-supporting-facts.html')

@app.route('/FinalProjectCognitive-three-supporting-fact',methods=['GET','POST'])
def FinalProjectCognitivethreesupportingfact():
	return render_template('FinalProjectCognitive-three-supporting-fact.html')


@app.route('/FinalProjectCognitive-two-arg-relations',methods=['GET','POST'])
def FinalProjectCognitivetwoargrelations():
	return render_template('FinalProjectCognitive-two-arg-relations.html')

@app.route('/FinalProjectCognitive-three-arg-relations',methods=['GET','POST'])
def FinalProjectCognitivethreergrelations():
	return render_template('FinalProjectCognitive-three-arg-relations.html')

@app.route('/FinalProjectCognitive-qa6_yes-no-questions',methods=['GET','POST'])
def FinalProjectCognitiveqa6_yesnoquestions():
	return render_template('FinalProjectCognitive-qa6_yes-no-questions.html')

@app.route('/FinalProjectCognitive-qa11_basic-coreference_test',methods=['GET','POST'])
def FinalProjectCognitiveqa11_basiccoreference_test():
	return render_template('FinalProjectCognitive-qa11_basic-coreference_test.html')

@app.route('/FinalProjectCognitive-simple-negation',methods=['GET','POST'])
def FinalProjectCognitivesimplenegation():
	return render_template('/FinalProjectCognitive-simple-negation.html')


@app.route('/',methods=['GET','POST'])
def selectmodel():
	return render_template("home.html")


@app.route('/modelselection', methods=['GET','POST'])
def selectionmade():
	direc =request.form['selectmodel']
	sample=None
	with(open(direc+"/"+'sample.txt','r')) as file:
		sample=file.readlines()
	allowedwords=None
	with(open(direc+"/"+"word_idx.txt",'r')) as file:
		allowedwords=ast.literal_eval(file.readline())

	return render_template("writequestion.html",direc=direc,allowedwords=allowedwords,sample=sample)

@app.route('/predictanswer', methods=['GET','POST'])
def predictanswer():
	question=request.form['question']
	story=request.form['story']

	direc=request.form['direc']
	storynew=[]
	story=story.split("\n")
	for i in story:
		for x in i.split(" "):
			if(x in ['','\r']):
				continue
			if('.' in x):

				storynew.append(x[:x.find('.')])
				storynew.append('.')
			else:
				storynew.append(x)

	questionnew=[]
	if '?' not in question:
		question=question+"?"
	for x in question.split(" "):
		if(x in ['','\r']):
			continue
		if('?' in x):
			questionnew.append(x[:x.find('?')])
			questionnew.append('?')
		else:
			questionnew.append(x)

	finstory=(storynew,questionnew)

	'''
	story=parse_stories(story,only_supporting=False)

	vocab = set()
	for story1, q in story:
		for story2 in story1:
			vocab |= set(story2)
		vocab |=set(q)
	vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	story_maxlen=0
	#print(story)
	for story1,q in story:
		for story2 in story1:
			if(len(story2)>story_maxlen):
				story_maxlen=len(story2)
		query_maxlen=len(q)
	word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
	idx_word = dict((i+1, c) for i,c in enumerate(vocab))'''
	with open(direc+'/'+'word_idx.txt','r') as file:
		word_idx=ast.literal_eval(file.read())
	file.close()
	with open(direc+'/'+'idx_word.txt','r') as file:
		idx_word=ast.literal_eval(file.read())
	file.close()
	with open(direc+'/'+'story_maxlen.txt','r') as file:
		story_maxlen=int(file.readline())
	file.close()
	with open(direc+'/'+'query_maxlen.txt','r') as file:
		query_maxlen=int(file.readline())
	file.close()

	current_story, current_query = vectorize_stories([finstory],word_idx,story_maxlen,query_maxlen)
	json_file = open(direc+'/'+'model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
# load weights into new model
	loaded_model.load_weights(direc+'/'+"model.h5")
	#print(len(current_story))

	#print(len(current_story[0]))
	#print(len(current_query[0]))
	current_prediction = loaded_model.predict([current_story, current_query])
	current_prediction = idx_word[np.argmax(current_prediction)]
	return render_template("test.html",prediction=current_prediction)



if __name__ == "__main__":

	app.secret_key= "Tusharapplicationasddsasadsdasa"
	app.run(port=80,debug=True)
