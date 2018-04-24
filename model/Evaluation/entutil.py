## the entutil class
## Yang Xu
## 8/9/2015

import math
from nltk.util import ngrams
import sys
import numpy
from collections import Counter
vocab_num=50000
# define the entropy util class
class entutil(object):
	"""docstring for entutil"""

	# constructor
	def __init__(self, mode = 1):
		"""
		mode 1: <s> is used for bigram and trigram, but not for unigram
		mode 2: one <s> is inserted before the sentence, and one </s> inserted after
		mode 3: two <s>s inserted before, and one </s> after
		"""
		# ngrams count
		self.uni = {}
		self.bi = {}
		self.tri = {}

		# smoothed ngrams count
		self.uni_s = {}
		self.bi_s = {}
		self.tri_s = {}

		# N vars 
		self.token_N = 0 # count of pure tokens
		self.uni_N = 0 # count of all unigrams
		self.bi_N = 0  # count of all bigrams
		self.tri_N = 0 # count of all trigrams

		self.first_unkown = [] # the word appeared first time will be here, while preparing the vocabulary

		# mode and mode-specific vars
		self.mode = mode
		if mode == 1:
			self.s_N = 0 # record the number of training sentences
			self.V = ['UNK']
		else:
			self.V = ['UNK', '<s>', '</s>']

	def prepVnew(self,fn):
		word_counter = Counter()
		lineno=0
		for line in open(fn,'r'):
			line_words = line.strip().split()
			s = [x for x in line_words]
			word_counter.update(s)
			lineno+=1
			if lineno%10000==0:
				print lineno
		vocab_count = word_counter.most_common(vocab_num)
		for (word, count) in vocab_count:
			if not word in self.V:
				self.V.append(word)
				self.token_N += 1
	# prepares the vocabulary
	def prepV(self, tokens):
		for t in tokens:
			self.token_N += 1
			if t not in self.V:
				if t not in self.first_unkown:
					self.first_unkown.append(t)
				else:
					self.V.append(t)
	
	# trains the models
	def train(self, tokens):
		# replace unkown words with 'UNK'
		tokens = ['UNK' if t not in self.V else t for t in tokens]
		# decide based on mode
		if self.mode == 1:
			tokens = ['<s>'] + tokens
		elif self.mode == 2:
			tokens = ['<s>'] + tokens
			tokens.append('</s>')
		elif self.mode == 3:
			tokens = ['<s>', '<s>'] + tokens
			tokens.append('</s>')
		# train unigram
		if self.mode == 1: # in mode 1, <s> and </s> are not included in unigrams
			for key in tokens:
				if key != '<s>':
					self.uni_N += 1
					self.uni.setdefault(key, 0)
					self.uni[key] += 1
				else:
					self.s_N += 1
		else:
			for key in tokens:
				self.uni_N += 1
				self.uni.setdefault(key, 0)
				self.uni[key] += 1
		# train bigram
		#if self.mode == 2:
		bi_tokens = list(ngrams(tokens, 2))
		for tup in bi_tokens:
			self.bi_N += 1
			key = ' '.join(tup)
			self.bi.setdefault(key, 0)
			self.bi[key] += 1
		# train trigram
		#if self.mode == 3:
		tri_tokens = list(ngrams(tokens, 3))
		for tup in tri_tokens:
			self.tri_N += 1
			key = ' '.join(tup)
			self.tri.setdefault(key, 0)
			self.tri[key] += 1

	# builds bi_s and tri_s using add-one smoothing
	def smooth(self):
		# smoothing unigram
		self.uni_s = self.uni.copy()
		self.uni_s.update((key, float(val + 1) * self.uni_N / (self.uni_N + len(self.V))) \
			for key, val in self.uni_s.iteritems())
		# smoothing bigram
		self.bi_s = self.bi.copy()
		for key, val in self.bi_s.iteritems():
			uni_key = key.split()[0]
			if uni_key == '<s>' and self.mode == 1:
				uni_val = self.s_N
			else:
				uni_val = self.uni_s[uni_key]
			new_val = float(val + 1) * uni_val / (uni_val + len(self.V))
			self.bi_s[key] = new_val
		# smoothing trigram
		self.tri_s = self.tri.copy()
		for key, val in self.tri_s.iteritems():
			bi_key = ' '.join(key.split()[:2])
			bi_val = self.bi_s[bi_key]
			new_val = float(val + 1) * bi_val / (bi_val + len(self.V))
			self.tri_s[key] = new_val

	# computes the entropy of a given sentence
	def compute(self, tokens, smoothed = True):
		H = 0
		H_word=[]
		lambda1 = 0.5
		lambda2 = 0.3
		# return NaN for emptry list
		if len(tokens) == 0:
			return float('nan')
		# replace unkown tokens
		tokens = ['UNK' if t not in self.V else t for t in tokens]
		# add <s> and </s> by mode
		if self.mode == 1:
			pass
		elif self.mode == 2:
			tokens = ['<s>'] + tokens
			tokens.append('</s>')
		elif self.mode == 3:
			tokens = ['<s>', '<s>'] + tokens
			tokens.append('</s>')
		# decide the ngrams to be used
		if smoothed:
			uni_dict = self.uni_s
			bi_dict = self.bi_s
			tri_dict = self.tri_s
		else:
			uni_dict = self.uni
			bi_dict = self.bi
			tri_dict = self.tri
		# compute
		if self.mode == 1: # mode 1
			for i, t in enumerate(tokens):
				# if it is the first word, compute the unigram and the bigram, t1 = <s>
				if i == 0:
					pu = float(uni_dict[t]) / self.uni_N
					bi_key = ' '.join(('<s>', t))
					if bi_key in bi_dict:
						pb = float(bi_dict[bi_key]) / self.s_N
					else:
						pb = 1.0 / (self.s_N + len(self.V))
					# interpolation
					p = (lambda1 + lambda2) * pb + (1 - lambda1 - lambda2) * pu
					H += p*math.log(p, 2)
					H_word.append(-p*math.log(p, 2))
				# if it is the second word, compute the unigram, bigram, and trigram, t2 = <s>
				elif i == 1:
					t1 = tokens[i-1]
					# unigram
					pu = float(uni_dict[t]) / self.uni_N
					# bigram
					bi_key = ' '.join((t1, t))
					if bi_key in bi_dict:
						pb = float(bi_dict[bi_key]) / uni_dict[t1]
					else: # add-one smooth
						pb = 1.0 / (uni_dict[t1] + len(self.V))
					# trigram
					tri_key = ' '.join(('<s>', t1, t))
					bi_key_pre = ' '.join(('<s>', t1))
					if tri_key in tri_dict:
						pt = float(tri_dict[tri_key]) / bi_dict[bi_key_pre]
					elif bi_key_pre in bi_dict:
						pt = 1.0 / (bi_dict[bi_key_pre] + len(self.V))
					else:
						pt = 1.0 / (1.0 * self.s_N / (self.s_N + len(self.V)) + len(self.V))
					# interpolation
					p = lambda1 * pt + lambda2 * pb + (1 - lambda1 - lambda2) * pu
					H += p*math.log(p, 2)
					H_word.append(-p*math.log(p, 2))
				# if it is beyond the third word, compute the unigram, bigram, and trigram
				else:
					t1 = tokens[i-1]
					t2 = tokens[i-2]
					# unigram
					pu = float(uni_dict[t]) / self.uni_N
					# bigram
					bi_key = ' '.join((t1, t))
					if bi_key in bi_dict:
						pb = float(bi_dict[bi_key]) / uni_dict[t1]
					else: # add-one smooth
						pb = 1.0 / (uni_dict[t1] + len(self.V))
					# trigram
					tri_key = ' '.join((t2, t1, t))
					bi_key_pre = ' '.join((t2, t1))
					if tri_key in tri_dict:
						pt = float(tri_dict[tri_key]) / bi_dict[bi_key_pre]
					elif bi_key_pre in bi_dict:
						pt = 1.0 / (bi_dict[bi_key_pre] + len(self.V))
					else:
						pt = 1.0 / (1.0 * uni_dict[t2] / (uni_dict[t2] + len(self.V)) + len(self.V))
					# interpolation
					p = lambda1 * pt + lambda2 * pb + (1 - lambda1 - lambda2) * pu
					#p = pu
					H += p*math.log(p, 2)
					H_word.append(-p*math.log(p, 2))
			#return - H / len(tokens) , H_word
			return - H, H_word
		elif self.mode == 2: # mode 2
			for i, t in enumerate(tokens[1:]): # skip the first <s>
				t1 = tokens[i-1]
				if i == 0: # for the first word, compute only the pu and pb, because it does not form a trigram
					pu = float(uni_dict[t]) / self.uni_N
					bi_key = ' '.join((t1, t))
					if bi_key in bi_dict:
						pb = float(bi_dict[bi_key]) / uni_dict[t1]
					else:
						pb = 1.0 / (uni_dict[t1] + len(self.V))
					# interpolation
					p = (lambda1 + lambda2) * pb + (1 - lambda1 - lambda2) * pu
					H += math.log(p, 2)
				else: # for the words beyond the second one, compute the pu, pb, and pt
					t2 = tokens[i-2]
					# unigram prob
					pu = float(uni_dict[t]) / self.uni_N
					# bigram prob
					bi_key = ' '.join((t1, t))
					if bi_key in bi_dict:
						pb = float(bi_dict[bi_key]) / uni_dict[t1]
					else: # use add-one smooth
						pb = 1.0 / (uni_dict[t1] + len(self.V))
					# trigram prob
					bi_key = ' '.join((t2, t1))
					tri_key = ' '.join((t2, t1, t))
					if tri_key in tri_dict:
						pt = float(tri_dict[tri_key]) / bi_dict[bi_key]
					elif bi_key in bi_dict: # only needs to smooth the trigram term
						pt = 1.0 / (bi_dict[bi_key] + len(self.V))
					else: # needs to smooth both bigram and trigram terms
						pt = 1.0 / (1.0 * uni_dict[t2] / (uni_dict[t2] + len(self.V)) + len(self.V))
					# interpolation
					p = lambda1 * pt + lambda2 * pb + (1 - lambda1 - lambda2) * pu
					H += math.log(p, 2)
			return - H / (len(tokens))
		elif self.mode == 3: # mode 3, 
			for i, t in enumerate(tokens[2:]): # skip the first two <s>s
				t1 = tokens[i-1]
				t2 = tokens[i-2]
				# unigram prob
				pu = float(uni_dict[t]) / self.uni_N
				# bigram prob
				bi_key = ' '.join((t1, t))
				if bi_key in bi_dict:
					pb = float(bi_dict[bi_key]) / uni_dict[t1]
				else: # use add-one smooth
					pb = 1.0 / (uni_dict[t1] + len(self.V))
				# trigram prob
				bi_key = ' '.join((t2, t1))
				tri_key = ' '.join((t2, t1, t))
				if tri_key in tri_dict:
					pt = float(tri_dict[tri_key]) / bi_dict[bi_key]
				elif bi_key in bi_dict: # only needs to smooth the trigram term
					pt = 1.0 / (bi_dict[bi_key] + len(self.V))
				else: # needs to smooth both bigram and trigram terms
					pt = 1.0 / (1.0 * uni_dict[t2] / (uni_dict[t2] + len(self.V)) + len(self.V))
				# interpolation
				p = lambda1 * pt + lambda2 * pb + (1 - lambda1 - lambda2) * pu
				H += math.log(p, 2)
			return - H / (len(tokens))


	# resets all dicts
	def reset(self):
		self.uni = {}
		self.bi = {}
		self.tri = {}

		self.uni_s = {}
		self.bi_s = {}
		self.tri_s = {}

		self.token_N = 0
		self.uni_N = 0
		self.bi_N = 0
		self.tri_N = 0

		self.first_unkown = []

		if self.mode == 1:
			self.s_N = 0
			self.V = ['UNK']
		else:
			self.V = ['UNK', '<s>', '</s>']
			
	
	def computefile(self,fn,smoothed = True):
		scores_sent=[]
		scores_word=[]
		with file(fn,'r') as fin:
			for line in fin:
				line=line.strip()
				ent_sent,ent_words = self.compute(line.split(), smoothed)
				scores_sent.append(ent_sent)
				scores_word+=ent_words
		print 'sentence entropy:', numpy.mean(scores_sent)
		print 'word entropy:', numpy.mean(scores_word)

if __name__=="__main__":
	enter = entutil(mode=1)
	print "building vocabulary..."
	#lineno=0
	#with file(sys.argv[1],'r') as fin:
	#	for line in fin:
	#		lineno+=1
	#		enter.prepV(line.strip().split())
	#		if lineno%1000==0:
	#			print lineno
	enter.prepVnew(sys.argv[1])
	print "training..."
	lineno=0
	with file(sys.argv[1],'r') as fin:
		for line in fin:
			enter.train(line.strip().split())
			lineno+=1
			if lineno%10000==0:
				print lineno
	print 'computing entropy...'

	for i in xrange(2,len(sys.argv)):
		print sys.argv[i]
		enter.computefile(sys.argv[i],smoothed=False)
		print ''
	
