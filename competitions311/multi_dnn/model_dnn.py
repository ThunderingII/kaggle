import numpy as np
import os, time, sys 
import tensorflow as tf
from data_process import batch_yield 
from utils import get_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
class mul_dnn(object):
	"""docstring for mul_dnn"""
	def __init__(self, args, tag2label, paths, config):
		self.batch_size = args.batch_size
		self.epoch_num = args.epoch
		self.optimier = args.optimizer
		self.hidden_dim = args.hidden_dim
		self.dropout_keep_prob = args.dropout
		self.beta = args.beta
		self.lr = args.lr
		self.clip_grad = args.clip
		self.optimizer = args.optimizer
		self.tag2label = tag2label
		self.num_tags = len(tag2label)

		self.config = config
		self.model_path = paths['model_path']
		self.summary_path = paths['summary_path'] 
		self.logger = get_logger(paths['log_path'])
		self.result_path = paths['result_path']
	def build_graph(self):
		self.add_placeholders()
		self.mul_dnn_op()
		self.softmax_pred_op()
		self.loss_op()
		self.trainstep_op()
		self.init_op()
	def mul_dnn_op(self):
		with tf.variable_scope('multi_dnn'):
			W1 = tf.get_variable(name='W1',
								shape=[43, self.hidden_dim],
								initializer=tf.contrib.layers.xavier_initializer(),
								dtype=tf.float32)
			b1 = tf.get_variable(name='b1',
								shape=[self.hidden_dim],
								initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
			W2 = tf.get_variable(name='W2',
								shape=[self.hidden_dim, self.hidden_dim],
								initializer=tf.contrib.layers.xavier_initializer(),
								dtype=tf.float32)
			b2 = tf.get_variable(name='b2',
								shape=[self.hidden_dim],
								initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
			W3 = tf.get_variable(name='W3',
								shape=[self.hidden_dim, self.hidden_dim],
								initializer=tf.contrib.layers.xavier_initializer(),
								dtype=tf.float32)
			b3 = tf.get_variable(name='b3',
								shape=[self.hidden_dim],
								initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

			W_out = tf.get_variable(name='W_out',
								shape=[self.hidden_dim, self.num_tags],
								initializer=tf.contrib.layers.xavier_initializer(),
								dtype=tf.float32)
			b_out = tf.get_variable(name='b_out',
								shape=[self.num_tags],
								initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
			hidden_layer1 = tf.matmul(self.features, W1) + b1
			h1 = tf.nn.relu(hidden_layer1)

			hidden_layer2 = tf.matmul(h1, W2) + b2
			h2 = tf.nn.relu(hidden_layer2)

			hidden_layer3 = tf.matmul(h2, W3) + b3
			h3 = tf.nn.relu(hidden_layer3)

			self.logits = tf.matmul(h3, W_out) + b_out
			#L2
			self.regularization = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W_out)

	def add_placeholders(self):
		self.features = tf.placeholder(tf.float32, shape=[None,43], name='features')
		self.labels = tf.placeholder(tf.int32, shape=[None],name='labels')
		self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
		self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

	def softmax_pred_op(self):
		self.pred_label = tf.argmax(self.logits,axis=-1)
		self.pred_label = tf.cast(self.pred_label, tf.int32)

	def loss_op(self):
		#损失函数
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels) + self.beta * self.regularization
		self.loss = tf.reduce_mean(losses)
		tf.summary.scalar('loss',self.loss)

	def trainstep_op(self):
		with tf.variable_scope('train_step'):
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			if self.optimizer == 'Adam':
				optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
			elif self.optimizer == 'Adadelta':
				optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
			elif self.optimizer == 'Adagrad':
			    optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
			elif self.optimizer == 'RMSProp':
			    optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
			elif self.optimizer == 'Momentum':
			    optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
			elif self.optimizer == 'SGD':
			    optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
			else:
			    optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
			grads_and_vars = optim.compute_gradients(self.loss)
			grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
			self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

	def init_op(self):
		self.init_op = tf.global_variables_initializer()

	def add_summary(self, sess):
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

	def train(self,train,dev):
		saver = tf.train.Saver(tf.global_variables())
		with tf.Session(config=self.config) as sess:
			sess.run(self.init_op)
			self.add_summary(sess)

			for epoch in range(self.epoch_num):
				print('=====epoch====',epoch)
				self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)


	def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
		num_batches = (len(train) + self.batch_size - 1) // self.batch_size
		batches = batch_yield(train, self.batch_size, tag2label)
		for step, (feats, label) in enumerate(batches):
			step_num = epoch * num_batches + step +1
			feed_dict = self.get_feed_dict(feats, label, self.lr, self.dropout_keep_prob)
			_, loss_train, summary, step_num_, test_labels, test_logists = sess.run([self.train_op, self.loss, self.merged, self.global_step, self.labels, self.logits],
														feed_dict=feed_dict)
			if step + 1 == 1 or (step + 1) % 100 == 0 or step + 1 == num_batches:
				self.logger.info('epoch {}, step {},loss: {:.4},global_step: {}'.format( epoch + 1, step + 1,
	                loss_train,step_num))
			self.file_writer.add_summary(summary, step_num)

			if step + 1 == num_batches:
				saver.save(sess, self.model_path, global_step=step_num)

		self.logger.info('===========validation / test===========')
		label_list_dev = self.dev_one_epoch(sess, dev)
		accuracy, precision, recall, f1 = self.evaluate(label_list_dev, dev, epoch)
		print ("accuracy:%.5f,precision:%.5f,recall:%.5f, f1:%.5f" % (accuracy, precision, recall, f1))
		self.logger.info(
		    'accuracy:{},precision:{},recall:{},f1:{}'.format(
		        accuracy, precision, recall, f1
		        )
		    )
		
	def get_feed_dict(self, feats, label=None, lr=None, dropout=None):
		feed_dict = {self.features:feats}
		if label is not None:
			#label = []
			#for label_ in label:
			#	label.append(label_)
			feed_dict[self.labels] = label
		if lr is not None:
			feed_dict[self.lr_pl] = lr
		if dropout is not None:
			feed_dict[self.dropout_pl] = dropout
		return feed_dict


	def dev_one_epoch(self, sess, dev):
		label_list = []
		batches = batch_yield(dev, self.batch_size, self.tag2label)
		for step, (feats, label) in enumerate(batches):
			label_list_ = self.predict_one_batch(sess, feats, label)
			label_list.extend(label_list_)
		return label_list
	def predict_one_batch(self, sess, feats, label):
		feed_dict = self.get_feed_dict(feats, dropout=1.0)
		#print('====feed_dict====',feed_dict)
		label_list = sess.run(self.pred_label, feed_dict=feed_dict)	
		return label_list
	def evaluate(self, label_list, data, epoch=None):
		label2tag = {}
		for tag, label in self.tag2label.items():
			label2tag[label] = tag 
		pred_label = [label2tag[label_] for label_ in label_list]
		true_label = []
		for feats, tag in data:
			true_label.append(tag[0])
		accuracy = 0
		precision = 0
		recall = 0
		f1 = 0
		try:
			accuracy = accuracy_score(true_label,pred_label)
			precision = precision_score(true_label,pred_label,average='micro')
			recall = recall_score(true_label,pred_label,average='micro')
			f1 = f1_score(true_label,pred_label,average='micro')
		except ValueError:
			print('===true_label===',true_label)
			print('===pred_label===',pred_label)
		return accuracy, precision, recall, f1







