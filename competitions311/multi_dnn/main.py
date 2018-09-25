#encoding:utf-8
import tensorflow as tf
import numpy as np   
import os, argparse, time, random
from model_dnn import mul_dnn
from utils import get_logger
from data_process import tag2label, read_corpus, read_corpus_test
#session configuration
config = tf.ConfigProto()
#hyperparameters
parser = argparse.ArgumentParser(description='mul-dnn for recommendation')
parser.add_argument('--train_data', type=str, default='./data', help='train data source')
parser.add_argument('--test_data', type=str, default='./data', help='test data source')
parser.add_argument('--batch_size', type=int, default=128, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=20, help='#epoch of training')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta', type=float, default=0.001, help='l2 beta')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--hidden_dim1', type=int, default=60, help='#dim of hidden state')
parser.add_argument('--hidden_dim2', type=int, default=30, help='#dim of hidden state')
parser.add_argument('--hidden_dim3', type=int, default=10, help='#dim of hidden state')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1537768833', help='model for test and demo')

args = parser.parse_args()
## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

#training model
train_path = os.path.join('.',args.train_data,'train_data.in')
test_path = os.path.join('.',args.test_data,'test_data.in')
if args.mode == 'train':
	train_data = read_corpus(train_path)

	print("train data: {}".format(len(train_data)))
	train = train_data[:600000]
	val = train_data[600000:]
	input_size = len(train[0][0])
	print('input_size',input_size)
	model = mul_dnn(args, tag2label, input_size, paths, config=config)
	model.build_graph()
	model.train(train=train, dev=val)
elif args.mode == 'test':
	test_data = read_corpus_test(test_path)
	print("test data: {}".format(len(test_data)))
	ckpt_file = tf.train.latest_checkpoint(model_path)
	print(ckpt_file)
	paths['model_path'] = ckpt_file
	input_size = len(test_data[0])
	print('input_size',input_size)
	model = mul_dnn(args, tag2label, input_size, paths, config=config)
	model.build_graph()
	model.test(test=test_data)
else:
	print('invalid mode parameter')













