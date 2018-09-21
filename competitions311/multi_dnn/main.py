#encoding:utf-8
import tensorflow as tf
import numpy as np   
import os, argparse, time, random
from model_dnn import mul_dnn
from utils import get_logger, get_entity
from data_process import tag2label, read_corpus
#session configuration
config = tf.ConfigProto()
#hyperparameters
parser = argparse.ArgumentParser(description='mul-dnn for recommendation')
parser.add_argument('--train_data', type=str, default='./data', help='train data source')
parser.add_argument('--test_data', type=str, default='./data', help='test data source')
parser.add_argument('--batch_size', type=int, default=128, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta', type=float, default=0.001, help='l2 beta')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--hidden_dim', type=int, default=10, help='#dim of hidden state')

args = parser.parse_args()
## paths setting
paths = {}
timestamp = str(int(time.time()))
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
train_data = read_corpus(train_path)

print("train data: {}".format(len(train_data)))
train = train_data[:600000]
test = train_data[600000:]
model = mul_dnn(args, tag2label, paths, config=config)
model.build_graph()
model.train(train=train, dev=test)













