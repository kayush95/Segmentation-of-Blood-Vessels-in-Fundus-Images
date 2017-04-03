from __future__ import division
import os
import sys
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange

import math
import json
import random
import pprint
import scipy.misc

from time import gmtime, strftime
import glob

from ops import *
from utils import *
import evaluate 
#from inception_score import inception_score
n_labels = 2
F = tf.app.flags.FLAGS

class PreGAN(object):
	def __init__(self, sess):
		self.sess = sess
		if F.dataset != "lsun" and F.inc_score:
			print("Loading inception module")
			self.inception_module = inception_score(self.sess)
			print("Inception module loaded")
		self.build_model()

	def build_model(self):
		self.images = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, F.c_dim], name='real_images')

		self.labels = tf.placeholder(tf.int64, [F.batch_size], name='real_labels')  
		self.labels_1_hot = tf.one_hot(self.labels, depth=n_labels+1, dtype = tf.float32)
		self.z = tf.placeholder(tf.float32, [None, F.z_dim], name='z')
		self.keep_prob = tf.placeholder(tf.float32)

		self.G_mean, self.G_var, self.G_sample = self.generator(self.z)
		self.D, self.D_logits = self.discriminator(self.images, self.keep_prob, reuse=False)
		self.D_, self.D_logits_, = self.discriminator(self.G_mean,self.keep_prob,  reuse=True)

		#self.d_sum = tf.histogram_summary("d", self.D)
		#self.d__sum = tf.histogram_summary("d_", self.D_)
		#self.G_sum = tf.image_summary("G_mean", self.G_mean)
		#self.G_v_sum = tf.histogram_summary("G_var", self.G_var)
		#self.G_v_mean = tf.scalar_summary("G_v_mean", tf.reduce_mean(self.G_var))

		self.d_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels_1_hot,logits =self.D_logits))
		
		
		self.fake_label = n_labels*tf.ones((F.batch_size,), dtype = tf.uint8)   # index is num_classes -1 always in 1-hot change
		self.one_hot_fake = tf.one_hot(self.fake_label, depth = n_labels+1, dtype = tf.float32)  #change
		self.d_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels = self.one_hot_fake,logits = self.D_logits_ ))
		self.d_loss = self.d_loss_real + self.d_loss_fake

		#========== Calculating generator loss ===================
				
		self.g_loss_actual = -1* tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(labels = self.one_hot_fake ,logits = self.D_logits_))
				
		#=============================================			
		self.g_loss = tf.constant(0)  # please ignore for time being

		#  Calculate batchwise classification accuracy based on segmentation labels on real images==========
		self.D_class_labels_tensor = tf.slice(self.D, begin=(0,0), size=(-1,2))  #change
		self.D_class_labels = tf.argmax(self.D_class_labels_tensor, axis = 1)   #axis = 1
		#=============================================

		####################### tensorboard visualiztion ##########################
		self.correct_prediction = tf.equal(self.D_class_labels, self.labels) 
		self.accuracy_batch = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		# self.train_accuracy = tf.scalar_summary("train_accuracy", self.accuracy_batch)
		tf.summary.scalar("train_accuracy", self.accuracy_batch)
		self.summary_op = tf.summary.merge_all()
		###########################################################################	
		

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

	def train(self):
		"""Train DCGAN"""

		data = dataset()

		global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")

		# d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D) \
		#                   .minimize(self.d_loss, var_list=self.d_vars)
		# g_optim = tf.train.AdamOptimizer(F.learning_rate_G, beta1=F.beta1G) \
		#                   .minimize(self.g_loss_actual, var_list=self.g_vars)

		learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
							decay_steps=F.decay_step,
							decay_rate=F.decay_rate, staircase=True)
		learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
							 decay_steps=F.decay_step,
							 decay_rate=F.decay_rate, staircase=True)
		d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
			.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
			.minimize(self.g_loss_actual, var_list=self.g_vars)

		# tf.initialize_all_variables().run()
		tf.global_variables_initializer().run()

		#self.g_sum = tf.merge_summary(
		#	[self.z_sum, self.d__sum, self.G_sum, self.G_v_sum, self.G_v_mean, self.d_loss_fake_sum,
		#	 self.g_loss_actual_sum])
		#self.d_sum = tf.merge_summary(
		#	[self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = tf.summary.FileWriter(F.log_dir, self.sess.graph)

		counter = 0
		start_time = time.time()

		if F.load_chkpt:
			try:
				self.load(F.checkpoint_dir)
				print(" [*] Load SUCCESS")
			except:
				print(" [!] Load failed...")
		else:
			print(" [*] Not Loaded")

		self.ra, self.rb = -1, 1

		for epoch in xrange(F.epoch):

			idx = 0
			iscore = 0.0, 0.0 # self.get_inception_score()
			batch_iter = data.batch()

			
						
			temp1 = None
			temp2 = None
			t = 0 
			for sample_images, sample_labels in batch_iter:   #sample_images_files

				#label_array = sample_labels
				#_ , raw_labels = sample_p.where(label_array == 1)
				#print "Original labels::"     
				#print raw_labels
				sample_z = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
				
				if t == 0:
					temp1 = sample_images
					temp2 = sample_labels
					t = 1
				# data = [get_image(sample_images_file, 32, is_crop=False, resize_w=32, is_grayscale = True) for sample_images_file in sample_images_files]
				# sample_images = np.array(data).astype(np.float32)[:, :, :, None]

				# Update D network     ################ updated ############
				_,  dlossf, summary = self.sess.run(
					[d_optim,  self.d_loss_fake, self.summary_op],
					feed_dict={self.images: sample_images, self.labels: sample_labels, self.z: sample_z,
							   self.keep_prob:0.5, global_step: epoch})
				

				# Update G network
				
				iters = 1
				for i in range(iters):

					sample_z = np.random.uniform(
							self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)

					_,  gloss, dloss = self.sess.run(
					[g_optim, self.g_loss_actual, self.d_loss],
					feed_dict={self.images: sample_images, self.z: sample_z, self.labels: sample_labels,
					   self.keep_prob:0.5, global_step: epoch})
			
				

				errD_fake = self.d_loss_fake.eval({self.z: sample_z, self.labels: sample_labels, self.keep_prob:0.5})
				errD_real = self.d_loss_real.eval({self.images: sample_images, self.labels: sample_labels, self.keep_prob:0.5})
				errG = self.g_loss.eval({self.z: sample_z, self.labels: sample_labels, self.keep_prob:0.5})
				errG_actual = self.g_loss_actual.eval({self.z: sample_z, self.labels: sample_labels, self.keep_prob:0.5})


				pred_class_labels = self.D_class_labels.eval({self.images : sample_images, self.keep_prob:0.5})

				lrateD = learning_rate_D.eval({global_step: epoch})
				lrateG = learning_rate_G.eval({global_step: epoch})

				#print "Predicted labels:"
				#print pred_class_labels
				# Calculate classification accuracy each batch==============

				############################# previously used #########################################
				correct_rate = (pred_class_labels == sample_labels)
				correct_rate.astype(np.float32)
				accuracy_batch = np.mean(correct_rate)
				# train_accuracy_batch = self.accuracy_batch.eval({self.images : sample_images, self.labels: sample_labels, self.keep_prob:0.5})
				self.writer.add_summary(summary, epoch*data.num_batches + idx)
				###################################################################

				#print "accuracy: ", accuracy_batch
				counter += 1
				idx += 1
				print(("Epoch:[%2d] [%4d/%4d] l_D:%.2e l_G:%.2e d_loss_f:%.8f d_loss_r:%.8f " +
					  "accuracy:%.8f g_loss_act:%.8f  iscore:%f %f")
					  % (epoch, idx, data.num_batches, lrateD, lrateG, errD_fake,
						 errD_real, accuracy_batch, errG_actual,  iscore[0], iscore[1]))

				if np.mod(counter, 100) == 1:
					samples, d_loss, g_loss = self.sess.run(
						[self.G_mean, self.d_loss, self.g_loss_actual],
						feed_dict={self.z: sample_z, self.images: sample_images, 
														  self.labels: sample_labels, self.keep_prob:0.5})

					save_images(samples, [8, 8], F.sample_dir + "/sample.png")
					# save_images(sample_images, [8,8], F.sample_dir + "original.png")
					print "New samples stored"

				if np.mod(counter, 100) == 1:
					self.save(F.checkpoint_dir)
					print("Saving checkpoints ")
										
								  
			
			
			sample_z = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
			sample_images = temp1
			sample_labels = temp2
			samples, d_loss, g_loss = self.sess.run(
				[self.G_mean, self.d_loss, self.g_loss_actual],
				feed_dict={self.z: sample_z, self.images: sample_images, self.keep_prob:0.5, self.labels: sample_labels})

			save_images(samples, [8, 8], F.sample_dir + "/train_{:03d}.png".format(epoch))

			print "Samples stored at end of epoch::"
			if epoch % 5 == 0:
				iscore = self.get_inception_score()
			if epoch % 10 == 1 :
				evaluate.main()

	def get_inception_score(self):
		if F.dataset == "lsun" or not F.inc_score:
			return 0.0, 0.0

		samples = []
		for k in range(50000 // F.batch_size):
			sample_z = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
			images = self.sess.run(self.G_mean, {self.z: sample_z})
			samples.append(images)
		samples = np.vstack(samples)
		return self.inception_module.get_inception_score(samples)

	def discriminator(self, image, keep_prob, reuse=False):
		with tf.variable_scope('D'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			if F.dataset == "lsun":
				dim = 64
				h0 = lrelu(batch_norm(conv2d(image, dim, name='d_h0_conv')))
				h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv')))
				h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv')))
				h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv')))
				h3 = tf.reshape(h3, [F.batch_size, -1])
				h3 = minibatch_disc(h3, num_kernels=5, scope='d_mb3')
				h4 = linear(h3, 1, 'd_h3_lin')
				return tf.nn.sigmoid(h4), h4

			elif F.dataset == "retina_old":
				dim = 96
				h0 = lrelu(instance_norm(conv2d(image, dim, 3, 3, 1, 1, name='d_h0_conv')))
				h1 = lrelu(instance_norm(conv2d(h0, dim, 3, 3, 1, 1, name='d_h1_conv')))
				h2 = lrelu(instance_norm(conv2d(h1, dim, 3, 3, 2, 2, name='d_h2_conv')))
				h2 = tf.nn.dropout(h2, keep_prob)
				h3 = lrelu(instance_norm(conv2d(h2, 2 * dim, 3, 3, 1, 1, name='d_h3_conv')))
				h4 = lrelu(instance_norm(conv2d(h3, 2 * dim, 3, 3, 1, 1, name='d_h4_conv')))
				h5 = lrelu(instance_norm(conv2d(h4, 2 * dim, 3, 3, 2, 2, name='d_h5_conv')))
				h5 = tf.nn.dropout(h5, keep_prob)
				h6 = lrelu(instance_norm(conv2d(h5, 2 * dim, 3, 3, 1, 1, name='d_h6_conv')))
				h7 = lrelu(instance_norm(conv2d(h6, 2 * dim, 1, 1, 1, 1, name='d_h7_conv')))
				h8 = tf.reduce_mean(h7, [1, 2])
				#h8 = minibatch_disc(h8, num_kernels=50, scope="d_mb_8")
				h9 = linear(h8, n_labels+1, 'd_h9_lin')
				return tf.nn.sigmoid(h9), h9


			elif F.dataset == "retina":
				# keep_prob = 0.5
				fc_hidden_units1 = 512
				train_phase = True
				PATCH_DIM = 32 
				NUM_CLASSES = 3

				with tf.variable_scope('d_h_conv1') as scope:
					weights = tf.get_variable('weights', shape=[4, 4, 1, 64], 
											  initializer=tf.contrib.layers.xavier_initializer_conv2d())
					biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))
													
					# Flattening the 3D image into a 1D array
					x_image = tf.reshape(image, [-1, PATCH_DIM, PATCH_DIM, 1])		        
					# x_image_bn = batch_norm_layer(x_image, train_phase, scope.name)	
					x_image_bn = x_image		        
					z = tf.nn.conv2d(x_image_bn, weights, strides=[1, 1, 1, 1], padding='VALID') + biases			        
					h_conv1 = tf.nn.relu(z, name=scope.name)
					
				with tf.variable_scope('d_h_conv2') as scope:
					weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
											  initializer=tf.contrib.layers.xavier_initializer_conv2d())
					biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))			   			        
					# h_conv1_bn = batch_norm_layer(h_conv1, train_phase, scope.name)
					h_conv1_bn = h_conv1
					z = tf.nn.conv2d(h_conv1_bn, weights, strides=[1, 1, 1, 1], padding='SAME')+biases			        
					h_conv2 = tf.nn.relu(z, name=scope.name)	

				h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
				
				with tf.variable_scope('d_h_conv3') as scope:
					weights = tf.get_variable('weights', shape=[4, 4, 64, 64], 
											  initializer=tf.contrib.layers.xavier_initializer_conv2d())
					biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.05))			        			        
					# h_pool1_bn = batch_norm_layer(h_pool1, train_phase, scope.name)
					h_pool1_bn = h_pool1
					z = tf.nn.conv2d(h_pool1_bn, weights, strides=[1, 1, 1, 1], padding='SAME')+biases			        
					h_conv3 = tf.nn.relu(z, name=scope.name)
					
				h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
				
				
				with tf.variable_scope('d_h_fc1') as scope:
					weights = tf.get_variable('weights', shape=[8**2*64, fc_hidden_units1], 
											  initializer=tf.contrib.layers.xavier_initializer())
					biases = tf.get_variable('biases', shape=[fc_hidden_units1], initializer=tf.constant_initializer(0.05))		        			        
					h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])		        			        
					#h_pool2_flat_bn = batch_norm_layer(h_pool2_flat, train_phase, scope.name)		        
					h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases, name = 'h_fc1')			        
					h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
					
					
				with tf.variable_scope('d_h_fc2') as scope:
					weights = tf.get_variable('weights', shape=[fc_hidden_units1, NUM_CLASSES], 
											  initializer=tf.contrib.layers.xavier_initializer())
					biases = tf.get_variable('biases', shape=[NUM_CLASSES])
					#h_fc1_drop_bn = batch_norm_layer(h_fc1, train_phase, scope.name)			        
					logits = (tf.matmul(h_fc1_drop, weights) + biases)
					
				return tf.nn.sigmoid(logits), logits


			elif F.dataset == "mnist":
				h0 = tf.nn.tanh(linear(tf.reshape(image, [F.batch_size, -1]), 1000, 'd_h0_lin'))
				#h0 = minibatch_disc(h0, scope='d_h0_mbd')
				h1 = tf.nn.tanh(linear(h0, 1000, 'd_h1_lin'))
				h2 = linear(h1, 11, 'd_h2_lin')
				return tf.nn.sigmoid(h2), h2

			else:
				h0 = tf.nn.tanh(linear(tf.reshape(image, [F.batch_size, -1]), 1000, 'd_h0_lin'))
				h0 = minibatch_disc(h0, scope='d_h0_mbd')
				h1 = tf.nn.tanh(linear(h0, 1000, 'd_h1_lin'))
				h2 = linear(h1, 11, 'd_h2_lin')
				return tf.nn.sigmoid(h2), h2

	def generator(self, z):
		with tf.variable_scope("G"):
			if F.dataset == "lsun":
				s = F.output_size
				dim = 64
				s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

				h0 = linear(z, dim * 8 * s16 * s16, scope='g_h0_lin')
				h0 = tf.reshape(h0, [-1, s16, s16, dim * 8])
				h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0))

				h1 = deconv2d(h0, [F.batch_size, s8, s8, dim * 4], name='g_h1')
				h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

				h2 = deconv2d(h1, [F.batch_size, s4, s4, dim * 2], name='g_h2')
				h2 = tf.nn.relu(batch_norm(name='g_bn2')(h2))

				h3 = deconv2d(h2, [F.batch_size, s2, s2, dim * 1], name='g_h3')
				h3 = tf.nn.relu(batch_norm(name='g_bn3')(h3))

				h4 = deconv2d(h3, [F.batch_size, s, s, F.c_dim], name='g_h4')
				h4 = tf.nn.tanh(batch_norm(name='g_bn4')(h4))
				mean = h4

				vh0 = linear(z, dim * 4 * s16 * s16, scope='g_vh0_lin')
				vh0 = tf.reshape(vh0, [-1, s16, s16, dim * 4])
				vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

				vh1 = deconv2d(vh0, [F.batch_size, s8, s8, dim * 2], name='g_vh1')
				vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

				vh2 = deconv2d(vh1, [F.batch_size, s4, s4, dim * 1], name='g_vh2')
				vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

				vh3 = deconv2d(vh2, [F.batch_size, s2, s2, dim / 2], name='g_vh3')
				vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))

				vh4 = deconv2d(vh3, [F.batch_size, s, s, F.c_dim], name='g_vh4')
				vh4 = tf.nn.relu(batch_norm(name='g_vbn4')(vh4))
				var = vh4 + F.eps

				eps = tf.random_normal(mean.get_shape())
				sample = mean + eps * var
				return mean, var, sample

			elif F.dataset == "retina":
				s = F.output_size
				dim = 64
				s2, s4, s8 = int(s / 2), int(s / 4), int(s / 8)

				h0 = linear(z, dim * 8 * s8 * s8, scope='g_h0_lin')
				h0 = tf.reshape(h0, [-1, s8, s8, dim * 8])
				h0 = tf.nn.relu(instance_norm(h0))

				h1 = deconv2d(h0, [F.batch_size, s4, s4, dim * 4], name='g_h1')
				h1 = tf.nn.relu(instance_norm(h1))

				h2 = deconv2d(h1, [F.batch_size, s2, s2, dim * 2], name='g_h2')
				h2 = tf.nn.relu(instance_norm(h2))

				h3 = deconv2d(h2, [F.batch_size, s, s, F.c_dim], name='g_h3')
				h3 = tf.nn.tanh(instance_norm(h3))
				mean = h3

				vh0 = linear(z, dim * 4 * s8 * s8, scope='g_vh0_lin')
				vh0 = tf.reshape(vh0, [-1, s8, s8, dim * 4])
				vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

				vh1 = deconv2d(vh0, [F.batch_size, s4, s4, dim * 2], name='g_vh1')
				vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

				vh2 = deconv2d(vh1, [F.batch_size, s2, s2, dim], name='g_vh2')
				vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

				vh3 = deconv2d(vh2, [F.batch_size, s, s, F.c_dim], name='g_vh3')
				vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))
				var = vh3 + F.eps

				eps = tf.random_normal(mean.get_shape())
				sample = mean + eps * var
				return mean, var, sample

			else:
				h0 = tf.nn.tanh(linear(z, 1000, 'g_h0_lin'))
				h1 = tf.nn.tanh(linear(h0, 1000, 'g_h1_lin'))
				h2 = tf.nn.tanh(linear(h1, 784, 'g_h2_lin'))
				mean = h2
				mean = tf.reshape(mean, [F.batch_size, 28, 28, 1])

				vh0 = tf.nn.tanh(linear(z, 1000, 'g_vh0_lin'))
				vh1 = tf.nn.tanh(linear(vh0, 1000, 'g_vh1_lin'))
				vh2 = tf.nn.relu(linear(vh1, 784, 'g_vh2_lin', bias_start=F.var))
				var = vh2 + F.eps
				var = tf.reshape(var, [F.batch_size, 28, 28, 1])
				eps = tf.random_normal(mean.get_shape())

				sample = mean + eps * var
				return mean, var, sample

	def save(self, checkpoint_dir):
		model_name = "model.ckpt"
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

