from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import os
import sys
import numpy as np
#from skimage import io, color, measure
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from predict import *
import time
from sklearn.metrics import confusion_matrix

test_rgb_dir = os.path.join(os.path.curdir,'data','retina','images')
fov_dir = os.path.join(os.path.curdir,'data','retina','mask')
gt_dir = os.path.join(os.path.curdir,'data','retina','1st_manual')
path_experiment = os.path.join(os.path.curdir,'data','retina','visualization')

F = tf.app.flags.FLAGS  # for flag handling
batch_size =64
patch_size =32
c_dim =1
output_size = patch_size
# checkpoint_dir = '/home/15EC90J02/semi_GAN/checkpoint/retina/'
checkpoint_dir = '/home/bt1/13CS10058/btp/semi_GAN/checkpoint/retina/'


def discriminator_old(image, keep_prob, reuse =None):
  with tf.variable_scope('D'):
	if reuse:
	  tf.get_variable_scope().reuse_variables()
	#h0 = tf.nn.tanh(linear(tf.reshape(image, [batch_size, -1]), 1000, 'd_h0_lin'))
	#h0 =  minibatch_disc(h0, scope='d_h0_mbd')
	#h1 = tf.nn.tanh(linear(h0, 1000, 'd_h1_lin'))
	#h2 = linear(h1, 3, 'd_h2_lin')
	#return tf.nn.sigmoid(h2), h2
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
	
	h9 = linear(h8, 3, 'd_h9_lin')
	return tf.nn.sigmoid(h9), h9

def discriminator(image, keep_prob, reuse =None):
    fc_hidden_units1 = 512
    train_phase = False
    PATCH_DIM = 32 
    NUM_CLASSES = 3

    with tf.variable_scope('D'):
		if reuse:
			tf.get_variable_scope().reuse_variables()



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
		weights = tf.get_variable('weights', shape=[7**2*64, fc_hidden_units1], 
								  initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('biases', shape=[fc_hidden_units1], initializer=tf.constant_initializer(0.05))		        			        
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])		        			        
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



def fg_prob(probs):
  # probs.shape should be BX3 ; B= batch size
  probability = tf.slice(probs, begin =[0,1], size=[-1,1]) # all rows and second col
  return probability

def load(checkpoint_dir, sess, saver):
  print(" [*] Reading checkpoints...")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
	ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
	print "Succesfully loaded check point files:: "
	return True
  else:
	print "No checkpoint files found"
	return False



def main():
  with tf.Graph().as_default():
	images = tf.placeholder(tf.float32, [batch_size, output_size, output_size, c_dim], name='real_images')
	keep_prob = tf.placeholder(tf.float32)
	D, D_logits = discriminator(images, keep_prob, reuse=None)
	vessel_prob = fg_prob(D)

	# Create a saver for writing training checkpoints.
	saver = tf.train.Saver()

	# Create a session for running Ops on the Graph.
	with tf.Session() as sess:
	  #saver.restore(sess, MODEL_PATH)
	  load(checkpoint_dir, sess, saver)  #load saved model 
	  test_imgs = [imgs for imgs in os.listdir(test_rgb_dir) if imgs.endswith('clahe_test.tif')] 
	  print "Total test images found :  ", len(test_imgs[0:2])

	  # Define placeholders for storing performance metrics==========
	  AUC_ROC = np.zeros(len(test_imgs[0:2]))
	  accuracy = np.zeros(len(test_imgs[0:2]))
	  # process each image
	  counter = 0
	  for im in test_imgs[0:2]:
		print "Processing test image:  ", im
		img_number = im[0:im.find('_')]
		test = Image.open(os.path.join(test_rgb_dir, im))
		test_img = np.asarray(test, dtype=np.float32)
		test_img = test_img/127.5 - 1  # to make suitable for discriminator in GAN framework   
		#test_img = test_img_rgb[:,:,2]  # take only green one for toy example. later do histeq
		gt = Image.open(os.path.join(gt_dir, img_number+'_manual1.gif'))
		gt_image = np.asarray(gt, dtype=np.float32)
		gt_image = gt_image/255.
                #print "Range of test and gt", test_img.min(), test_img.max(), gt_image.min(), gt_image.max()
		fov = Image.open(os.path.join(fov_dir, img_number+'_test_mask.gif'))
		fov_image = np.asarray(fov)
		[ori_H, ori_W] = test_img.shape
		test_img = np.reshape(test_img,(ori_H, ori_W, 1))
		#print "** H and W and C:  ", ori_H, ori_W, test_img.shape[2] 
		
		test_img_padded = image_pad(test_img, patch_size)
		p_H, p_W, p_C = test_img_padded.shape
		test_patches = extract_test_patches(test_img_padded, patch_size, ori_H, ori_W)
		print "Patch extraction from test image done:", test_patches.shape
		time.sleep(5)


		gt_img = np.reshape(gt_image,(ori_H, ori_W, 1))
		gt_image_padded = image_pad(gt_img, patch_size)
		assert(p_H == gt_image_padded.shape[0])
		assert(p_W == gt_image_padded.shape[1])
		gt_patches = extract_test_patches(gt_image_padded, patch_size, ori_H, ori_W)
		assert(test_patches.shape == gt_patches.shape)
		print "Patches extracted for gt done: "
		print "Shape of test patches: ", test_patches.shape
		print "Shape of gt_patches: ", gt_patches.shape
		time.sleep(10)
                #print "Range of test and gt", test_img_padded.min(), test_img_padded.max(), gt_image_padded.min(), gt_image_padded.max()
                #============ Save original and padded images and maps
                t= np.reshape((test_img*127.5 +1.0),(ori_H,ori_W)) 
                g =np.reshape(gt_image, (ori_H, ori_W))
                t_p = np.reshape(test_img_padded*127.5+1, (p_H,p_W))
                g_p = np.reshape(gt_image_padded, (p_H, p_W))
                print "test image shape*****", t.shape
                scipy.misc.imsave("test_img.png", t)
                scipy.misc.imsave("gt_img.png", g)
                scipy.misc.imsave("test_img_padded.png", t_p)
                scipy.misc.imsave("gt_img_padded.png", g_p)
                
                #----------------------------------------------------------------------------------
                print "Images dumped success::  "
                #sys.exit()
	   #==== make placeholders for storing batchwise prediction results ===============
		predictions = np.zeros((test_patches.shape[0]))
		total_full_batches = int(predictions.shape[0]/ batch_size)
		left_out = predictions.shape[0]% batch_size   # remaining patches in last fractional patch 
		if left_out == 0:
		  pad_req = 0
		else:
		  pad_req = batch_size - left_out  # dummy samples required to make up a batch size
		pad_samples = np.zeros((pad_req,patch_size,patch_size,1))
		print "Total batches, Left Out, Pad_req: ", total_full_batches,left_out, pad_req
	   #==========================================================================================

		# ============= convert gt patches for gt values based on central pixel value
		gt_values = gt_patch_to_value(gt_patches, patch_size, n_samples = predictions.shape[0])
		print gt_values.shape
		assert(gt_values.shape == predictions.shape)
		#================================================================



		#=============== begin batch wise evaluation =================
		for batch in range(total_full_batches):
		  image_feed = test_patches[batch*batch_size:batch*batch_size+batch_size,:,:,:]
		  preds = sess.run(vessel_prob, feed_dict={images:image_feed, keep_prob:1.0})
		  #print "Shape of preds:  ", preds.shape, preds
		  #if batch ==3:
		  #  break
		  if batch%50 ==0:
			print "Batch number processed:", batch
		  predictions[batch*batch_size:batch*batch_size+batch_size] = preds[:,0]

		dummy_feed = np.concatenate((test_patches[-left_out:], pad_samples), axis=0)
		print "Shape of dummy feed is : ", dummy_feed.shape
		mixed_preds= sess.run(vessel_prob, feed_dict={images:dummy_feed, keep_prob:1.0})
		if left_out ==1:
		  predictions[-left_out:] = mixed_preds[0,0]
		else:
		  predictions[-left_out:] = mixed_preds[0:left_out,0]
		print "Predictions Done"
		
		print "Shape of Predictions: ", predictions.shape
		 
		# stich back predictions and gt_values back to image level for comparison ========
		pred_image = recompose(predictions, patch_size, p_H, p_W, ori_H, ori_W)
		gtruth = recompose(gt_values,  patch_size, p_H, p_W, ori_H, ori_W)
		assert(pred_image.shape == gtruth.shape)
		#=========================================================================

                #===== saving of Recomposed Images and Maps images==================
                pi = np.reshape((pred_image*127.5 + 1.0),(p_H,p_W))
                gtru = np.reshape(gtruth, (p_H,p_W))
                scipy.misc.imsave(img_number+"pred_image.png", pi)
                scipy.misc.imsave(img_number+"gtruth_recomposed.png", gtru)
                #save_images(pred_image, [1,1], "pred_image.png")
                #save_images(gtruth, [1,1], "gtruth_recomposed.png")


		#======================== visualization ==================================
		#visualize(pred_image, path_experiment+"pred_" + str(img_number))#.show()
		#visualize(gtruth, path_experiment+"gt_" + str(img_number))#.show()
		#=========================================================================

		y_scores, y_true = pred_only_FOV(pred_image,gtruth, fov_image)
 
		#Area under the ROC curve
		fpr, tpr, thresholds = roc_curve((y_true), y_scores)
		AUC_ROC[counter] = roc_auc_score(y_true, y_scores)
		# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
		
		# confusion matrix and accuracy ===================================
  
		#Confusion matrix
		threshold_confusion = 0.75
		print "\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion)
		y_pred = np.empty((y_scores.shape[0]))
		for i in range(y_scores.shape[0]):
		  if y_scores[i]>=threshold_confusion:
			y_pred[i]=1
		  else:
			y_pred[i]=0
		confusion = confusion_matrix(y_true, y_pred)
		print confusion
		accuracy[counter] = 0
		if float(np.sum(confusion))!=0:
		  accuracy[counter] = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
		

		#print "Shape of y_scores and y_true:", y_scores.shape, y_true.shape
		print "Completed processing  image number: ", counter
		print "Current  AUC is :", AUC_ROC[counter]
		print "Current Accuracy: ", accuracy[counter]
		counter += 1
  #print "Average AUC is :", np.mean(AUC_ROC)
  #Save the results
  print "Saving the mean performance metrics on file"
  file_perf = open('mean_performances.txt', 'w')
  file_perf.write("Area under the ROC curve: "+str(np.mean(AUC_ROC))
				  +"\nACCURACY: " +str(accuracy)
				  )
  file_perf.close()


if __name__ == "__main__":
	main()
