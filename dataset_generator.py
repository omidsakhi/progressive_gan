import csv
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2

input_image_folder = 'C:/Projects/datasets/img_align_celeba'
output_image_folder = 'C:/Projects/datasets/img_align_celeba_2'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.uint8)
    return img

def raw_image(addr):
	f = open(addr, "rb")
	b = f.read()
	f.close()
	return b

writer = None
file_num = 0

def remove_blanks(a_list):
    new_list = []
    for item in a_list:
        if item != "":
            new_list.append(item)
    return new_list

with open(input_image_folder + '/list_attr_celeba.txt', newline='') as csvfile:
  csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
  attr_names = []
  for idx1, row in tqdm(enumerate(csvreader)):  	
    if (idx1 == 0):
      writer = tf.python_io.TFRecordWriter('data_' + str(file_num) + '.tfrecords')
    elif (idx1 == 1):
      attr_names = row        
    else:
      if (idx1 % 1000 == 0):
        writer.close()
        file_num = file_num + 1
        writer = tf.python_io.TFRecordWriter('data_' + str(file_num) + '.tfrecords')

      row = remove_blanks(row)        
      img218x178 = load_image(input_image_folder + '/' + row[0])
      img178x178 = img218x178[20:218-20,:,:]
      img128x128 = cv2.resize(img178x178, (128, 128))
      img64x64 = cv2.resize(img128x128, (64, 64))
      img32x32 = cv2.resize(img64x64, (32, 32))
      img16x16 = cv2.resize(img32x32, (16, 16))
      img8x8 = cv2.resize(img16x16, (8, 8))
      cv2.imwrite(output_image_folder + '/' + row[0], img128x128); 
      attrs = map(int, row[1:])
      attrs = map(lambda x: 0 if x == -1 else 1, attrs)
      attrs = np.array(list(attrs), np.uint8)                
      encode_param95 = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
      encode_param100 = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
      _, bImg128x128 = cv2.imencode('.jpg', img128x128, encode_param95)
      _, bImg64x64 = cv2.imencode('.jpg', img64x64, encode_param95)
      _, bImg32x32 = cv2.imencode('.jpg', img32x32, encode_param95)
      _, bImg16x16 = cv2.imencode('.jpg', img16x16, encode_param100)
      _, bImg8x8 = cv2.imencode('.jpg', img8x8, encode_param100)
      feature = {
      'name' : _bytes_feature(tf.compat.as_bytes(row[0])),
      'image': _bytes_feature(tf.compat.as_bytes(bImg128x128.tostring())),
      'image64': _bytes_feature(tf.compat.as_bytes(bImg64x64.tostring())),
      'image32': _bytes_feature(tf.compat.as_bytes(bImg32x32.tostring())),
      'image16': _bytes_feature(tf.compat.as_bytes(bImg16x16.tostring())),
      'image8': _bytes_feature(tf.compat.as_bytes(bImg8x8.tostring())),
      'labels': _bytes_feature(tf.compat.as_bytes(attrs.tostring()))
      }     
      example = tf.train.Example(features=tf.train.Features(feature=feature))    
      writer.write(example.SerializeToString())

writer.close()