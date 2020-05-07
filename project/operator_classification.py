import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, transform, morphology
import skimage.io
from sklearn.manifold import TSNE
import cv2
from sklearn.metrics import mean_squared_error

def pre_process(im):
    _,ret_im = cv2.threshold(im, 130, 255, cv2.THRESH_BINARY_INV)
    return ret_im

def operator_classify(inp_img):
	col_dir = 'data/train_operators/'
	plus = cv2.imread(col_dir + str(1) + '.png', 0)
	div = cv2.imread(col_dir + str(2) + '.png', 0)
	eq = cv2.imread(col_dir + str(3) + '.png', 0)
	mult = cv2.imread(col_dir + str(4) + '.png', 0)
	minus = cv2.imread(col_dir + str(5) + '.png', 0)

	_,plus = cv2.threshold(plus, 130, 255, cv2.THRESH_BINARY_INV)
	_,div = cv2.threshold(div, 130, 255, cv2.THRESH_BINARY_INV)
	_,eq = cv2.threshold(eq, 130, 255, cv2.THRESH_BINARY_INV)
	_,mult = cv2.threshold(mult, 130, 255, cv2.THRESH_BINARY_INV)
	_,minus = cv2.threshold(minus, 130, 255, cv2.THRESH_BINARY_INV)

	images = [plus, div, eq, mult, minus]

	moments = np.zeros([5,7])
	for i, im in enumerate(images):
		moments[i,:] = cv2.HuMoments(cv2.moments(im)).flatten()

	dict_ = {0:'plus',1:'div',2:'eq',3:'mult',4:'minus'}

	inp_thresh = pre_process(inp_img)
	inp_moment = cv2.HuMoments(cv2.moments(inp_thresh)).flatten()
	dist = np.zeros(5)
	for i in range(5):
		dist[i] = mean_squared_error(inp_moment,moments[i,:])
	min_index = np.argmin(dist)

	return dict_[min_index]

if __name__ == '__main__':
	test_dir = 'data/test_operators/'
	test = cv2.imread(test_dir + str(3) + '.png', 0)
	print(operator_classify(test))
	
