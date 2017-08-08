from numpy import *
import matplotlib.pyplot as plt
import numpy as np

def compute_error(b, m, points):
	totalError = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m*x +b )) **2
	return totalError/N

def step_gradient(b_current, m_current, points, learningRate):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - ((m_current*x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current*x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learningRate)
		rows,cols = points.T.shape
		x = np.arange(20, 90, 0.01)
		y = m*x + b
		plt.scatter(points[range(0,cols),0], points[range(0,cols),1])
		plt.plot(x,y,'r')
		plt.title('Linear Regression model')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.xlim([25,80])
		plt.ylim([20,100])
		plt.pause(0.1)
		plt.clf()
	return [b, m]

def run():
	points = genfromtxt('data.csv', delimiter=',')
	#hyoeroarameters
	learning_rate = 0.0001
	#y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 20
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points))

if __name__ == '__main__':
	run()
