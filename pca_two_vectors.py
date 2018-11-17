'''
File:			pca_two_vectors.py
Author:			Chloe Jackson	chloexxyzz@gmail.com
Version:		16-Nov-2018

Description:	This is the automation for PCA using the Karhunen-Loeve
				Transformation using two eigenvectors
'''


import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2


'''
	read_csv

	read a csv file into lists for future use
'''
def read_csv(file_name, records, header):
	with open(file_name) as data:
		header = data.readline()
		header = header.split(',')
		for i in range (0, len(header)):
			header[i] = header[i].strip()

		# save all the data 
		lines = data.readlines()
		for line in lines:
			values = line.split(',')
			# convert string to int
			for v in range(0,len(values)):
				values[v] = int(values[v].strip())
			records.append(values)
	return records, header


'''
	compute_covariance

	computes the covariance matrix for the data
'''
def compute_covariance(records):
	r = []
	for rec in records:
		r.append(rec[1:])
	r = np.asarray(r).T
	return np.cov(r)


'''
	compute_eigenv

	computes the eigenvalues and normalized eigenvectors for a covariance matrix
'''
def compute_eigenv(C):
	eigenvalues , eigenvectors = np.linalg.eig(C)
	return eigenvalues , eigenvectors


'''
	normalize_eigenvalues

	divides each eigenvalue by the sum of the absolute magnitude of each value
'''
def normalize_eigenvalues(eigenvalues):
	N = np.sum(eigenvalues)
	return eigenvalues / N	


'''
	plot_eigenvalues

	plots the normalized eigenvalues
'''
def plot_eigenvalues(eigenvalues):
	e = np.append(np.asarray([0]),eigenvalues)
	y = np.asarray([])
	x = range(0,13)
	S = 0
	for v in np.nditer(e):
		S += v
		y = np.append(y,np.asarray([S]))
	
	# plot x and y
	plt.figure(1)
	plt.plot(x,y,'bo',x,y,'k')
	plt.xlabel('Eigenvalue ID (decreasing magnitude)')
	plt.ylabel('Variance captured')
	plt.title('Eigenvalues And Their Captured Variance')
	plt.show()


'''
	two_best_vectors

	returns the two 'best' eigenvectors
'''
def two_best_vectors(eigenvalues,eigenvectors):
	vals = eigenvalues.tolist()
	vecs = eigenvectors.tolist()
	best_index = 0
	best_mag = 0
	second_index = 0
	second_mag = 0
	for i in range(0,len(vals)):
		if abs(vals[i]) > best_mag:
			second_mag = best_mag
			best_mag = abs(vals[i])
			second_index = best_index
			best_index = i
		elif abs(vals[i]) > second_mag:
			second_mag = abs(vals[i])
			second_index = i
	
	return vecs[best_index] , vecs[second_index]	


'''
	subtract_means

	subtracts the mean value for each attribute of all records for the 
	Karhunen-Loeve Transformation for PCA
'''
def subtract_means(data):
	mdata = []
	means = []
	N = len(data)
	# compute means
	for att in range(1,len(data[0])):
		mean = 0
		for d in data:
			mean += d[att]
		mean = mean / N
		means.append(mean)

	# create new records
	for d in data:
		mdata.append(d[1:])
	
	# subtract the means
	for mean in range(0,len(means)):
		for md in range(0,N):
			mdata[md][mean] = mdata[md][mean] - means[mean]	

	return means, mdata


'''
	project_with_two_vectors

	reduces the original data down to two dimensions using the two 'best'
	eigenvectors
'''
def project_with_two_vectors(data,vec1,vec2):
	new_points = []
	x = []
	y = []
	v1 = vec1.T
	v2 = vec2.T
	# multiply by the eigenvectors transpose
	for d in data:
		x.append(np.dot(d,v1))
		y.append(np.dot(d,v2))
		new_points.append([np.dot(d,v1),np.dot(d,v2)])

	# plot the data in PCA space
	plt.figure(1)
	plt.plot(x,y,'bo')
	plt.xlabel('AMOUNT OF PRINCIPAL COMPONENT # 1')
	plt.ylabel('AMOUNT OF PRINCIPAL COMPONENT # 2')
	plt.title('HW AG DATA PROJECTED ONTO 2D PCA SPACE USING THE KLT TRANSFORMATION')
	plt.show()

	return new_points

	
'''
	round_vectors

	round vectors to hundredth place
'''
def round_vectors(vectors):
	for v in range(0,len(vectors)):
		vec = vectors[v]
		for a in range(0,len(vec)):
			att = vec[a]
			att = (round(100*att))/100
			vec[a] = att
		vectors[v] = vec
	return vectors


'''
	reproject_centroids

	multiply the centroids by the projection vectors
'''
def reproject_centroids(centroids,vec1,vec2,means):
	cens = centroids.tolist()
	new_cens = []
	for i in range(0,len(cens)):
		cen = np.asarray([cens[i]])
		v1 = np.asarray([vec1.tolist()])
		v2 = np.asarray([vec2.tolist()])
		rep1 = np.dot(cen.T,v1)
		rep2 = np.dot(cen.T,v2)
		cen = np.add(rep1,rep2)
		cen = np.add(cen,means)
		cen = cen.tolist()
		new_cens.append(cen[0])

	new_cens = round_vectors(new_cens)
	for cen in new_cens:
		print('Centroid:')
		print(cen)	


'''
	usage

	print a usage message to the console
'''
def usage():
	print('USAGE: python3 pca_two_vectors.py <filename>.csv')
	quit()


'''
	main

	runs the program
'''
def main():
	records = []
	fields = []
	
	if len(sys.argv) != 2:
		usage()
	
	try:
		open(sys.argv[1], 'r')
	except IOError:
		print('invalid filename')
		usage()

	print('\nPart one: reading in csv ...')
	records , fields = read_csv(sys.argv[1],records,fields)
	
	print('\nPart two: Covariance Matrix  C =\n')
	C = compute_covariance(records)
	print(C)
	
	print('\nPart three: Proper vectors for C\n')
	eigenvalues , eigenvectors = compute_eigenv(C)
	eigenvectors = eigenvectors.T
	print('Values: \n' ,eigenvalues)
	print('Vectors:\n' ,eigenvectors)

	print('\nPart four: sorting eigenvalues:\n')
	new_e = -np.sort(-eigenvalues)
	print(new_e)

	print('\nPart five: normalize eigenvalues:\n')
	new_e = normalize_eigenvalues(new_e)
	print(new_e)
	plot_eigenvalues(new_e)

	print('\nPart six: show eigenvectors associated with the 2 largest eigenvalues\n')
	v1 , v2 = two_best_vectors(eigenvalues,eigenvectors)
	vs = round_vectors([v1,v2])
	v1 = vs[0]
	v2 = vs[1]
	print('Best:\n',v1)
	print('Second:\n',v2)
	# both are negative, turn positive
	v1 = -np.asarray(v1)
	v2 = -np.asarray(v2)

	print('\nPart seven: project data with the 2 largest eigenvalues\n')
	means, sub_means = subtract_means(records)
	new_points = project_with_two_vectors(sub_means,v1,v2)
		
	print('\nPart eight: perform k-Means clustering with Euclidean distance and k = 3 ...\n')
	centroids , labels = kmeans2(new_points, 3, iter=20,minit='random')

	print('\nPart nine: print centroids found from clustering in PCA space\n')
	print('Centroids:\n')
	print(centroids)

	print('\nPart ten: multiply centroids by the \'best\' eigenvectors\n')
	reproject_centroids(centroids,v1,v2,means)


# send it
main()
