from numpy import genfromtxt
import numpy as np

original_filename = "train_triples.txt"
trunc_filename = "truncated_data.txt"

def trunc_file(line_no = 200000):
	f = open("train_triplets.txt", 'r')
	f2 = open("truncated_data.txt", 'w')

	for i in xrange(line_no):
		line = f.readline()
		f2.write(line)

	f.close()
	f2.close()



def getUsersSongs():
	print "Getting users and songs...",
	f = open(trunc_filename, 'r')
	song_list = open("unique_songs.txt",'w')

	users = []
	songs = []
	for line in f:
		words = line.split('\t')
		users.append(words[0])
		songs.append(words[1])

	users = list(set(users))
	songs = list(set(songs))

	f.close()

	for i in xrange(len(songs)):
		song_list.write(songs[i]+"\n")

	print "Done"
	return users, songs


def similarity(u,v):
	# print "Calculating sim between " + str(u) + " and " + str(v),
	numerator = float(np.inner(u,v))
	norm_u = np.linalg.norm(u)
	norm_v = np.linalg.norm(v)
	denominator = norm_u * norm_v
	# print numerator, denominator
	# print " = " + str(numerator/denominator)
	return numerator/denominator

def score(user_index,song_index,data):
	u = data[user_index]
	u_avg = np.mean(u)
	numerator = 0
	denominator = 0
	# print 'similarity = '
	for u1 in data:
		u1_avg = np.mean(u1)
		sim = similarity(u,u1)
		# print sim,
		numerator += sim * (u[song_index]-u1_avg)
		denominator += abs(sim)
	return u_avg + (numerator/denominator)	

def prediction(user_index,data,k=16):
	score_list = []
	print "Constructing score list...",
	for i in xrange(data.shape[1]):
		s = score(user_index, i, data)
		score_list.append((s, i))
	score_list.sort()
	print "Done"
	recommendations = score_list[data.shape[1]-k:data.shape[1]]
	rec_songs = [y for (x,y) in recommendations]
	# print score_list
	return rec_songs

def constructMatrix(users, songs):
	dictionary_user = open("dictionary_user.txt",'w')
	dictionary_song = open("dictionary_song.txt",'w')
	matrix = open("matrix.txt",'w')
	dict_users = dict()
	dict_songs = dict()
	#data_matrix = [[0]*len(songs)]*len(users)
	data_matrix = [[0]*len(songs) for _ in xrange(len(users))]

	for i in xrange(len(users)):
		dict_users[users[i]] = i
		dictionary_user.write(users[i]+'\t'+str(i)+'\n')

	for i in xrange(len(songs)):
	    dict_songs[songs[i]] = i
	    dictionary_song.write(songs[i]+'\t'+str(i)+'\n')

	print "Constructing count matrix...",
	f2 = open('truncated_data.txt', 'r')

	for line in f2:
		#print line
		data = line.split('\t')
		user_no = data[0]
		song_no = data[1]
		song_count = int(data[2])

		user_index = dict_users[user_no]
		song_index = dict_songs[song_no]

		data_matrix[ user_index ][ song_index ] = song_count
	#print data_matrix
	# for u in data_matrix:
	# 	u = np.asarray(u)
	# 	for i in xrange(u.shape[0]):
	# 		matrix.write(str(u[i])
	# 	matrix.write('\n')	

	print "Done"
	#print data_matrix
	data_matrix = np.asarray(data_matrix)


	np.savetxt("matrix.csv",data_matrix,fmt='%.4f',delimiter=',', newline='\n')
	#print data_matrix
	f2.close()
	return data_matrix

def normalizeMatrix(data_matrix):

	num_rows = data_matrix.shape[0]
	num__cols = data_matrix.shape[1]

	data_matrix_normalized = [[0]*num__cols] * num_rows

	#normalizing the data
	for i in xrange(num_rows):
		data_matrix_normalized[i] = data_matrix[i] / float(np.amax(data_matrix[i]))

	return np.asarray(data_matrix_normalized)

if __name__ == '__main__':
	trunc_file(3000)
	users, songs = getUsersSongs()
	data = constructMatrix(users, songs)
	data_normal = normalizeMatrix(data)
	print prediction(len(users)-1,data_normal)
	print len(users) 
	print len(songs)

	#user_music = genfromtxt('matrix.csv', delimiter=',')
	#print user_music

