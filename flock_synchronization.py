import time
import cv2
import numpy as np
import copy

ARENA_SIDE_LENGTH 		= 1000		# in pixels
NUMBER_OF_ROBOTS  		= 100
STEPS             		= 1500
MAX_SPEED         		= 10
MAX_SEPARATION			= 200		# d
FPS 					= 30
ACTIVATION_INCREASE 	= 0.5		# increase pr second
ACTIVATION_NEEDED   	= 1			# value
PERIOD  				= ACTIVATION_NEEDED / ACTIVATION_INCREASE	# T
PULSE_COUPLING_CONSTANT = 0.5		# e
RECORD 					= True


# Positions
x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

# Velocities
vx = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))
vy = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))

# Activation
activation = np.random.uniform(low=0, high=ACTIVATION_NEEDED, size=(NUMBER_OF_ROBOTS,))




def wrapdist(x1, x2):	# Distance points towards x1
	d = x1 - x2
	if abs(d) > abs(d + ARENA_SIDE_LENGTH):
		return d + ARENA_SIDE_LENGTH
	if abs(d) > abs(d - ARENA_SIDE_LENGTH):
		return d - ARENA_SIDE_LENGTH
	else:
		return d


# Make the environment toroidal 
def wrap(z):    
	return z % ARENA_SIDE_LENGTH


def findNeighbors(x1, y1, x, y, activation, activation_copy):
	seperation = np.array([0., 0.])
	a_i = 0
	for x_, y_, a_c_ in zip(x, y, activation_copy):		# Counts number of flashes seen by agent
		dist = np.linalg.norm([wrapdist(x1, x_), wrapdist(y1, y_)])
		if 0 < dist < MAX_SEPARATION and a_c_ == 0:
			a_i += 1
	return (1 / PERIOD) / FPS + PULSE_COUPLING_CONSTANT * a_i * activation



def animate():
	global x, y, vx, vy, a
	pixel_array = np.full((ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH, 3), (255,255,255) , dtype=np.uint8)
	x = np.array(list(map(wrap, x + vx)))
	y = np.array(list(map(wrap, y + vy)))

	#for x_, y_ in zip(x, y):
	for i in range(len(x)):
		if activation[i] > ACTIVATION_NEEDED:
			pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 20, (0,0,255), 10)
			activation[i] = 0
		else:
			pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 5, (0,0,0), 3)
			pass

	activation_copy = copy.deepcopy(activation)
	for i in range(len(x)):
		activation[i] += findNeighbors(x[i], y[i], x, y, activation[i], activation_copy)
		# print(activation[i])
		#activation[i] += ACTIVATION_INCREASE/FPS
		

	#cv2.imshow("idfk", pixel_array)
	#cv2.waitKey(int(1000/FPS))

	
	#print('Step ', i + 1, '/', STEPS, end='\r')
	
	return pixel_array

if RECORD == True:
	fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
	video=cv2.VideoWriter('output2.mp4',fourcc,FPS,(ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH))

for i in range(0,STEPS):
	
	if RECORD == True:
		video.write(animate())
	else: 
		animate()

if RECORD == True:
	video.release()