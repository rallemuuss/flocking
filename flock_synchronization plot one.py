import time
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from numba import jit

plt.style.use('seaborn-pastel')

ARENA_SIDE_LENGTH 		= 1000		# in pixels
NUMBER_OF_ROBOTS  		= 100
STEPS             		= 10000
MAX_SPEED         		= 10
MAX_SEPARATION			= 200		# d
FPS 					= 30
ACTIVATION_INCREASE 	= 0.5		# increase pr second
ACTIVATION_NEEDED   	= 1			# value
PERIOD  				= ACTIVATION_NEEDED / ACTIVATION_INCREASE	# T
PULSE_COUPLING_CONSTANT = 0.1		# e
RECORD 					= False
graphic					= False 	or RECORD
# Positions
x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

# Velocities
vx = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))
vy = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))

# Activation
activation = np.random.uniform(low=0, high=ACTIVATION_NEEDED, size=(NUMBER_OF_ROBOTS,))

activation_level = []
xactivation_level = []
variance = []
time_to_synchronize = []
xtime_to_synchronize = [[]]


@jit(nopython=True)	
def wrapdist(x1, x2):	# Distance points towards x1
	d = x1 - x2
	if abs(d) > abs(x1 - (x2 + ARENA_SIDE_LENGTH)):
		return x1 - (x2 + ARENA_SIDE_LENGTH)
	if abs(d) > abs(x1 - (x2 - ARENA_SIDE_LENGTH)):
		return x1 - (x2 - ARENA_SIDE_LENGTH)
	else:
		return d


# Make the environment toroidal 
@jit(nopython=True)	
def wrap(z):    
	return z % ARENA_SIDE_LENGTH


@jit(nopython=True)	
def findNeighbors(x1, y1, x, y, activation, activation_copy):
	a_i = 0
	for x_, y_, a_c_ in zip(x, y, activation_copy):		# Counts number of flashes seen by agent
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION and a_c_ == 0:
			a_i += 1
	#print(PULSE_COUPLING_CONSTANT)
	return (1 / PERIOD) / FPS + PULSE_COUPLING_CONSTANT * a_i * activation


def reset():
	global x, y, vx, vy, activation, activation_level, xactivation_level, variance
	# Positions
	x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
	y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

	# Velocities
	vx = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))
	vy = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))

	# Activation
	activation = np.random.uniform(low=0, high=ACTIVATION_NEEDED, size=(NUMBER_OF_ROBOTS,))

	activation_level = []
	xactivation_level = []
	variance = []

def animate():
	global x, y, vx, vy, activation
	if graphic:
		pixel_array = np.full((ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH, 3), (255,255,255) , dtype=np.uint8)
	x = np.array(list(map(wrap, x + vx)))
	y = np.array(list(map(wrap, y + vy)))

	#for x_, y_ in zip(x, y):
	for i in range(len(x)):
		if activation[i] > ACTIVATION_NEEDED:
			if graphic:
				pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 20, (0,0,255), 10)
			activation[i] = 0
		else:
			if graphic:
				pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 5, (0,0,0), 3)
			pass

	activation_level.append(activation[0])
	variance.append(np.var(activation))

	activation_copy = copy.deepcopy(activation)
	for i in range(len(x)):
		activation[i] += findNeighbors(x[i], y[i], x, y, activation[i], activation_copy)
		# print(activation[i])
		#activation[i] += ACTIVATION_INCREASE/FPS
		
	if graphic:
		cv2.imshow("idfk", pixel_array)
		cv2.waitKey(int(1000/FPS))
	
	
	#print('Step ', i + 1, '/', STEPS, end='\r')
	if graphic:
		return pixel_array



if RECORD == True:
	fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
	video=cv2.VideoWriter('output2.mp4',fourcc,FPS,(ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH))
reset()

for i in range(0,STEPS):
	
	if RECORD == True:
		video.write(animate())
	else: 
		animate()
	#xactivation_level.append([i] * len(activation_level[0]))
	#print(activation_level)
	#time.sleep(1)
	#print(variance[i])

	if i == 400:#not i%100 and i:
		plt.figure(1, figsize=(13, 6))
		plt.plot(np.arange(len(activation_level)), activation_level, 1, marker = ",")
		#plt.plot(xactivation_level, activation_level)
		plt.xlabel("Time steps")
		plt.ylabel("Activation level")
		plt.title("Period: " + str(PERIOD*FPS) + ",   Pulse coupling const: " + str(PULSE_COUPLING_CONSTANT) + ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
		plt.savefig("sync/Single_Activation_Period-" + str(PERIOD*FPS) + "_Pulse-coupling-const-" + str(PULSE_COUPLING_CONSTANT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")

		plt.figure(2, figsize=(13, 6))
		plt.scatter(np.arange(len(variance)), variance, 1, marker = ",")
		#plt.plot(xactivation_level, activation_level)
		plt.xlabel("Time steps")
		plt.ylabel("Variance level")
		plt.title("Period: " + str(PERIOD*FPS) + ",   Pulse coupling const: " + str(PULSE_COUPLING_CONSTANT) + ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
		plt.savefig("sync/Single_Variance-Period-" + str(PERIOD*FPS) + "_Pulse-coupling-const-" + str(PULSE_COUPLING_CONSTANT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")
		plt.show()

		if RECORD == True:
			video.release()
		break



#ACTIVATION_INCREASE 		# increase pr second
#ACTIVATION_NEEDED  		# value
#PERIOD  					# T
#PULSE_COUPLING_CONSTANT 	# e


if RECORD == True:
	video.release()