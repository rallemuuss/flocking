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
graphic					= False or RECORD
TESTING_ROBOTS 			= np.array([10, 20, 40, 60, 80, 100, 150, 200])
TESTING_ROBOTS_n 		= 50

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
time_to_synchronize = np.zeros((len(TESTING_ROBOTS), TESTING_ROBOTS_n))
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
def findNeighbors(x1, y1, x, y, activation, activation_copy, couple):
	seperation = np.array([0., 0.])
	a_i = 0
	for x_, y_, a_c_ in zip(x, y, activation_copy):		# Counts number of flashes seen by agent
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION and a_c_ == 0:
			a_i += 1
	#print(PULSE_COUPLING_CONSTANT)
	return (1 / PERIOD) / FPS + couple * a_i * activation


def reset(robots):
	global x, y, vx, vy, activation, activation_level, xactivation_level, variance
	# Positions
	x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(robots,))
	y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(robots,))

	# Velocities
	vx = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(robots,))
	vy = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(robots,))

	# Activation
	activation = np.random.uniform(low=0, high=ACTIVATION_NEEDED, size=(robots,))

	activation_level = []
	xactivation_level = []
	variance = []

def animate(couple):
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

	activation_level.append(np.array(activation))
	variance.append(np.var(activation))

	activation_copy = copy.deepcopy(activation)
	for i in range(len(x)):
		activation[i] += findNeighbors(x[i], y[i], x, y, activation[i], activation_copy, couple)
		# print(activation[i])
		#activation[i] += ACTIVATION_INCREASE/FPS
		
	if graphic:
		cv2.imshow("idfk", pixel_array)
		cv2.waitKey(int(1000/FPS))
	
	
	#print('Step ', i + 1, '/', STEPS, end='\r')
	if graphic:
		return pixel_array


for j, robots in enumerate(TESTING_ROBOTS):
	
	NUMBER_OF_ROBOTS = robots
	print("Testing for ", NUMBER_OF_ROBOTS)

	for k in range(TESTING_ROBOTS_n):

		print("Test #", k)
		if RECORD == True:
			fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
			video=cv2.VideoWriter('output2.mp4',fourcc,FPS,(ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH))
		reset(robots)

		for i in range(0,STEPS):
			
			if RECORD == True:
				video.write(animate(PULSE_COUPLING_CONSTANT))
			else: 
				animate(PULSE_COUPLING_CONSTANT)
			xactivation_level.append([i] * len(activation_level[0]))
			#print(activation_level)
			#time.sleep(1)
			#print(variance[i])
			if variance[i] < 0.001:
				print(i)
				time_to_synchronize[j][k] = i
				break
			

			if i == STEPS-100:
				time_to_synchronize[j][k] = i
				print(i)
				input("Reaching " + str(STEPS-1) + ", Press enter..")

			if i == STEPS-1:
				time_to_synchronize[j][k] = i
				print(i)
				input("We've reached " + str(STEPS-1) + ", Press enter..")

			if False: # i == 400:#not i%100 and i:
				plt.figure(1, figsize=(13, 6))
				plt.scatter(xactivation_level, activation_level, 1, marker = ",")
				#plt.plot(xactivation_level, activation_level)
				plt.xlabel("Time steps")
				plt.ylabel("Activation level")
				plt.title("Period: " + str(PERIOD*FPS) + ",   Pulse coupling const: " + str(PULSE_COUPLING_CONSTANT) + ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
				plt.savefig("sync/Activation_Period-" + str(PERIOD*FPS) + "_Pulse-coupling-const-" + str(PULSE_COUPLING_CONSTANT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")

				plt.figure(2, figsize=(13, 6))
				plt.scatter(np.arange(len(variance)), variance, 1, marker = ",")
				#plt.plot(xactivation_level, activation_level)
				plt.xlabel("Time steps")
				plt.ylabel("Variance level")
				plt.title("Period: " + str(PERIOD*FPS) + ",   Pulse coupling const: " + str(PULSE_COUPLING_CONSTANT) + ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
				plt.savefig("sync/Variance-Period-" + str(PERIOD*FPS) + "_Pulse-coupling-const-" + str(PULSE_COUPLING_CONSTANT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")
				plt.show()

				if RECORD == True:
					video.release()
				break


means = np.mean(time_to_synchronize, 1)
stds = np.std(time_to_synchronize, 1)
fig, ax = plt.subplots( figsize=(13, 6))
ax.bar(np.arange(len(TESTING_ROBOTS)), means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Timesteps before synchronization')
ax.set_xticks(np.arange(len(TESTING_ROBOTS)))
ax.set_xticklabels(TESTING_ROBOTS)
ax.set_title("Period: " + str(PERIOD*FPS) + ",   No. robots: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION) + ",   Trials: " + str(TESTING_ROBOTS_n))
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('sync/robotsn_bar_plot_with_error_bars.png')
plt.show()


#ACTIVATION_INCREASE 		# increase pr second
#ACTIVATION_NEEDED  		# value
#PERIOD  					# T
#PULSE_COUPLING_CONSTANT 	# e


if RECORD == True:
	video.release()