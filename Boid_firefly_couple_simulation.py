from operator import itemgetter
from matplotlib.markers import MarkerStyle
#from tkinter.tix import MAX
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import time
import cv2
import matplotlib.pyplot as plt
from numba import jit
import copy

plt.style.use('seaborn-pastel')

ARENA_SIDE_LENGTH 	= 1000		# in pixels
NUMBER_OF_ROBOTS  	= 100
STEPS             	= 10000
MAX_SPEED         	= 10
MAX_SEPARATION		= 200
CLUSTER_RANGE  		= MAX_SEPARATION * 0.5
FPS 				= 30

SAVE_FIG = False
DRAW_BOIDS			 	= False
SEPERATION_WEIGHT = 0.2
ALLIGNMENT_WEIGHT = 0.1
COHESION_WEIGHT = 1
ARROW_SCALE = 30

ACTIVATION_INCREASE 	= 0.5		# increase pr second
ACTIVATION_NEEDED   	= 1			# value
PERIOD  				= ACTIVATION_NEEDED / ACTIVATION_INCREASE	# T

RECORD 					= DRAW_BOIDS
graphic					= False or RECORD

BIN_SCALE = 0.2

# Positions
x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

# Velocities
vx = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))
vy = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))

TESTING_E 				= np.array([0.01, 0.05, 0.1, 0.15, 0.2])
TESTING_E_n 			= 50

# Activation
activation = np.random.uniform(low=0, high=ACTIVATION_NEEDED, size=(NUMBER_OF_ROBOTS,))

# For plotting
activation_level = []
xactivation_level = []
variance = []
time_to_synchronize = np.zeros((len(TESTING_E), TESTING_E_n))
xtime_to_synchronize = [[]]


# Set up the output (1024 x 768):
#fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
#ax = plt.axes(xlim=(0, ARENA_SIDE_LENGTH), ylim=(0, ARENA_SIDE_LENGTH))
#points, = ax.plot([], [], 'bo', lw=0, )
#arrow = ax.arrow([],[],[],[])

@jit(nopython=True)	
def findNeighbors(x1, y1, x, y, activation, activation_copy, couple):
	a_i = 0
	for x_, y_, a_c_ in zip(x, y, activation_copy):		# Counts number of flashes seen by agent
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION and a_c_ == 0:
			a_i += 1
	#print(PULSE_COUPLING_CONSTANT)
	return (1 / PERIOD) / FPS + couple * a_i * activation


@jit(nopython=True)	
def wrappoint(x1, x2):	# Wraps position of x2
	d = x1 - x2
	if abs(d) > abs(x1 - (x2 + ARENA_SIDE_LENGTH)):
		return x2 + ARENA_SIDE_LENGTH
	if abs(d) > abs(x1 - (x2 - ARENA_SIDE_LENGTH)):
		return x2 - ARENA_SIDE_LENGTH
	else:
		return x2

@jit(nopython=True)	
def wrapdist(x1, x2):	# Distance points towards x1
	d = x1 - x2
	if abs(d) > abs(x1 - (x2 + ARENA_SIDE_LENGTH)):
		return x1 - (x2 + ARENA_SIDE_LENGTH)
	if abs(d) > abs(x1 - (x2 - ARENA_SIDE_LENGTH)):
		return x1 - (x2 - ARENA_SIDE_LENGTH)
	else:
		return d

	#min(x1 - x2, x1 - (x2 + ARENA_SIDE_LENGTH), x1 - (x2 - ARENA_SIDE_LENGTH))

@jit(nopython=True)	
def getSeperation(x1, y1, x, y):
	seperation = np.array([0., 0.])
	for x_, y_ in zip(x, y):
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION:
			direction =  np.array([	wrapdist(x1, x_), 
							wrapdist(y1, y_)])
			direction = direction / np.linalg.norm(direction)
			weight = ((MAX_SEPARATION/dist) ) - 1
			seperation -= direction * weight
			#cv2.arrowedLine(pixel_array, (int(x1), int(y1)), (int(x1)-int(wrapdist(x1, x_)), int(y1)-int(wrapdist(y1, y_))), (255,0,0), 2)
			#print(seperation)
	return seperation
	
@jit(nopython=True)	
def getAllignment(x1, y1, x, y, vx, vy):
	heading = np.array([0., 0.])
	for x_, vx_, y_, vy_ in zip(x, vx, y, vy):
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION:
			heading += np.array([vx_, vy_])
	if np.any(heading != 0):
		return heading / np.linalg.norm(heading)
	else:
		return heading

@jit(nopython=True)
def getCohesion(x1, y1, x, y):
	center = np.array([0., 0.])
	i = 0
	for x_, y_ in zip(x, y):
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION:
			center += np.array([wrappoint(x1, x_), wrappoint(y1, y_)])
			i += 1
	if i > 0:
		center = center / i
		direction = (center - np.array([x1, y1])) 
		direction = direction / np.linalg.norm(direction)
	else:
		direction = np.array([0., 0.])
	return direction 

@jit(nopython=True)	
def getClosestNeighbor(x1, y1, x, y):
	closest = 1000
	for x_, y_ in zip(x, y):
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < closest:
			closest = dist
	return closest

@jit(nopython=True)	
def countNeighbor(x1, y1, x, y):
	count = 0
	for x_, y_ in zip(x, y):
		dist = np.linalg.norm(np.array([wrapdist(x1, x_), wrapdist(y1, y_)]))
		if 0 < dist < MAX_SEPARATION:
			count +=1
	return count

# Make the environment toroidal 
@jit(nopython=True)	
def wrap(z):    
	return z % ARENA_SIDE_LENGTH


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

#def init():
#	points.set_data([], [])
#	arrow.set_data([], [], [], [])
#	return points, arrow
def animate(couple):
	global x, y, vx, vy
	if DRAW_BOIDS:
		pixel_array = np.full((ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH, 3), (255,255,255) , dtype=np.uint8)
	x = np.array(list(map(wrap, x + vx)))
	y = np.array(list(map(wrap, y + vy)))

	#for x_, y_ in zip(x, y):
	for i in range(len(x)):
		if np.linalg.norm([vx[i], vy[i]]) > MAX_SPEED:
			#print("speed: ", np.linalg.norm([vx[i], vy[i]]))
			vx[i] = (vx[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
			vy[i] = (vy[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
			#print("speed after cap: ", np.linalg.norm([vx[i], vy[i]]))
		#pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), MAX_SEPARATION, (200,200,255), 3)
		if DRAW_BOIDS:
			pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 5, (255,0,0), 3)

		distvec = getSeperation(x[i], y[i], x, y)
		vx[i] -= distvec[0] * SEPERATION_WEIGHT
		vy[i] -= distvec[1] * SEPERATION_WEIGHT
		if abs(distvec[0]) > 0 or abs(distvec[1]) > 0:
			if DRAW_BOIDS:
				pixel_array = cv2.arrowedLine(pixel_array, (int(x[i]), int(y[i])), (int(x[i])-int(distvec[0]*SEPERATION_WEIGHT*ARROW_SCALE), int(y[i])-int(distvec[1]*SEPERATION_WEIGHT*ARROW_SCALE)),
										(255,0,0), 2)
			pass

		allignmentvec = getAllignment(x[i], y[i], x, y, vx, vy)
		currentSpeed = np.linalg.norm([vx[i], vy[i]])
		#print(currentSpeed)
		vx[i] = vx[i] + allignmentvec[0] * ALLIGNMENT_WEIGHT
		vy[i] = vy[i] + allignmentvec[1] * ALLIGNMENT_WEIGHT
		if abs(allignmentvec[0]) > 0 or abs(allignmentvec[1]) > 0:
			if DRAW_BOIDS:
				pixel_array = cv2.arrowedLine(pixel_array, (int(x[i]), int(y[i])), (int(x[i])+int(allignmentvec[0]*ALLIGNMENT_WEIGHT*ARROW_SCALE), int(y[i])+int(allignmentvec[1]*ALLIGNMENT_WEIGHT*ARROW_SCALE)),
										(0,255,0), 2)
			pass
		vx_ = (vx[i] / np.linalg.norm([vx[i], vy[i]])) * currentSpeed
		vy[i] = (vy[i] / np.linalg.norm([vx[i], vy[i]])) * currentSpeed
		vx[i] = vx_

		centervec = getCohesion(x[i], y[i], x, y)
		vx[i] += centervec[0] * COHESION_WEIGHT
		vy[i] += centervec[1] * COHESION_WEIGHT
		if abs(centervec[0]) > 0 or abs(centervec[1]) > 0:
			if DRAW_BOIDS:
				pixel_array = cv2.arrowedLine(pixel_array, (int(x[i]), int(y[i])), (int(x[i])+int(centervec[0]*COHESION_WEIGHT*ARROW_SCALE), int(y[i])+int(centervec[1]*COHESION_WEIGHT*ARROW_SCALE)),
										(0,0,255), 2)
			pass
		#print("One down")
		#points.set_data(x, y)
		#points, = ax.plot(x, y, 'bo', lw=0, )
		#ax.plot(x, y, 'bo', lw=0, )

		vx[i] = (vx[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
		vy[i] = (vy[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED

	# ------------------ FROM SYNC --------------------------------

	#for x_, y_ in zip(x, y):
	for i in range(len(x)):
		if activation[i] > ACTIVATION_NEEDED:
			if graphic:
				pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 20, (0,0,255), 10)
			activation[i] = 0
		#else:
			#if graphic:
			#	pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 5, (0,0,0), 3)
			#pass

	activation_level.append(np.array(activation))
	variance.append(np.var(activation))

	activation_copy = copy.deepcopy(activation)
	for i in range(len(x)):
		activation[i] += findNeighbors(x[i], y[i], x, y, activation[i], activation_copy, couple)
		# print(activation[i])
		#activation[i] += ACTIVATION_INCREASE/FPS
	# ------------------ END SYNC --------------------------------

	if DRAW_BOIDS:
		cv2.imshow("boids", pixel_array)
		cv2.waitKey(int(1000/FPS))

	if DRAW_BOIDS:
		return pixel_array
	return



for j, coupling in enumerate(TESTING_E):
	
	PULSE_COUPLING_CONSTANT = coupling
	print("Testing for ", PULSE_COUPLING_CONSTANT)
	for k in range(TESTING_E_n):

		print("Test #", k)
		reset()

		if RECORD and DRAW_BOIDS:
			fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
			video=cv2.VideoWriter('sync/output2_test.mp4',fourcc,FPS,(ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH))

		for i in range(0,STEPS):
			if RECORD and DRAW_BOIDS:
				video.write(animate(coupling))
			else: 
				animate(coupling)

			xactivation_level.append([i] * len(activation_level[0]))
			#print(activation_level)
			#time.sleep(1)
			#print(variance[i])
			if variance[i] < 0.001:
				print(i)
				time_to_synchronize[j][k] = i
				break
			
			if i == STEPS-1:
				time_to_synchronize[j][k] = i
				print(i)

		

		if RECORD == True:
			video.release()
			quit()

means = np.mean(time_to_synchronize, 1)
stds = np.std(time_to_synchronize, 1)
fig, ax = plt.subplots( figsize=(13, 6))
ax.bar(np.arange(len(TESTING_E)), means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Timesteps before synchronization')
ax.set_xticks(np.arange(len(TESTING_E)))
ax.set_xticklabels(TESTING_E)
ax.set_title("Period: " + str(PERIOD*FPS) + ",   No. robots: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION) + ",   Trials: " + str(TESTING_E_n))
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('sync/BOID_bar_plot_with_error_bars.png')
plt.show()