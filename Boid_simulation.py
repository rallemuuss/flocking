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

plt.style.use('seaborn-pastel')

ARENA_SIDE_LENGTH 	= 1000		# in pixels
NUMBER_OF_ROBOTS  	= 20
STEPS             	= 1500
MAX_SPEED         	= 10
MAX_SEPARATION		= 200
CLUSTER_RANGE  		= MAX_SEPARATION * 0.5
FPS 				= 30
RECORD = False
SAVE_FIG = False
DRAW_BOIDS = True
SEPERATION_WEIGHT = 1
ALLIGNMENT_WEIGHT = 1
COHESION_WEIGHT = 1
ARROW_SCALE = 30

BIN_SCALE = 0.2

# Positions
x = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
y = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

# Velocities
vx = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))
vy = np.random.uniform(low=-MAX_SPEED/2, high=MAX_SPEED/2, size=(NUMBER_OF_ROBOTS,))
vx[0] = 0.1
vy[0] = 0.001

# For plotting
directions = []
xdirections = []
shortest_neighbor_dist = []
shortest_neighbor_mean = []
neighbor_count = []

# Set up the output (1024 x 768):
#fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
#ax = plt.axes(xlim=(0, ARENA_SIDE_LENGTH), ylim=(0, ARENA_SIDE_LENGTH))
#points, = ax.plot([], [], 'bo', lw=0, )
#arrow = ax.arrow([],[],[],[])

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

#def init():
#	points.set_data([], [])
#	arrow.set_data([], [], [], [])
#	return points, arrow
def animate():
	global x, y, vx, vy, directions
	if DRAW_BOIDS:
		pixel_array = np.full((ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH, 3), (255,255,255) , dtype=np.uint8)
	bin_array = np.full((int(ARENA_SIDE_LENGTH*3*BIN_SCALE), int(ARENA_SIDE_LENGTH*3*BIN_SCALE), 1), 0 , dtype=np.uint8)
	x = np.array(list(map(wrap, x + vx)))
	y = np.array(list(map(wrap, y + vy)))

	shortest_neighbor_dist.append(np.zeros(NUMBER_OF_ROBOTS))
	neighbor_count.append(np.zeros(NUMBER_OF_ROBOTS))
	#for x_, y_ in zip(x, y):
	for i in range(len(x)):
		#if np.linalg.norm([vx[i], vy[i]]) > MAX_SPEED:
			#print("speed: ", np.linalg.norm([vx[i], vy[i]]))
		#vx[i] = (vx[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
		#vy[i] = (vy[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
			#print("speed after cap: ", np.linalg.norm([vx[i], vy[i]]))
		#pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), MAX_SEPARATION, (200,200,255), 3)
		if DRAW_BOIDS:
			pixel_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), 5, (255,0,0), 3)
		#bin_array = cv2.circle(pixel_array, (int(x[i]), int(y[i])), MAX_SEPARATION, 255, -1)
		#bin_array = cv2.circle(pixel_array, (int(x[i]+ARENA_SIDE_LENGTH), int(y[i]+ARENA_SIDE_LENGTH)), MAX_SEPARATION, 255, -1)
		for j in [0, 1, 2]:
			for k in [0, 1, 2]:
				bin_array = cv2.circle(bin_array, (int((x[i] + ARENA_SIDE_LENGTH*j)*BIN_SCALE), int((y[i] + ARENA_SIDE_LENGTH*k)*BIN_SCALE)), int(CLUSTER_RANGE*BIN_SCALE), 255, -1)
	#	print(getSeperation(x_, y_, x, y))
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
		


		shortest_neighbor_dist[-1][i] = getClosestNeighbor(x[i], y[i], x, y)
		neighbor_count[-1][i] = countNeighbor(x[i], y[i], x, y)


	shortest_neighbor_mean.append(np.mean(shortest_neighbor_dist[-1]))

	if DRAW_BOIDS:
		cv2.imshow("boids", pixel_array)
		cv2.waitKey(int(1000/FPS))
	
	(numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(bin_array)
	clusters = 0
	for center in centroids[1:]:
		#print(center)
		bin_array = cv2.circle(bin_array, (int(center[0]), int(center[1])), 3, 177, -1)
		if ARENA_SIDE_LENGTH*BIN_SCALE < center[0] < 2*ARENA_SIDE_LENGTH*BIN_SCALE:
			if ARENA_SIDE_LENGTH*BIN_SCALE < center[1] < 2*ARENA_SIDE_LENGTH*BIN_SCALE:
				clusters += 1
		#print(center )
	print(clusters)
	directions.append(np.arctan2(vy,vx))
	len_dir = len(directions)
	#print(len_dir)
	if len_dir > 1:		# Fixing wrapping jump in plot by setting nan if change too big.
		for i in range(len(directions[0])):
			if abs(directions[len_dir-1][i] - directions[len_dir-2][i]) > np.pi :
				directions[len_dir-2][i] = np.nan
	

	cv2.imshow("drenge", bin_array[int(ARENA_SIDE_LENGTH*BIN_SCALE):int(ARENA_SIDE_LENGTH*2*BIN_SCALE) , int(ARENA_SIDE_LENGTH*BIN_SCALE):int(ARENA_SIDE_LENGTH*2*BIN_SCALE)])
	cv2.waitKey(int(1000/FPS))
	#print(shortest_neighbor_dist)

		
	#directions = np.arctan2(vy,vx)
	#print(directions)
	#print(directions)
	#print(np.shape(directions))
	
	#print('Step ', i + 1, '/', STEPS, end='\r')
	if DRAW_BOIDS:
		return pixel_array
	return

if RECORD and DRAW_BOIDS:
	fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
	video=cv2.VideoWriter('output2.mp4',fourcc,FPS,(ARENA_SIDE_LENGTH, ARENA_SIDE_LENGTH))

for i in range(0,STEPS):
	
	if RECORD and DRAW_BOIDS:
		video.write(animate())
	else: 
		animate()
	xdirections.append([i] * len(directions[0]))

	if i == 400:#not i%100 and i:
		plt.figure(1, figsize=(13, 6))
		#plt.scatter(xdirections, directions, marker = "_")
		plt.plot(xdirections, directions)
		plt.xlabel("Time steps")
		plt.ylabel("Boid angle")
		plt.title("Separation: " + str(SEPERATION_WEIGHT) + ",   Alignment: "+ str(ALLIGNMENT_WEIGHT) + ",   Cohesion: " + str(COHESION_WEIGHT) 
			+ ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
		if SAVE_FIG : plt.savefig("Angle__Separation-" + str(SEPERATION_WEIGHT) + "_Alignment-"+ str(ALLIGNMENT_WEIGHT) + "_Cohesion-" + str(COHESION_WEIGHT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")

		plt.figure(2, figsize=(13, 6))
		plt.plot(xdirections, shortest_neighbor_dist)
		plt.xlabel("Time steps")
		plt.ylabel("Nearest neighbor distance")
		plt.title("Separation: " + str(SEPERATION_WEIGHT) + ",   Alignment: "+ str(ALLIGNMENT_WEIGHT) + ",   Cohesion: " + str(COHESION_WEIGHT) 
			+ ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
		plt.plot(np.arange(len(shortest_neighbor_mean)), shortest_neighbor_mean, linewidth=5.0, linestyle=':', color='#4b0082', dash_capstyle='round')
		if SAVE_FIG : plt.savefig("Distance__Separation-" + str(SEPERATION_WEIGHT) + "_Alignment-"+ str(ALLIGNMENT_WEIGHT) + "_Cohesion-" + str(COHESION_WEIGHT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")

		plt.figure(3, figsize=(13, 6))
		plt.plot(xdirections, neighbor_count)
		plt.xlabel("Time steps")
		plt.ylabel("Visible BOIDS")
		plt.title("Separation: " + str(SEPERATION_WEIGHT) + ",   Alignment: "+ str(ALLIGNMENT_WEIGHT) + ",   Cohesion: " + str(COHESION_WEIGHT) 
			+ ",   No BOIDS: " + str(NUMBER_OF_ROBOTS) + ",   Neighbourhood distance: " + str(MAX_SEPARATION))
		#plt.plot(np.arange(len(shortest_neighbor_mean)), shortest_neighbor_mean, linewidth=5.0, linestyle=':', color='#4b0082', dash_capstyle='round')
		if SAVE_FIG : plt.savefig("Neighbors__Separation-" + str(SEPERATION_WEIGHT) + "_Alignment-"+ str(ALLIGNMENT_WEIGHT) + "_Cohesion-" + str(COHESION_WEIGHT) + "_No-BOIDS-" + str(NUMBER_OF_ROBOTS) + "_Neighbourhood-distance-" + str(MAX_SEPARATION) + ".png")
		plt.show()

		quit()

if RECORD == True:
	video.release()


'''
anim = FuncAnimation(fig, animate, init_func=init,
							frames=STEPS, interval=1, blit=True)


videowriter = animation.FFMpegWriter(fps=60)
anim.save("output.mp4", writer=videowriter)
'''