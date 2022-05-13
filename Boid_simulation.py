from operator import itemgetter
from matplotlib.markers import MarkerStyle
#from tkinter.tix import MAX
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import time
import cv2
import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')

ARENA_SIDE_LENGTH 	= 1000		# in pixels
NUMBER_OF_ROBOTS  	= 50
STEPS             	= 1500
MAX_SPEED         	= 10
MAX_SEPARATION		= 200
FPS 				= 30
RECORD = False
DRAW_BOIDS = True
SEPERATION_WEIGHT = 0.10
ALLIGNMENT_WEIGHT = 0.50
COHESION_WEIGHT = 0.1

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

# Set up the output (1024 x 768):
#fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
#ax = plt.axes(xlim=(0, ARENA_SIDE_LENGTH), ylim=(0, ARENA_SIDE_LENGTH))
#points, = ax.plot([], [], 'bo', lw=0, )
#arrow = ax.arrow([],[],[],[])




def wrapdist(x1, x2):	# Distance points towards x1
	d = x1 - x2
	if abs(d) > abs(d + ARENA_SIDE_LENGTH):
		return d + ARENA_SIDE_LENGTH
	if abs(d) > abs(d - ARENA_SIDE_LENGTH):
		return d - ARENA_SIDE_LENGTH
	else:
		return d

	#min(x1 - x2, x1 - (x2 + ARENA_SIDE_LENGTH), x1 - (x2 - ARENA_SIDE_LENGTH))

def getSeperation(x1, y1, x, y):
	seperation = np.array([0., 0.])
	for x_, y_ in zip(x, y):
		dist = np.linalg.norm([wrapdist(x1, x_), wrapdist(y1, y_)])
		if 0 < dist < MAX_SEPARATION:
			direction =  [	wrapdist(x1, x_), 
							wrapdist(y1, y_)]
			direction = direction / np.linalg.norm(direction)
			weight = ((MAX_SEPARATION/dist) ) - 1
			seperation -= direction * weight
			#cv2.arrowedLine(pixel_array, (int(x1), int(y1)), (int(x1)-int(wrapdist(x1, x_)), int(y1)-int(wrapdist(y1, y_))), (255,0,0), 2)
			#print(seperation)
	return seperation
	
def getAllignment(x1, y1, x, y, vx, vy):
	heading = np.array([0., 0.])
	for x_, vx_, y_, vy_ in zip(x, vx, y, vy):
		dist = np.linalg.norm([wrapdist(x1, x_), wrapdist(y1, y_)])
		if 0 < dist < MAX_SEPARATION:
			#weight = ((MAX_SEPARATION/dist) ) - 1
			heading += [vx_, vy_] 
	if any(heading != 0):
		return heading / np.linalg.norm(heading)
	else:
		return heading


def getCohesion(x1, y1, x, y):
	center = np.array([0., 0.])
	i = 0
	for x_, y_ in zip(x, y):
		dist = np.linalg.norm([wrapdist(x1, x_), wrapdist(y1, y_)])
		if 0 < dist < MAX_SEPARATION:
			center += np.array([x_, y_])
			i += 1
	if i > 0:
		center = center / i
		direction = center - [x1, y1]
	else:
		direction = np.array([0., 0.])
	return direction

# Make the environment toroidal 
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
	x = np.array(list(map(wrap, x + vx)))
	y = np.array(list(map(wrap, y + vy)))

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
	#	print(getSeperation(x_, y_, x, y))
		distvec = getSeperation(x[i], y[i], x, y)
		vx[i] -= distvec[0] * 0.1
		vy[i] -= distvec[1] * 0.1

		allignmentvec = getAllignment(x[i], y[i], x, y, vx, vy)
		currentSpeed = np.linalg.norm([vx[i], vy[i]])
		#print(currentSpeed)
		vx[i] = vx[i] + allignmentvec[0] * 0.5
		vy[i] = vy[i] + allignmentvec[1] * 0.5
		time.sleep(0.001)
		vx_ = (vx[i] / np.linalg.norm([vx[i], vy[i]])) * currentSpeed
		vy[i] = (vy[i] / np.linalg.norm([vx[i], vy[i]])) * currentSpeed
		vx[i] = vx_

		centervec = getCohesion(x[i], y[i], x, y)
		vx[i] += centervec[0] * 0.01
		vy[i] += centervec[1] * 0.01
		#print("One down")
		#points.set_data(x, y)
		#points, = ax.plot(x, y, 'bo', lw=0, )
		#ax.plot(x, y, 'bo', lw=0, )

		vx[i] = (vx[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
		vy[i] = (vy[i] / np.linalg.norm([vx[i], vy[i]])) * MAX_SPEED
		
		if abs(distvec[0]) > 0 or abs(distvec[1]) > 0:
			if DRAW_BOIDS:
				pixel_array = cv2.arrowedLine(pixel_array, (int(x[i]), int(y[i])), (int(x[i])-int(distvec[0]*10), int(y[i])-int(distvec[1]*10)),
										(0,100,0), 2)
			pass

	if DRAW_BOIDS:
		cv2.imshow("boids", pixel_array)
		cv2.waitKey(int(1000/FPS))
	directions.append(np.arctan2(vy,vx))
	len_dir = len(directions)
	print(len_dir)
	if len_dir > 1:		# Fixing wrapping jump in plot by setting nan if change too big.
		for i in range(len(directions[0])):
			if abs(directions[len_dir-1][i] - directions[len_dir-2][i]) > np.pi :
				directions[len_dir-2][i] = np.nan

		
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

	if not i%100 :
		plt.figure(figsize=(13, 6))
		#plt.scatter(xdirections, directions, marker = "_")
		plt.plot(xdirections, directions)
		plt.xlabel("time steps")
		plt.ylabel("Boid angle")
		plt.title("Separation: " + str(SEPERATION_WEIGHT) + ",   Alignment: "+ str(ALLIGNMENT_WEIGHT) + ",   Cohesion: " + str(COHESION_WEIGHT))
		plt.show()

if RECORD == True:
	video.release()


'''
anim = FuncAnimation(fig, animate, init_func=init,
							frames=STEPS, interval=1, blit=True)


videowriter = animation.FFMpegWriter(fps=60)
anim.save("output.mp4", writer=videowriter)
'''