from timeit import default_timer as timer
import descartes
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from shapely.geometry import box, Point, LineString
from shapely.ops import cascaded_union, linemerge

WIDTH = 50
HEIGHT = 50
MAX_SIZE = 5
MAX_BALLS = 8
DELAY = 1

NORTH = 0
NORTHEAST = 1
EAST = 2
SOUTHEAST = 3
SOUTH = 4
SOUTHWEST = 5
WEST = 6
NORTHWEST = 7

class Environment:

	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.balls = []
		self.sensors = []
		self.box = box(0,0,width,height)
		self.env_grid = [[0 for x in range(self.width)] for y in range(self.height)]
		self.steps = 0
	
	def add_ball(self, x, y, radius, direction, speed):
		self.balls.append(Ball(x,y,radius,direction, speed))

		
	def add_sensor(self, sensor):
			self.sensors.append(sensor)

	def onscreen(self, ball):
		b = Point(ball.x, ball.y).buffer(ball.radius)
		return b.intersects(self.box)		

	def valid_start(self,x,y,radius):
		new = Point(x,y).buffer(radius)
		for ball in self.balls:
			taken = Point(ball.x,ball.y).buffer(ball.radius)
			if new.intersects(taken):
				return False
		return True

	def get_valid_randoms(self):

		side = np.random.randint(0,3)
		if side == 0:  # bottom
			x = np.random.randint(0,self.width)
			y = 0
		elif side == 1:  # left
			x = 0
			y = np.random.randint(0,self.height)	
		elif side == 2:  # right
			x = self.width
			y = np.random.randint(0,self.height)	
		else:			# top
			x = np.random.randint(0,self.width)
			y = self.height
		r = np.random.randint(1, MAX_SIZE)
		if self.valid_start(x,y,r):
			return x,y,r
		else:
			return self.get_valid_randoms()

	def update(self):
		self.steps += 1
		offscreen = []
		self.env_grid = [[0 for x in range(self.width)] for y in range(self.height)]
		ball_polys = [Point(ball.x,ball.y).buffer(ball.radius) for ball in self.balls]
		ball_poly = cascaded_union(ball_polys)
		for x in range(self.width):
			for y in range(self.height):
				b = box(x,y,x+1,y+1)
				if ball_poly.intersects(b):
					self.env_grid[self.height-1-y][x] = 1
		for ball in self.balls:
			ball.move()
			if not self.onscreen(ball):
				offscreen.append(ball)
		self.balls = [ball for ball in self.balls if ball not in offscreen]
		if len(self.balls) < np.random.randint(1, MAX_BALLS):
			xpos, ypos, r = self.get_valid_randoms()
			direction = np.random.randint(0,7)
			speed = np.random.randint(1,10)
			self.add_ball(xpos,ypos,r,direction,speed)
	

	def print_grid(self):
		for y in range(self.height):
			print(self.env_grid[y])	
	
	def get_balls(self):
		return self.balls	
	
		
class Ball:

	def __init__(self, x, y, radius, direction, speed):
		self.x = x
		self.y = y
		self.radius = radius
		self.direction = direction
		self.speed = speed
		self.onscreen = False


	def move(self): 
		if self.direction == NORTH:
			self.y += self.speed
		elif self.direction == NORTHEAST:
			self.x += self.speed
			self.y += self.speed
		elif self.direction == EAST:
			self.x += self.speed	
		elif self.direction == SOUTHEAST:
			self.x += self.speed
			self.y -= self.speed
		elif self.direction == SOUTH:
			self.y -= self.speed
		elif self.direction == SOUTHWEST:
			self.y -= self.speed
			self.x -= self.speed
		elif self.direction == WEST:
			self.x -= self.speed
		elif self.direction == NORTHWEST:
			self.x -= self.speed
			self.y += self.speed


class Gui():

	def __init__(self, env):
		self.env = env
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False, 
					 xlim=(0,env.width), ylim=(0,env.height))
		self.patches = []
		for sensor in env.sensors:
			sensor.update_plots([plt.plot([], [], color=sensor.color, 
							 lw=1)[0] for _ in range(181)])	
			self.patches += sensor.plots
		self.patches += []
	
	def init(self):
		# init lines
		for sensor in self.env.sensors:
			for plot_line in sensor.plots:
				plot_line.set_data([], [])
		return self.patches


	def animate(self,i):
		patches = []
		
		# animate balls
		for ball in self.env.balls:
			patch = descartes.PolygonPatch(Point(ball.x,
									       ball.y).buffer(ball.radius))
			patches.append(self.ax.add_patch(patch))
		
		# animate ray trace lines
		polygons = []
		for ball in self.env.balls:
			polygons.append(Point(ball.x,ball.y).buffer(ball.radius))
		for sensor in self.env.sensors:
			patches += sensor.update_raytrace(polygons)
			sensor.update_sensor_grids()
			#sensor.print_visibility_grid()	
			#sensor.print_occupancy_grid()	
	
		# update environment
		self.env.update()
		self.patches = patches
		return self.patches
		

class Sensor:
	
	def __init__(self, x, y, direction, width, height):
		self.x = x
		self.y = y
		self.direction = direction
		self.width = width
		self.height = height
		self.fixed_rays = []
		self.rays = []
		self.plots = []
		self.visibility_grid = [[0 for x in range(width)] for y in range(height)]
		self.occupancy_grid = [[0 for x in range(width)] for y in range(height)]
		self.create_rays()
	
	def create_rays(self):
		x = self.x
		y = self.y
		width = self.width
		height = self.height
		if self.direction == NORTH: # facing up
			self.color = 'red'
			for i in range(90):
				x2 = width
				y2 = (width-x) * np.tan(np.radians(i))
				if y2 > height:
					y2 = height
					x2 = x + height / (np.tan(np.radians(i)))
				x3 = 0
				y3 = x * np.tan(np.radians(i))
				if y3 > height:
					y3 = height
					x3 = x - height / (np.tan(np.radians(i)))
				fixed_ray1 = Sensor.Ray(x,y,x2,y2,i)	
				fixed_ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				ray1 = Sensor.Ray(x,y,x2,y2,i)	
				ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				self.fixed_rays.append(fixed_ray1)
				self.fixed_rays.append(fixed_ray2)
				self.rays.append(ray1)
				self.rays.append(ray2)
			fixed_ray3 = Sensor.Ray(x,y,x,width,90) #ray for 90 degrees
			ray3 = Sensor.Ray(x,y,x,width,90)
			self.fixed_rays.append(fixed_ray3)
			self.rays.append(ray3)
		elif self.direction == NORTHEAST:
			self.color = 'purple'
			for i in range(90):
				if i != 45:
					x2 = y/np.tan(45+i) + x
					if x2 <= width:
						y2 = 0
						x3 = 0
						y3 = x2
					else:
						x2 = width
						y2 = np.tan(45+i)*(width-x2)
						x3 = y2
						y3 = height
					fixed_ray1 = Sensor.Ray(x,y,x2,y2,i)
					fixed_ray2 = Sensor.Ray(x,y,x3,y3,i+91)
					ray1 = Sensor.Ray(x,y,x2,y2,i)
					ray2 = Sensor.Ray(x,y,x3,y3,i+91)
				else:
					fixed_ray1 = Sensor.Ray(x,y,width,y,45)
					fixed_ray2 = Sensor.Ray(x,y,x,height,135)
					ray1 = Sensor.Ray(x,y,x2,y2,45)
					ray2 = Sensor.Ray(x,y,x3,y3,135)
				self.fixed_rays.append(fixed_ray1)
				self.fixed_rays.append(fixed_ray2)
				self.rays.append(ray1)
				self.rays.append(ray2)
			fixed_ray3 = Sensor.Ray(x,y,width,height,90) # ray for 90 degrees
			ray3 = Sensor.Ray(x,y,width,height,90)
			self.fixed_rays.append(fixed_ray3)
			self.rays.append(ray3)
		elif self.direction == EAST: # facing right
			self.color = 'blue'
			for i in range(90):
				x2 = y * np.tan(np.radians(i))
				y2 = 0
				if x2 > width:
					y2 = y - width / np.tan(np.radians(i))
					x2 = width
				x3 = (height-y) * np.tan(np.radians(i))
				y3 = height
				if x3 > width:
					y3 = y + width / np.tan(np.radians(i))
					x3 = width
				fixed_ray1 = Sensor.Ray(x,y,x2,y2,i)	
				fixed_ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				ray1 = Sensor.Ray(x,y,x2,y2,i)	
				ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				self.fixed_rays.append(fixed_ray1)
				self.fixed_rays.append(fixed_ray2)
				self.rays.append(ray1)
				self.rays.append(ray2)
			fixed_ray3 = Sensor.Ray(x,y,width,y,90) #ray for 90 degrees
			ray3 = Sensor.Ray(x,y,width,y,90)
			self.fixed_rays.append(fixed_ray3)
			self.rays.append(ray3)
		elif self.direction == SOUTH: # facing down
			self.color = 'green'
			for i in range(90):
				x2 = 0
				y2 = y - x * np.tan(np.radians(i))
				if y2 < 0:
					y2 = 0
					x2 = x - height / (np.tan(np.radians(i)))
				x3 = width
				y3 = y - (width-x) * np.tan(np.radians(i))
				if y3 < 0:
					y3 = 0
					x3 = width - height / (np.tan(np.radians(i)))
				fixed_ray1 = Sensor.Ray(x,y,x2,y2,i)	
				fixed_ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				ray1 = Sensor.Ray(x,y,x2,y2,i)	
				ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				self.fixed_rays.append(fixed_ray1)
				self.fixed_rays.append(fixed_ray2)
				self.rays.append(ray1)
				self.rays.append(ray2)
			fixed_ray3 = Sensor.Ray(x,y,x,0,90) #ray for 90 degrees
			ray3 = Sensor.Ray(x,y,x,0,90)
			self.fixed_rays.append(fixed_ray3)
			self.rays.append(ray3)
		else: # facing left
			self.color = 'orange'
			for i in range(90):
				x2 = width - (height-y) * np.tan(np.radians(i))
				y2 = height
				if x2 < 0:
					y2 = height - width / np.tan(np.radians(i))
					x2 = 0
				x3 = width - y * np.tan(np.radians(i))
				y3 = 0
				if x3 < height:
					y3 = y - width / np.tan(np.radians(i))
					x3 = 0
				fixed_ray1 = Sensor.Ray(x,y,x2,y2,i)	
				fixed_ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				ray1 = Sensor.Ray(x,y,x2,y2,i)	
				ray2 = Sensor.Ray(x,y,x3,y3,i+91)	
				self.fixed_rays.append(fixed_ray1)
				self.fixed_rays.append(fixed_ray2)
				self.rays.append(ray1)
				self.rays.append(ray2)
			fixed_ray3 = Sensor.Ray(x,y,0,y,90) #ray for 90 degrees
			ray3 = Sensor.Ray(x,y,0,y,90)
			self.fixed_rays.append(fixed_ray3)
			self.rays.append(ray3)

	def min_from_sensor(self, xvals, yvals):
		xarray = np.asarray(xvals)
		yarray = np.asarray(yvals)
		xidx = (np.abs(xarray - self.x)).argmin()
		yidx = (np.abs(yarray - self.y)).argmin()
		return xarray[xidx], yarray[yidx]
		
	def update_plots(self, plot):
		self.plots = plot

	def update_raytrace(self, objects):
		for i,ray in enumerate(self.plots):
			xmin = 2 * self.width
			ymin = 2 * self.height
			xval = 2 * self.width
			yval = 2 * self.width
			found_intersect = False
			for polygon in objects:
				if self.fixed_rays[i].LineString.intersects(polygon):
					found_intersect = True
					intersect = self.fixed_rays[i].LineString.intersection(polygon)
					if isinstance(intersect, LineString):
						x,y = intersect.xy
					elif isinstance(intersect, Point):
						x,y = intersect.coords.xy
					else:
						x,y = intersect.geoms[0].coords.xy	
					x,y = self.min_from_sensor(x,y)
					if abs(x-self.x) < xmin:
						xmin = abs(x-self.x)
						xval = x
					if abs(y-self.y) < ymin:
						ymin = abs(y-self.y)
						yval = y
			if not found_intersect:
				xval = self.fixed_rays[i].x2
				yval = self.fixed_rays[i].y2
			self.rays[i].update_ray(xval,yval)
			updated_ray = self.rays[i]
			x,y = updated_ray.LineString.xy
			ray.set_data(x,y)
		return self.plots

	def update_sensor_grids(self):
		self.visibility_grid = [[0 for x in range(self.width)] for y in range(self.height)]
		self.occupancy_grid = [[0 for x in range(self.width)] for y in range(self.height)]
		lines = [ray.LineString for ray in self.rays]
		linepoly = linemerge(lines)		
		for x in range(self.width):
			for y in range(self.height):
				b = box(x,y,x+1,y+1)
				if linepoly.intersects(b):
					self.visibility_grid[self.height-1-y][x] = 1 
				"""
				for ray in self.rays:
					if ray.LineString.intersects(b):
						self.visibility_grid[self.height-1-y][x] = 1 
						break		
				"""	
		for ray in self.rays:
			x_index = int(ray.x2)
			y_index = int(ray.y2)
			if(x_index >= 0 and x_index < self.width and y_index >= 0 and y_index < self.height):
				self.occupancy_grid[self.height-1-y_index][x_index] = 1

	def print_visibility_grid(self):
		for y in range(self.height):
			print(self.visibility_grid[y])	
	
	def print_occupancy_grid(self):
		for y in range(self.height):
			print(self.occupancy_grid[y])	
						
	class Ray:

		def __init__(self, x, y, x2, y2, theta):
			self.x = x
			self.y = y
			self.x2 = x2
			self.y2 = y2
			self.LineString = LineString([(x,y), (x2,y2)])
			self.theta = theta
		
		def update_ray(self, x2, y2):
			self.LineString = LineString([(self.x,self.y), (x2,y2)])
			self.x2 = x2
			self.y2 = y2
	
	
def main():

	env = Environment(WIDTH, HEIGHT)
	
	# Sensor Creation
	sensor_0 = Sensor(WIDTH/2, HEIGHT, SOUTH, WIDTH, HEIGHT) # top center
	#sensor_1 = Sensor(WIDTH, HEIGHT, SOUTHWEST, WIDTH, HEIGHT) # top right
	sensor_2 = Sensor(WIDTH, HEIGHT/2, WEST, WIDTH, HEIGHT) # right center
	#sensor_3 = Sensor(WIDTH, 0, NORTHWEST, WIDTH, HEIGHT) # bottom right
	sensor_4 = Sensor(WIDTH/2, 0, NORTH, WIDTH, HEIGHT) # bottom center
	#sensor_5 = Sensor(0, 0, NORTHEAST, WIDTH, HEIGHT) # bottom left
	sensor_6 = Sensor(0, HEIGHT/2, EAST, WIDTH, HEIGHT) # left center
	#sensor_7 = Sensor(0, HEIGHT, SOUTHEAST, WIDTH, HEIGHT) # top left
	
	env.add_sensor(sensor_0)
	#env.add_sensor(sensor_1)
	env.add_sensor(sensor_2)
	#env.add_sensor(sensor_3)
	env.add_sensor(sensor_4)
	#env.add_sensor(sensor_5)
	env.add_sensor(sensor_6)
	#env.add_sensor(sensor_7)
		
	#gui = Gui(env)

	#ani = animation.FuncAnimation(gui.fig, gui.animate, init_func=gui.init,
								  #frames=600, interval=DELAY, blit=True)

	#plt.show()	

	data = []	
	while(env.steps < 10):
		start = timer()
		polygons = []
		step_data = []
		for ball in env.balls:
			polygons.append(Point(ball.x,ball.y).buffer(ball.radius))
		step_data.append(env.env_grid)
		#print('Step {0} environment'.format(env.steps))
		#env.print_grid()
		for i,sensor in enumerate(env.sensors):
			sensor_data = []
			sensor.update_raytrace(polygons)
			sensor.update_sensor_grids()
			sensor_data.append(sensor.visibility_grid)
			sensor_data.append(sensor.occupancy_grid)
			#print('Sensor {0} visibility grid at step {1}'.format(i, env.steps))
			#sensor.print_visibility_grid()	
			#print('Sensor {0} occupancy grid at step {1}'.format(i, env.steps))
			#sensor.print_occupancy_grid()
			step_data.append(sensor_data)
		data.append(step_data)	
		env.update()
		end = timer()
		print('{0}: {1}, {2}'.format(str(env.steps), str(end-start), str(len(env.balls))))	
	np.save('training_data', data)

if __name__ == '__main__':
	main()	
	
