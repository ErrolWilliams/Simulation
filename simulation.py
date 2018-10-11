import pdb
import descartes
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from shapely.geometry import box, Point, LineString

WIDTH = 50
HEIGHT = 50
MAX_SIZE = 5
MAX_BALLS = 5
STEP = 1
DELAY = 1

class Environment:

	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.balls = []
		self.sensors = []
		self.box = box(0,0,width,height)
		self.env_map = [[0 for x in range(self.width)] for y in range(self.height)]
	
	def add_ball(self, x, y, radius, direction):
		self.balls.append(Ball(x,y,radius,direction))

		
	def add_sensor(self, position):
		if position == 0:  # top
			sensor = Sensor(self.width/2, self.height, 2, self.width, self.height)
			self.sensors.append(sensor)
		elif position == 1: # right
			sensor = Sensor(self.width, self.height/2, 3, self.width, self.height)
			self.sensors.append(sensor)
		elif position == 2: # bottom 
			sensor = Sensor(self.width/2, 0, 0, self.width, self.height)
			self.sensors.append(sensor)
		else:               # left
			sensor = Sensor(0, self.height/2, 1, self.width, self.height)
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
		offscreen = []
		self.env_map = [[0 for x in range(self.width)] for y in range(self.height)]
		for ball in self.balls:
			"""
			b1 = Point(ball.x,ball.y).buffer(ball.radius)
			b2 = Point(ball.x,ball.y).buffer(ball.radius)
			x,y = b1.intersection(b2).coords.xy
			print(x)
			print(y)
			exit(0)
			for i in range(len(x)):
				xcoord = x[i]
				ycoord = y[i]
				print('({0},{1})'.format(xcoord,ycoord))
			exit(0)
			"""
			ball.move()
			if not self.onscreen(ball):
				offscreen.append(ball)
		self.balls = [ball for ball in self.balls if ball not in offscreen]
		if len(self.balls) < np.random.randint(0, MAX_BALLS+1):
			xpos, ypos, r = self.get_valid_randoms()
			direction = np.random.randint(0,7)
			self.add_ball(xpos,ypos,r,direction)
	

	def get_balls(self):
		return self.balls	
	
		
class Ball:

	def __init__(self, x, y, radius, direction):
		self.x = x
		self.y = y
		self.radius = radius
		self.direction = direction
		self.onscreen = False


	def move(self): 
		if self.direction == 0:
			self.x += STEP
		elif self.direction == 1:
			self.x += STEP
			self.y += STEP
		elif self.direction == 2:
			self.y += STEP	
		elif self.direction == 3:
			self.x -= STEP
			self.y += STEP
		elif self.direction == 4:
			self.x -= STEP
		elif self.direction == 5:
			self.y -= STEP
			self.x -= STEP
		elif self.direction == 6:
			self.y -= STEP
		elif self.direction == 7:
			self.x += STEP
			self.y -= STEP


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
		if self.direction == 0: # facing up
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
		elif self.direction == 1: # facing right
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
		elif self.direction == 2: # facing down
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
		#offset = np.tan(np.radians(1))/2
		for ray in self.rays:
			"""	
			for x in range(self.width):
				for y in range(self.height):
					x_dist = ray.x2 - ray.x
					y_dist = ray.y2 - ray.y
					if ray.theta != 90:
						if x != 0
						if (y/x) > np.tan(np.radians(ray.theta))-offset and (y/x) > np.tan(np.radians(ray.theta))+offset:
							visibility_grid[self.height-1-y][x] = 1
		    """	
			x_index = int(ray.x2)
			y_index = int(ray.y2)
			if(x_index >= 0 and x_index < self.width and y_index >= 0 and y_index < self.height):
				self.occupancy_grid[self.height-1-y_index][x_index] = 1
				#self.visibility_grid[self.height-1-y_index][x_index] = 1

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
	#env.add_sensor(0) # North Sensor
	#env.add_sensor(1) # East Sensor
	env.add_sensor(2) # South Sensor
	#env.add_sensor(3) # West Sensor
	gui = Gui(env)

	ani = animation.FuncAnimation(gui.fig, gui.animate, init_func=gui.init,
								  frames=600, interval=DELAY, blit=True)

	plt.show()	


if __name__ == '__main__':
	main()	
	
