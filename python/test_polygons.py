from shapely.geometry import Polygon

# p1: a list of tuples
p1 = Polygon([(0,0), (1,1), (1,0)])
p2 = Polygon([(0,1), (1,0), (1,1)])
print(p1.intersection(p2))