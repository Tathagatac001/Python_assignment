class AreaBasecls(object):
	def __init__(self,a,b,c):
		self.a=a
		self.b=b
		self.c=c
class AreaChildcls(AreaBasecls):
	def __init__(self,a,b,c):
		super(AreaChildcls,self).__init__(a,b,c)
	def area(self):
 		s=(self.a+self.b+self.c)/2
 		area=(s*(s-self.a)*(s-self.b)*(s-self.c)) ** 0.5
 		return area 
  
 #Create an object of child class
 
obj1=AreaChildcls(4,8,10)
print "Area of the triangle is :{0:.2f}".format(obj1.area())
 		
