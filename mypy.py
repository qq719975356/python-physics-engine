from tkinter import *
import sys
import time
import math
import random
class V:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __neg__(self):
        return V(-self.x,-self.y)
    def __add__(self,other): #V+V
        return V(self.x + other.x,self.y + other.y)
    def __sub__(self,other): #V-V
        return V(self.x - other.x,self.y - other.y)
    def __truediv__(self,other): #V-num
        return V(self.x / other,self.y / other)
    def __floordiv__(self,other): #V//num
        return V(self.x // other,self.y // other)
    def __mul__(self,other): 
        if isinstance(other,V): #V dot V
            return self.x * other.x + self.y * other.y
        else:
            return V(self.x * other,self.y * other) #V*num
    def __rmul__(self,other): #num*V
        return V(self.x * other,self.y * other)
    def __pow__(self,other): #V cross V, returns a scalar
        return self.x*other.y-self.y*other.x
    def __rpow__(self,other): #num cross V, [0,0,a]X[b,c,0]=[-ac,ab,0]
        return V(-other*self.y,other*self.x)
    def __str__(self): #str(V)
        return ('({}, {})'.format(self.x, self.y))
    def mag(self): 
        return math.sqrt(self.x * self.x + self.y * self.y)
    def mag_sq(self):
        return self.x * self.x + self.y * self.y
    def rot(self,radian): #clockwise
        sin=math.sin(radian)
        cos=math.cos(radian)
        return V(self.x*cos-self.y*sin,self.y*cos+self.x*sin)
    def unitVector(self):
        m = self.mag()
        if m:
            return V(self.x / m,self.y / m)
        else:
            return V(0,0)
class Impulus:
    def __init__(self,vec,pos):
        self.vec=vec
        self.pos=pos

class AABB:
    def __init__(self,minX,maxX,minY,maxY):
        self.minX=minX
        self.minY=minY
        self.maxX=maxX
        self.maxY=maxY
    def set(self,minX,maxX,minY,maxY):
        self.minX=minX
        self.minY=minY
        self.maxX=maxX
        self.maxY=maxY
    def noIntersect(aabb1,aabb2):
        return aabb1.maxX<aabb2.minX or aabb1.maxY<aabb2.minY or aabb1.minX>aabb2.maxX or aabb1.minY>aabb2.maxY
class Entity:
    #canvas: tkinter.Canvas
    #m: mass
    #x,y,rot: center of mass position and rotation
    #dynamic: allowed to move
    #e: coefficient of restitution
    #v: linear  velocity
    #w: angular velocity 
    def __init__(self,canvas,m=0,x=0,y=0,rot=0,dynamic=True,e=1,v=V(0,0),w=0):
        self.pos = V(x,y)
        self.rot = rot
        if dynamic:
            self.m = m
            self.rm = 1 / m
            self.I=None
            self.rI=None
        else:
            self.m = None
            self.I = None
            self.rm = 0
            self.rI = 0
        self.dynamic = dynamic
        self.e = e
        self.v = v
        self.w = w
        self.canvas = canvas
        self.impuluses = []
        self.aabb=AABB(0,0,0,0)

    def velocityAt(self,pos):
        relPos=pos-self.pos
        return self.v+V(-relPos.y,relPos.x)*self.w
    def updateAABB(self):
        raise Exception("abstract method called")
    def update(self,dt):
        raise Exception("abstract method called")
    def calculateInertia(self):
        raise Exception("abstract method called")
class Ball(Entity):
    def __init__(self,canvas,m=0,x=0,y=0,rot=0,dynamic=True,e=1,v=V(0,0),w=0,r=10):
        Entity.__init__(self,canvas,m,x,y,rot,dynamic,e,v,w)
        self.r = r
        self.updateAABB()
        self.shape = self.canvas.create_oval(x - r,y - r,x + r,y + r)
        if self.dynamic:
            self.I=self.calculateInertia()
            self.rI=1/self.I
            print("mass    = "+str(self.m))
            print("Inertia = "+str(self.I))
    def calculateInertia(self):
        return self.m*self.r*self.r/2
    def updateAABB(self):
        self.aabb.set(self.pos.x-self.r,self.pos.x+self.r,self.pos.y-self.r,self.pos.y+self.r)
    def update(self, dt):
        while self.impuluses:
            impulus=self.impuluses.pop()
            self.v = self.v + impulus.vec * self.rm
            #rotation not implemented
        dr = self.v * dt
        self.pos = self.pos + dr
        self.updateAABB()
        self.canvas.coords(self.shape,self.pos.x-self.r,self.pos.y-self.r,self.pos.x+self.r,self.pos.y+self.r)
class Polygon(Entity):
    def __init__(self,canvas,m=0,x=0,y=0,rot=0,dynamic=True,e=1,v=V(0,0),w=0,verteies=None):#verteies will be stored clockwise
        if len(verteies)<3:
            raise Exception("polygon.verteies argument invalid")
        Entity.__init__(self,canvas,m,x,y,rot,dynamic,e,v,w)
        self.verteies=verteies                       #list<V>   reletive position
        self.verteiesCount=len(verteies)
        if not Polygon.isVerteiesClockwise(verteies):
            self.verteies.reverse()
        self.reallocateCenterOfMass()
        self.verteies_real=[None]*len(self.verteies) #list<V>   world position
        self.verteies_draw=[None]*(len(self.verteies)*2) #list<num> draw position for canvas
        self.unitNormals=[None]*len(self.verteies)
        self.updateVerteciesPosition()
        self.updateNormals()
        self.updateAABB()
        self.shape=self.canvas.create_polygon(self.verteies_draw,fill='white',outline='black')
        if dynamic:
            self.I=self.calculateInertia()
            self.rI=1/self.I
            print("mass    = "+str(self.m))
            print("Inertia = "+str(self.I))
    def updateAABB(self):
        minX=sys.maxsize
        minY=sys.maxsize
        maxX=-sys.maxsize
        maxY=-sys.maxsize
        for vertex in self.verteies_real:
            if vertex.x<minX:
                minX=vertex.x
            if vertex.y<minY:
                minY=vertex.y
            if vertex.x>maxX:
                maxX=vertex.x
            if vertex.y>maxY:
                maxY=vertex.y
        self.aabb.set(minX,maxX,minY,maxY)
    def reallocateCenterOfMass(self):
        print("Polygon.reallocateCenterOfMass not implemented")
    def getUnitNormal(v1,v2):
        v=v1-v2
        return V(-v.y,v.x).unitVector()
    def updateNormals(self):
        for i in range(0,len(self.verteies_real)):
            self.unitNormals[i]=Polygon.getUnitNormal(self.verteies_real[i],self.verteies_real[(i+1)%self.verteiesCount])
    def updateVerteciesPosition(self):
        for i in range(0,len(self.verteies)):
            self.verteies_real[i]=self.verteies[i].rot(self.rot)+self.pos
            self.verteies_draw[i*2]  =self.verteies_real[i].x
            self.verteies_draw[i*2+1]=self.verteies_real[i].y
    def update(self, dt):
        while self.impuluses:
            impulus=self.impuluses.pop()
            self.v = self.v + impulus.vec * self.rm
            self.w = self.w + impulus.vec ** (self.pos-impulus.pos) * self.rI
        self.pos = self.pos + self.v * dt
        self.rot = self.rot + self.w * dt
        self.updateVerteciesPosition()
        self.updateAABB()
        self.updateNormals()
        self.canvas.coords(self.shape,self.verteies_draw)
    def calculateInertia(self): #https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        numerator=0
        denominator=0
        for n in range(0,len(self.verteies)):
            v1=self.verteies[n]
            v2=self.verteies[(n+1)%len(self.verteies)]
            numerator+=abs((v1**v2)*(v1*v1+v1*v2+v2*v2))
            denominator+=abs(v1**v2)
        return (self.m/6*numerator/denominator)
    def isVerteiesClockwise(verteies):
        print("Polygon.isVerteiesClockwise(verteies) not implemented")
        return True
class Penetration:
    def __init__(self,e1,e2,normal,depth):
        self.e1=e1
        self.e2=e2
        self.normal=normal
        self.depth=depth
class World:
    def __init__(self,width=800,height=600,gravaty=5,worldSpeedMultiplier=1):
        self.tk = Tk() #main window
        self.width = width
        self.height = height
        self.c = Canvas(self.tk,width=self.width,height=self.height,bg="white")
        self.c.pack()
        self.entityList = []
        self.penetrations = []
        if isinstance(gravaty,V):
            self.gravaty = gravaty
        else:
            self.gravaty = V(0,gravaty)
        self.worldSpeedMultiplier=worldSpeedMultiplier
        self.timeOfLastFrame = None
    def AddEntity(self,entity):
        if entity:
            self.entityList.append(entity)
    def delEntity(self,index):
        if entityList[index]:
            self.c.delete(self.entityList[index].shape)
            del self.entityList[index]
        else:
            raise Exception("index invalid")
    def delLastEntity(self):
        if len(self.entityList):
            self.c.delete(self.entityList.pop().shape)
        else:
            raise Exception("entityList is empty")
    def Start(self):
        self.timeOfLastFrame = time.clock()
        self.Loop()
    def getDeltaTime(self):
        clock=time.clock()*self.worldSpeedMultiplier
        result=clock - self.timeOfLastFrame
        self.timeOfLastFrame=clock
        return result
    def Loop(self):
        while True:
            cursorX=self.tk.winfo_pointerx() - self.tk.winfo_rootx()
            cursorY=self.tk.winfo_pointery() - self.tk.winfo_rooty()
            print(str(cursorX)+":"+str(cursorY))
            dt = self.getDeltaTime()
            for e in self.entityList:
                if e.dynamic:
                    if self.gravaty:
                        e.v = e.v + self.gravaty*dt
                    e.update(dt)
            for i in range(0,len(self.entityList)):
                for j in range(i + 1,len(self.entityList)):
                    if self.entityList[i].dynamic or self.entityList[j].dynamic:
                        if self.CheckCollision(self.entityList[i],self.entityList[j]):
                            #print("\a")
                            pass
            while self.penetrations:
                self.unpenetrate(self.penetrations.pop())
            self.tk.update()
    def unpenetrate(self,penetration):#e1 will move towards normal
        e1=penetration.e1
        e2=penetration.e2
        correction = penetration.depth / (e1.rm + e2.rm) * penetration.normal *0.5
        e1.pos += e1.rm * correction
        e2.pos -= e2.rm * correction
    def CheckCollision_ball_ball(self,b1,b2):
        dis = b1.pos - b2.pos
        if (b1.r + b2.r)**2 < dis.mag_sq():
            return False
        else:
            normal = dis.unitVector()
            relVel = b1.v - b2.v
            if normal*relVel>0:
                return True
            e = min(b1.e,b2.e)
            impulus = (relVel * normal * (1 + e) / (b1.rm + b2.rm)) * normal
            b1.impuluses.append(Impulus(vec=-impulus,pos=None))
            b2.impuluses.append(Impulus(vec=impulus,pos=None))
            self.penetrations.append(Penetration(b1,b2,normal,b1.r+b2.r-dis.mag()))
            return True
    def CheckCollision_ball_poly(self,b1,p1):
        minPenetrationDepth=sys.maxsize
        minPenetrationNormal=None
        collisionPos=None
        for vertex in p1.verteies_real:
            normal=(b1.pos-vertex).unitVector()
            p1MinProj=sys.maxsize
            p1MaxProj=-sys.maxsize
            for vertex in p1.verteies_real:
                projection=vertex*normal
                if p1MinProj>projection:
                    p1MinProj=projection
                if p1MaxProj<projection:
                    p1MaxProj=projection
            b1CenterProj=b1.pos*normal
            b1MinProj=b1CenterProj-b1.r
            b1MaxProj=b1CenterProj+b1.r
            if p1MinProj>b1MaxProj or p1MaxProj<b1MinProj:
                return False
            else:
                depth=min(abs(b1MaxProj-p1MinProj),abs(p1MaxProj-b1MinProj))
                if minPenetrationDepth>depth:
                    minPenetrationDepth=depth
                    minPenetrationNormal=normal
                    collisionPos=vertex
        for i in range(0,len(p1.unitNormals)):
            normal=p1.unitNormals[i]
            p1MinProj=sys.maxsize
            p1MaxProj=-sys.maxsize
            for vertex in p1.verteies_real:
                projection=vertex*normal
                if p1MinProj>projection:
                    p1MinProj=projection
                if p1MaxProj<projection:
                    p1MaxProj=projection
            b1CenterProj=b1.pos*normal
            b1MinProj=b1CenterProj-b1.r
            b1MaxProj=b1CenterProj+b1.r
            if p1MinProj>b1MaxProj or p1MaxProj<b1MinProj:
                return False
            else:
                if b1MinProj<=p1.verteies_real[i]*normal<=b1MaxProj:
                    depth=min(abs(b1MaxProj-p1MinProj),abs(p1MaxProj-b1MinProj))
                    if minPenetrationDepth>depth:
                        minPenetrationDepth=depth
                        minPenetrationNormal=normal
                        collisionPos=b1.pos-b1.r*normal
        self.penetrations.append(Penetration(b1,p1,minPenetrationNormal,minPenetrationDepth))
        normal=minPenetrationNormal
        relVel=p1.velocityAt(collisionPos)-b1.velocityAt(collisionPos)
        if relVel*normal<0:
            return True
        e = min(p1.e,b1.e)
        r1=collisionPos-b1.pos
        r2=collisionPos-p1.pos
        impulus=(relVel*normal*(1+e)/(b1.rm+p1.rm+(((r1**normal)**r1)*b1.rI+((r2**normal)**r2)*p1.rI)*normal))*normal
        b1.impuluses.append(Impulus(vec=impulus,pos=collisionPos))
        p1.impuluses.append(Impulus(vec=-impulus,pos=collisionPos))
        return True
    def CheckCollision_poly_poly(self,p1,p2):
        minPenetrationDepth=sys.maxsize
        minPenetrationNormal=None
        collisionPos=None
        tempCollisionPos=None
        for i in range(0,len(p1.unitNormals)):
            normal=p1.unitNormals[i]
            p1MinProj=sys.maxsize
            p1MaxProj=-sys.maxsize
            for vertex in p1.verteies_real:
                projection=vertex*normal
                if p1MinProj>projection:
                    p1MinProj=projection
                if p1MaxProj<projection:
                    p1MaxProj=projection
            p2MinProjVertex=None
            p2MaxProjVertex=None
            p2MinProj=sys.maxsize
            p2MaxProj=-sys.maxsize
            for vertex in p2.verteies_real:
                projection=vertex*normal
                if p2MinProj>projection:
                    p2MinProj=projection
                    p2MinProjVertex=vertex
                if p2MaxProj<projection:
                    p2MaxProj=projection
                    p2MaxProjVertex=vertex
            if p1MinProj>p2MaxProj or p1MaxProj<p2MinProj:
                return False
            elif p2MinProj<=p1.verteies_real[i]*normal<=p2MaxProj:
                    projectionDis1=abs(p2MaxProj-p1MinProj)
                    projectionDis2=abs(p1MaxProj-p2MinProj)
                    
                    if projectionDis1<projectionDis2:
                        depth=projectionDis1
                        tempCollisionPos=p2MaxProjVertex
                    else:
                        depth=projectionDis2
                        tempCollisionPos=p2MinProjVertex
                    if minPenetrationDepth>depth:
                        minPenetrationDepth=depth
                        minPenetrationNormal=-normal
                        collisionPos=tempCollisionPos
        for i in range(0,len(p2.unitNormals)):
            normal=p2.unitNormals[i]
            p2MinProj=sys.maxsize
            p2MaxProj=-sys.maxsize
            for vertex in p2.verteies_real:
                projection=vertex*normal
                if p2MinProj>projection:
                    p2MinProj=projection
                if p2MaxProj<projection:
                    p2MaxProj=projection
            p1MinProjVertex=None
            p1MaxProjVertex=None
            p1MinProj=sys.maxsize
            p1MaxProj=-sys.maxsize
            for vertex in p1.verteies_real:
                projection=vertex*normal
                if p1MinProj>projection:
                    p1MinProj=projection
                    p1MinProjVertex=vertex
                if p1MaxProj<projection:
                    p1MaxProj=projection
                    p1MaxProjVertex=vertex
            if p2MinProj>p1MaxProj or p2MaxProj<p1MinProj:
                return False
            elif p1MinProj<=p2.verteies_real[i]*normal<=p1MaxProj:
                    projectionDis1=abs(p2MaxProj-p1MinProj)
                    projectionDis2=abs(p1MaxProj-p2MinProj)
                    if projectionDis1<projectionDis2:
                        depth=projectionDis1
                        tempCollisionPos=p1MinProjVertex
                    else:
                        depth=projectionDis2
                        tempCollisionPos=p1MaxProjVertex
                    if minPenetrationDepth>depth:
                        minPenetrationDepth=depth
                        minPenetrationNormal=normal
                        collisionPos=tempCollisionPos
        self.penetrations.append(Penetration(p1,p2,minPenetrationNormal,minPenetrationDepth))
        normal=minPenetrationNormal
        relVel=p2.velocityAt(collisionPos)-p1.velocityAt(collisionPos)
        if relVel*normal<0:
            return True
        e = min(p1.e,p1.e)
        r1=collisionPos-p2.pos
        r2=collisionPos-p1.pos
        impulus=(relVel*normal*(1+e)/(p2.rm+p1.rm+(((r1**normal)**r1)*p2.rI+((r2**normal)**r2)*p1.rI)*normal))*normal      
        p1.impuluses.append(Impulus(vec=impulus,pos=collisionPos))
        p2.impuluses.append(Impulus(vec=-impulus,pos=collisionPos))
        return True
    def CheckCollision(self,e1,e2):
        if AABB.noIntersect(e1.aabb,e2.aabb):
            return False
        if isinstance(e1,Ball):
            if isinstance(e2,Polygon):
                return self.CheckCollision_ball_poly(e1,e2)
            if isinstance(e2,Ball):
                return self.CheckCollision_ball_ball(e1,e2)
        elif isinstance(e1,Polygon):
            if isinstance(e2,Polygon):
                return self.CheckCollision_poly_poly(e1,e2)
            if isinstance(e2,Ball):
                return self.CheckCollision_ball_poly(e2,e1)
HEIGHT = 1000
WIDTH = 1920
PI=3.1415926
def createBalls(w,numBalls=100,e=0.5):
    for i in range(0,numBalls):
        radius=random.randrange(10, 70)
        mass=radius**2*PI/10
        w.AddEntity(Ball(w.c,m=mass,x=random.randrange(150, WIDTH - 150),y=random.randrange(150,HEIGHT - 150),v=V(random.randrange(-3,3),random.randrange(-30,30)),r=radius,e=e))
        for j in range(0,len(w.entityList) - 1):
            if w.CheckCollision(w.entityList[-1],w.entityList[j]):
                w.delLastEntity()
def createContainerBalls(w):
    for i in range(1,8):
        w.AddEntity(Ball(w.c,m=100,x=WIDTH / 8 * i,y=0,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
        w.AddEntity(Ball(w.c,m=100,x=WIDTH / 8 * i,y=HEIGHT,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
    for i in range(1,8):
        w.AddEntity(Ball(w.c,m=100,x=0,y=HEIGHT / 8 * i,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
        w.AddEntity(Ball(w.c,m=100,x=WIDTH,y=HEIGHT / 8 * i,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
def createContainerRects(w):
    w.AddEntity(Polygon(w.c,m=100,x=WIDTH/2,y=HEIGHT,v=V(0,0),dynamic=False,verteies=[V(-WIDTH/2,-20),V(WIDTH/2,-20),V(WIDTH/2,20),V(-WIDTH/2,20)]))
    w.AddEntity(Polygon(w.c,m=100,x=WIDTH/2,y=0,v=V(0,0),dynamic=False,verteies=[V(-WIDTH/2,-20),V(WIDTH/2,-20),V(WIDTH/2,20),V(-WIDTH/2,20)]))
    w.AddEntity(Polygon(w.c,m=100,x=0,y=HEIGHT/2,v=V(0,0),dynamic=False,verteies=[V(-20,-HEIGHT/2),V(20,-HEIGHT/2),V(20,HEIGHT/2),V(-20,HEIGHT/2)]))
    w.AddEntity(Polygon(w.c,m=100,x=WIDTH,y=HEIGHT/2,v=V(0,0),dynamic=False,verteies=[V(-20,-HEIGHT/2),V(20,-HEIGHT/2),V(20,HEIGHT/2),V(-20,HEIGHT/2)]))
def createSquares(w,numSquares=100,e=0.5):
    for i in range(0,numSquares):
        a=random.randrange(10, 70)
        mass=a**2/10
        w.AddEntity(Polygon(w.c,e=e,m=mass,x=random.randrange(150, WIDTH - 150),y=random.randrange(150,HEIGHT - 150),rot=random.randrange(-3,3),v=V(random.randrange(-3,3),random.randrange(-30,30)),w=random.randrange(-3,3),verteies=[V(-a,-a),V(a,-a),V(a,a),V(-a,a)]))
        for j in range(0,len(w.entityList) - 1):
            if w.CheckCollision(w.entityList[-1],w.entityList[j]):
                w.delLastEntity()
def main():
    world = World(WIDTH,HEIGHT,gravaty=10,worldSpeedMultiplier=10)
    createSquares(world,10)
    createBalls(world,10)
    #w.AddEntity(Polygon(w.c,m=10,x=550,y=700,rot=PI/4*5,v=V(0,10),w=0,verteies=[V(-30,-30),V(30,-30),V(30,30),V(-30,30)]))
    #w.AddEntity(Polygon(w.c,m=10,x=550,y=200,rot=PI/4,v=V(10,10),w=1,verteies=[V(-30,-30),V(30,-30),V(30,30),V(-30,30)]))
    #w.AddEntity(Ball(w.c,m=100,x=500,y=300,rot=1,v=V(15,-5),r=28))
    #w.AddEntity(Polygon(w.c,m=10,x=650,y=400,rot=0,v=V(-10,-5),w=1,verteies=[V(0,30),V(-20,20),V(-30,0),V(-20,-20),V(0,-30),V(20,-20),V(30,0),V(20,20)]))
    #createContainerBalls(w)
    createContainerRects(world)
    world.Start()
main()