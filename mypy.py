import sys
import time
import math
import random
from tkinter import *
from win32api import *
class V:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def set(self,x,y):
        self.x = x
        self.y = y
    def __eq__(self,other):
        if not isinstance(other,V):
            return False
        return self.x==other.x and self.y==other.y
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
    def clone(self):
        return V(self.x,self.y)
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
def CalculateTriangleArea(a,b,c):
    return abs(0.5*(a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y)))
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
    def __init__(self,world,canvas,m=0,x=0,y=0,rot=0,dynamic=True,e=1,v=V(0,0),w=0):
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
        self.world=world
        self.deleteMe=False
        self.shape=None
    def velocityAt(self,pos):
        relPos=pos-self.pos
        return self.v+V(-relPos.y,relPos.x)*self.w
    def updateAABB(self):
        raise Exception("abstract method called")
    def update(self,dt):
        raise Exception("abstract method called")
    def calculateInertia(self):
        raise Exception("abstract method called")
    def setColor(self,color):
        self.canvas.itemconfig(self.shape, fill=color)
class Ball(Entity):
    def __init__(self,world,canvas,m=0,x=0,y=0,rot=0,dynamic=True,e=1,v=V(0,0),w=0,r=10,color="#FFFFFF"):
        Entity.__init__(self,world,canvas,m,x,y,rot,dynamic,e,v,w)
        self.color=color
        self.r = r
        self.r2=r*r
        self.area=math.pi*self.r2
        self.updateAABB()
        self.shape = self.canvas.create_oval(x - r,y - r,x + r,y + r,outline=world.color)
        if self.dynamic:
            self.I=self.calculateInertia()
            self.rI=1/self.I
            print("Ball created")
            print("area    = "+str(self.area))
            print("mass    = "+str(self.m))
            print("Inertia = "+str(self.I))
        self.setColor(color)
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
    def __init__(self,world,canvas,m=0,x=0,y=0,rot=0,dynamic=True,e=1,v=V(0,0),w=0,verteies=None,color="#FFFFFF"):#verteies will be stored clockwise
        if len(verteies)<3:
            raise Exception("polygon.verteies argument invalid")
        Entity.__init__(self,world,canvas,m,x,y,rot,dynamic,e,v,w)
        self.color=color
        self.verteies=verteies                       #list<V>   reletive position
        self.verteiesCount=len(verteies)
        self.area=self.calArea()
        if not Polygon.areVerteiesClockwise(verteies):
            self.verteies.reverse()
        self.reallocateCenterOfMass()
        self.verteies_real=[None]*len(self.verteies) #list<V>   world position
        self.verteies_draw=[None]*(len(self.verteies)*2) #list<num> draw position for canvas
        self.unitNormals=[None]*len(self.verteies)
        self.updateVerteciesPosition()
        self.updateNormals()
        self.updateAABB()
        self.shape=self.canvas.create_polygon(self.verteies_draw,fill='white',outline=world.color)
        if dynamic:
            self.I=self.calculateInertia()
            self.rI=1/self.I
            print("Polygon created")
            print("area    = "+str(self.area))
            print("mass    = "+str(self.m))
            print("Inertia = "+str(self.I))
        self.setColor(color)
    def calArea(self):
        sum=0
        for i in range(0,self.verteiesCount):
            sum+=self.verteies[i]**self.verteies[(i+1)%self.verteiesCount]
        return abs(sum/2)
    def calAreabyVerteies(vert):
        sum=0
        for i in range(0,len(vert)):
            sum+=vert[i]**vert[(i+1)%len(vert)]
        return abs(sum/2)
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
        cx=0
        cy=0
        for i in range(0,self.verteiesCount):
            v1=self.verteies[i]
            v2=self.verteies[(i+1)%self.verteiesCount]
            cx+=(v1.x+v2.x)*(v1.x*v2.y-v2.x*v1.y)
            cy+=(v1.y+v2.y)*(v1.x*v2.y-v2.x*v1.y)
        cx=cx/6/self.area
        cy=cy/6/self.area
        cp=V(cx,cy)
        for i in range(0,self.verteiesCount):
            self.verteies[i]-=cp
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
    def areVerteiesClockwise(verteies):
        print("Polygon.isVerteiesClockwise(verteies) not implemented")
        return True
class Penetration:
    def __init__(self,e1,e2,normal,depth):
        self.e1=e1
        self.e2=e2
        self.normal=normal
        self.depth=depth
class World:
    cursorMemorySize=10
    def __init__(self,width=800,height=600,gravaty=5,worldSpeedMultiplier=1,color="#1E1E1E"):
        self.tk = Tk() #main window
        self.tk.title('')
        self.tk.wm_attributes('-fullscreen','true')
        self.width = width
        self.height = height
        self.color=color
        self.c = Canvas(self.tk,width=self.width,height=self.height,bg=color)
        self.c.pack()
        self.entityList = []
        self.penetrations = []
        if isinstance(gravaty,V):
            self.gravaty = gravaty
        else:
            self.gravaty = V(0,gravaty)
        self.g=self.gravaty.mag()
        self.worldSpeedMultiplier=worldSpeedMultiplier
        self.timeOfLastFrame = None
        self.paused = False
        self.cursorPos=V(0,0)
        self.cursorVelocity=V(0,0)
        self.cursorPosHistory=[self.cursorPos]*World.cursorMemorySize
        self.cursorPosHistoryIndex=0
        self.updateCursorPos()
    def addEntity(self,entity):
        if entity:
            self.entityList.append(entity)
    def addRandomBallAtPos(self,pos,e=1,v=V(0,0)):
        self.addEntity(self.generateRandomBallAtPos(pos,e,v=v))
    def addRandomPolygonAtPos(self,pos,e=1,v=V(0,0)):
        self.addEntity(self.generateRandomPolygonAtPos(pos,e=e,v=v))
    def generateRandomBallAtPos(self,pos,e=1,v=V(0,0)):
        radius=random.randrange(10, 70)
        mass=radius**2*PI/10
        return Ball(self,self.c,m=mass,x=pos.x,y=pos.y,v=v,r=radius,e=e,color=generateRandomColor())
    def generateRandomPolygonAtPos(self,pos,maxVertexCount=7,e=1,radius=100,density=0.1,v=V(0,0)):
        verteiesCount=random.randint(3,maxVertexCount+1)
        vertexAngles=[0]*verteiesCount
        verteies=[None]*verteiesCount
        for i in range(0,verteiesCount):
            vertexAngles[i]=random.uniform(0,2*math.pi)
        vertexAngles.sort()
        for i in range(0,verteiesCount):
            verteies[i]=V(0,radius).rot(vertexAngles[i])
        return Polygon(self,self.c,Polygon.calAreabyVerteies(verteies)*density,pos.x,pos.y,v=v,e=e,verteies=verteies,color=generateRandomColor())
    def IsPosOutOfBound(self,pos):
        return pos.x>self.width*2 or pos.y>self.height*2 or pos.x<-self.width or pos.y<-self.height
    def deleteOutOfBoundEntities(self):
        for e in self.entityList:
            if self.IsPosOutOfBound(e.pos):
                e.deleteMe=True
    def delEntityAt(self,index):
        if self.entityList[index]:
            self.c.delete(self.entityList[index].shape)
            del self.entityList[index]
        else:
            raise Exception("index invalid")
    def delLastEntity(self):
        if len(self.entityList):
            self.c.delete(self.entityList.pop().shape)
        else:
            raise Exception("entityList is empty")
    def start(self):
        self.timeOfLastFrame = time.clock()
        self.Loop()
    def updateCursorPos(self):
        self.cursorPos.set(self.tk.winfo_pointerx() - self.tk.winfo_rootx(),self.tk.winfo_pointery() - self.tk.winfo_rooty())
        self.cursorPosHistory[self.cursorPosHistoryIndex]=self.cursorPos.clone()
        self.cursorPosHistoryIndex=(self.cursorPosHistoryIndex+1)%World.cursorMemorySize
        self.cursorVelocity=self.cursorPos-self.cursorPosHistory[self.cursorPosHistoryIndex]
    def emptyCursorHistory(self):
        self.cursorPosHistory=[self.cursorPos]*World.cursorMemorySize
        self.cursorVelocity=V(0,0)
    def getDeltaTime(self):
        clock=time.clock()*self.worldSpeedMultiplier
        result=clock - self.timeOfLastFrame
        self.timeOfLastFrame=clock
        return result
    def Loop(self):
        controlledEntity=None
        while True:
            self.updateCursorPos()
            if GetAsyncKeyState(27)==-32767:#esc pressed, exit
                sys.exit()
            if GetAsyncKeyState(32)==-32767:#space-bar pressed, pause/unpause
                self.paused=not self.paused
                self.emptyCursorHistory()
            if GetAsyncKeyState(71):#g held, remove entities
                for i in reversed(range(0,len(self.entityList))):
                    if self.checkContainPoint(self.entityList[i],self.cursorPos):
                        self.delEntityAt(i)
            if self.paused:
                self.getDeltaTime()
                self.tk.update()
                continue
            dt = self.getDeltaTime()
            self.deleteOutOfBoundEntities()
            self.removeDeletedEntities()
            if controlledEntity:
                if GetAsyncKeyState(104): #numpad 8 held, controlled entity acc up
                    controlledEntity.v = controlledEntity.v + 30 * V(0,-1)*dt
                if GetAsyncKeyState(102): #numpad 6 held, controlled entity acc right
                    controlledEntity.v = controlledEntity.v + 30 * V(1, 0)*dt
                if GetAsyncKeyState(101): #numpad 5 held, controlled entity acc down
                    controlledEntity.v = controlledEntity.v + 30 * V(0, 1)*dt
                if GetAsyncKeyState(100): #numpad 4 held, controlled entity acc left
                    controlledEntity.v = controlledEntity.v + 30 * V(-1,0)*dt
                if GetAsyncKeyState(105): #numpad 9 held, controlled entity acc clockwise spin
                    controlledEntity.w = controlledEntity.w + 0.3*dt
                if GetAsyncKeyState(103): #numpad 7 held, controlled entity acc conter-clockwise spin
                    controlledEntity.w = controlledEntity.w - 0.3*dt   
                if GetAsyncKeyState(96): #numpad 0 held, repel every other entity
                    for e in self.entityList:
                        if e.dynamic and e is not controlledEntity:
                            e.v = e.v - 0.1 * (controlledEntity.pos-e.pos)*dt
                if GetAsyncKeyState(97): #numpad 1 held, attract every other entity
                    for e in self.entityList:
                        if e.dynamic and e is not controlledEntity:
                            e.v = e.v + 0.1 * (controlledEntity.pos-e.pos)*dt
            if GetAsyncKeyState(65):#a held, acc left
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(-1,0)*dt
            if GetAsyncKeyState(83):#s held, acc down
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(0,1)*dt
            if GetAsyncKeyState(68):#d held, acc right
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(1,0)*dt
            if GetAsyncKeyState(87):#w held, acc up
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(0,-1)*dt
            if GetAsyncKeyState(70)==-32767:#f pressed, velocity and angular velocity flip
                for e in self.entityList:
                    if e.dynamic:
                        e.v=-e.v
                        e.w=-e.w
            if GetAsyncKeyState(88):#x held, dec towards current velocity and angular velocity
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v*(1-0.4*dt)
                        e.w = e.w*(1-0.4*dt)
            if GetAsyncKeyState(81):#q held, acc towards current velocity and angular velocity
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v*(0.4*dt+1)
                        e.w = e.w*(1-0.4*dt)
            if GetAsyncKeyState(90):#z held, acc randomly
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 300*V(random.uniform(-1,1),random.uniform(-1,1))*dt
            if GetAsyncKeyState(82)==-32767:#r pressed, reverse gravity
                if self.gravaty==V(0,0):
                    self.gravaty=V(self.g,0)
                self.gravaty=-self.gravaty
            if GetAsyncKeyState(84)==-32767:#t pressed, enable/disable gravaty
                self.gravaty=V(0,self.g) if self.gravaty==V(0,0) else V(0,0)
            if GetAsyncKeyState(67)==-32767:#c pressed, add ball at cursor pos with cursor velocity
                self.addRandomBallAtPos(self.cursorPos,v=self.cursorVelocity)
            if GetAsyncKeyState(86)==-32767:#v pressed, add poly at cursor pos with cursor velocity
                self.addRandomPolygonAtPos(self.cursorPos,v=self.cursorVelocity)
            if GetAsyncKeyState(1):#left-mouse down, attract entities
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 0.1 * (self.cursorPos-e.pos)*dt
            if GetAsyncKeyState(2):#right-mouse down, repel entities
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v - 0.1 * (self.cursorPos-e.pos)*dt
            if GetAsyncKeyState(4):#mid-mouse down, select/de-select entity to be controlled
                foundEntity=False
                for i in reversed(range(0,len(self.entityList))):
                    if self.checkContainPoint(self.entityList[i],self.cursorPos):
                        if controlledEntity:
                            controlledEntity.setColor(controlledEntity.color)
                        controlledEntity=self.entityList[i]
                        controlledEntity.setColor("#000000")
                        foundEntity=True
                        break
                if not foundEntity:
                    if controlledEntity:
                        controlledEntity.setColor(controlledEntity.color)
                    controlledEntity=None
            for e in self.entityList:#apply gravaty
                if e.dynamic:
                    if self.gravaty:
                        e.v = e.v + self.gravaty*dt
                    e.update(dt)
            for i in range(0,len(self.entityList)):#check collision
                for j in range(i + 1,len(self.entityList)):
                    if self.entityList[i].dynamic or self.entityList[j].dynamic:
                        if self.checkCollision(self.entityList[i],self.entityList[j]):
                            #print("\a")
                            pass
            while self.penetrations:#un-penetrate
                self.unpenetrate(self.penetrations.pop())
            self.tk.update()#update this frame
    def removeDeletedEntities(self):
        for i in reversed(range(len(self.entityList))):
            if self.entityList[i].deleteMe:
                self.delEntityAt(i)
    def unpenetrate(self,penetration):#e1 will move towards normal
        e1=penetration.e1
        e2=penetration.e2
        correction = penetration.depth / (e1.rm + e2.rm) * penetration.normal *0.5
        e1.pos += e1.rm * correction
        e2.pos -= e2.rm * correction
    def checkCollision_ball_ball(self,b1,b2):
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
    def checkCollision_ball_poly(self,b1,p1):
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
    def checkCollision_poly_poly(self,p1,p2):
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
    def checkCollision_ball_V(self,b,v):
        return (b.pos-v).mag_sq()<b.r2
    def checkCollision_poly_V(self,p,v):
        areaSum=0
        for i in range(0,p.verteiesCount):
            v1=p.verteies_real[i]
            v2=p.verteies_real[(i+1)%p.verteiesCount]
            areaSum+=CalculateTriangleArea(v1,v2,v)
        if areaSum>p.area*1.01:
            return False
        return True
    def checkCollision_poly_V_by_verteies(vert,v,area):
        areaSum=0
        for i in range(0,len(vert)):
            v1=vert[i]
            v2=vert[(i+1)%len(vert)]
            areaSum+=CalculateTriangleArea(v1,v2,v)
        if areaSum>area:
            return False
        return True
    def checkContainPoint(self,e,v):
        if isinstance(e,Ball):
            return self.checkCollision_ball_V(e,v)
        if isinstance(e,Polygon):
            return self.checkCollision_poly_V(e,v)
    def checkCollision(self,e1,e2):
        #if isinstance(e1,V):
        #    if isinstance(e2,Ball):
        #        return self.CheckCollision_ball_V(e2,e1)
        #    if isinstance(e2,Polygon):
        #        return self.CheckCollision_poly_V(e2,e1)
        #if isinstance(e2,V):
        #    if isinstance(e1,Ball):
        #        return self.CheckCollision_ball_V(e1,e2)
        #    if isinstance(e1,Polygon):
        #        return self.CheckCollision_poly_V(e1,e2)
        if AABB.noIntersect(e1.aabb,e2.aabb):
            return False
        if isinstance(e1,Ball):
            if isinstance(e2,Polygon):
                return self.checkCollision_ball_poly(e1,e2)
            if isinstance(e2,Ball):
                return self.checkCollision_ball_ball(e1,e2)
        elif isinstance(e1,Polygon):
            if isinstance(e2,Polygon):
                return self.checkCollision_poly_poly(e1,e2)
            if isinstance(e2,Ball):
                return self.checkCollision_ball_poly(e2,e1)
HEIGHT = 1080
WIDTH = 1920
PI=math.pi
def generateRandomColor(min=150,max=255):
    r=random.randint(min,max)
    g=random.randint(min,max)
    b=random.randint(min,max)
    if (r-g)**2+(g-b)**2+(b-r)**2*40<(max-min)**2:
        return generateRandomColor(min,max)
    return '#%02x%02x%02x' % tuple([r,g,b])
def createBalls(w,numBalls=100,e=1):
    for i in range(0,numBalls):
        radius=random.randrange(10, 70)
        mass=radius**2*PI/10
        w.addEntity(Ball(w,w.c,m=mass,x=random.randrange(150, WIDTH - 150),y=random.randrange(150,HEIGHT - 150),v=V(random.randrange(-3,3),random.randrange(-30,30)),r=radius,e=e,color=generateRandomColor()))
        for j in range(0,len(w.entityList) - 1):
            if w.checkCollision(w.entityList[-1],w.entityList[j]):
                w.delLastEntity()
def createContainerBalls(w):
    for i in range(1,8):
        w.AddEntity(Ball(w,w.c,m=100,x=WIDTH / 8 * i,y=0,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
        w.AddEntity(Ball(w,w.c,m=100,x=WIDTH / 8 * i,y=HEIGHT,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
    for i in range(1,8):
        w.AddEntity(Ball(w,w.c,m=100,x=0,y=HEIGHT / 8 * i,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
        w.AddEntity(Ball(w,w.c,m=100,x=WIDTH,y=HEIGHT / 8 * i,v=V(0,0),r=WIDTH / 12,dynamic=FALSE))
def createContainerRects(w):
    w.addEntity(Polygon(w,w.c,m=100,x=WIDTH/2,y=HEIGHT,v=V(0,0),dynamic=False,verteies=[V(-WIDTH/2,-20),V(WIDTH/2,-20),V(WIDTH/2,20),V(-WIDTH/2,20)],color='#101010'))
    w.addEntity(Polygon(w,w.c,m=100,x=WIDTH/2,y=0,v=V(0,0),dynamic=False,verteies=[V(-WIDTH/2,-20),V(WIDTH/2,-20),V(WIDTH/2,20),V(-WIDTH/2,20)],color='#101010'))
    w.addEntity(Polygon(w,w.c,m=100,x=0,y=HEIGHT/2,v=V(0,0),dynamic=False,verteies=[V(-20,-HEIGHT/2),V(20,-HEIGHT/2),V(20,HEIGHT/2),V(-20,HEIGHT/2)],color='#101010'))
    w.addEntity(Polygon(w,w.c,m=100,x=WIDTH,y=HEIGHT/2,v=V(0,0),dynamic=False,verteies=[V(-20,-HEIGHT/2),V(20,-HEIGHT/2),V(20,HEIGHT/2),V(-20,HEIGHT/2)],color='#101010'))
def createSquares(w,numSquares=100,e=1):
    for i in range(0,numSquares):
        a=random.randrange(10, 70)
        mass=a**2/10
        w.addEntity(Polygon(w,w.c,e=e,m=mass,x=random.randrange(150, WIDTH - 150),y=random.randrange(150,HEIGHT - 150),rot=random.randrange(-3,3),v=V(random.randrange(-3,3),random.randrange(-30,30)),w=random.randrange(-3,3),color=generateRandomColor(),verteies=[V(-a,-a),V(a,-a),V(a,a),V(-a,a)]))
        for j in range(0,len(w.entityList) - 1):
            if w.checkCollision(w.entityList[-1],w.entityList[j]):
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
    world.start()
main()