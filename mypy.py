import sys
import time
import math
import random
from tkinter import *
#def import_pywin32():#this function fix the "Dll not loaded" problem, call this function before import win32api
#    """
#    For the module ``pywin32``,
#    this function tries to add the path to the DLL to ``PATH``
#    before throwing the exception: 
#    ``DLL load failed: The specified module could not be found``.
#    """
#    try:
#        import win32com
#    except ImportError as e :
#        if "DLL load failed:" in str(e):
#            import os,sys
#            path = os.path.join(os.path.split(sys.executable)[0], "Lib","site-packages","pywin32_system32")
#            os.environ["PATH"] = os.environ["PATH"] + ";" + path
#            try:
#                import win32com
#            except ImportError as ee :
#                dll = os.listdir(path)
#                dll = [os.path.join(path,_) for _ in dll if "dll" in _]
#                raise ImportError("some DLL must be copied:\n" + "\n".join(dll)) from e
#        else :
#            raise e
#import_pywin32()
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
    def rot(self,radian):#clockwise
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
    def combined(aabbList):
        minX=sys.maxsize
        minY=sys.maxsize
        maxX=-sys.maxsize
        maxY=-sys.maxsize
        for aabb in aabbList:
            if aabb.minX<minX:
                minX=aabb.minX
            if aabb.minY<minY:
                minY=aabb.minY
            if aabb.maxX>maxX:
                maxX=aabb.maxX
            if aabb.maxY>maxY:
                maxY=aabb.maxY
        return AABB(minX,maxX,minY,maxY)
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
        self.interactionList=[]
        self.aabb=AABB(0,0,0,0)
        self.world=world
        self.deleteMe=False
        self.shape=None
    def velocityAt(self,pos):
        relPos=pos-self.pos
        return self.v+V(-relPos.y,relPos.x)*self.w
    def posAt(self,relDis,relRot):
        return V(relDis,0).rot(relRot+self.rot)+self.pos
    def updateAABB(self):
        raise Exception("abstract method called")
    def update(self,dt):
        raise Exception("abstract method called")
    def calculateInertia(self):
        raise Exception("abstract method called")
    def draw(self):
        raise Exception("abstract method called")
    def setColor(self,color):
        self.canvas.itemconfig(self.shape, fill=color)
    def destoryAttachedInteractions(self):
        for interaction in self.interactionList:
            if interaction:
                interaction.destory()
    def destory(self):
        self.destoryAttachedInteractions()
        if self.shape:
            self.canvas.delete(self.shape)
        self.deleteMe=True
class CombinedEntity(Entity):
    def __init__(self,entities):
        self.entities=entities
        self.world=entities[0].world
        self.entityCount=len(self.entities)
        self.verteies=[None]*self.entityCount#entities position relative to center of mass
        self.verteiesRot=[None]*self.entityCount#entities default rot
        self.m=0
        self.rot=0
        self.dynamic=self.entities[0].dynamic
        self.deleteMe=False
        self.v=V(0,0)
        self.color=[None]*self.entityCount
        self.area=0
        for i in range(0,self.entityCount):
            self.m+=self.entities[i].m
            self.v+=self.entities[i].v*self.entities[i].m
            self.verteies[i]=self.entities[i].pos
            self.verteiesRot[i]=self.entities[i].rot
            self.color[i]=self.entities[i].color
            self.area+=self.entities[i].area
        self.v/=self.m
        self.rm=1/self.m
        self.density=self.m/self.area
        self.reallocateCenterOfMass()
        self.I=self.calculateInertia()
        self.rI=1/self.I
        self.w=0
        self.impuluses=[]
        self.interactionList=[]
        print("CombinedEntity created")
        print("area    = "+str(self.area))
        print("mass    = "+str(self.m))
        print("density = "+str(self.density))
        print("Inertia = "+str(self.I))
    def destory(self):
        self.destoryAttachedInteractions()
        for entity in self.entities:
            entity.destory()
        self.deleteMe=True
    def reallocateCenterOfMass(self):
        self.pos=V(0,0)
        for entity in self.entities:
            self.pos+=entity.m*entity.pos
        self.pos*=self.rm
        for i in range(0,self.entityCount):
            self.verteies[i]=self.entities[i].pos-self.pos
    def calculateInertia(self):
        I=0
        for i in range(0,self.entityCount):
            I+=self.entities[i].I+self.entities[i].m*self.verteies[i].mag_sq()
        return I
    def updateSubEntities(self,dt):
        for i in range(0,self.entityCount):
            self.entities[i].pos=self.verteies[i].rot(self.rot)+self.pos
            self.entities[i].rot=self.verteiesRot[i]+self.rot
            self.entities[i].update(dt)
    def updateAABB(self):
        self.aabb=AABB.combined([entity.aabb for entity in self.entities])
    def update(self, dt):
        while self.impuluses:
            impulus=self.impuluses.pop()
            self.v = self.v + impulus.vec * self.rm
            self.w = self.w + impulus.vec ** (self.pos-impulus.pos) * self.rI
        self.pos = self.pos + self.v * dt
        self.rot = self.rot + self.w * dt
        self.updateSubEntities(dt)
        self.updateAABB()
    def draw(self):
        for entity in self.entities:
            entity.draw()
    def setColor(self,color):
        if isinstance(color,list):
            for i in range(0,self.entityCount):
                self.entities[i].setColor(color[i])
        else:
            for entity in self.entities:
                entity.setColor(color)
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
            self.density=self.m/self.area
            self.I=self.calculateInertia()
            self.rI=1/self.I
            print("Ball created")
            print("area    = "+str(self.area))
            print("mass    = "+str(self.m))
            print("density = "+str(self.density))
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
    def draw(self):
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
            self.density=self.m/self.area
            self.I=self.calculateInertia()
            self.rI=1/self.I
            print("Polygon created")
            print("area    = "+str(self.area))
            print("mass    = "+str(self.m))
            print("density = "+str(self.density))
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
    def draw(self):
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
        print("Polygon.areVerteiesClockwise(verteies) not implemented")
        return True
class Interaction:
    def __init__(self,canvas=None,entityList=[]):#also push it self into Each entity's Interaction List
        self.entityCount=len(entityList)
        self.canvas=canvas
        self.entityList=entityList
        self.shape=None
        self.deleteMe=False
        for entity in entityList:
            entity.interactionList.append(self)
    def update(self,entity,dt):
        raise Exception("abstract method called")
    def applyInteractions(self,dt):
        raise Exception("abstract method called")
    def draw(self):
        raise Exception("abstract method called")
    def destory(self):
        if self.shape:
            self.canvas.delete(self.shape)
        self.deleteMe=True
class Spring(Interaction):
    def __init__(self,canvas,e1,e2,strength,length,relDis1=0,relDis2=0,relRot1=0,relRot2=0):
        Interaction.__init__(self,canvas,[e1,e2])
        self.strength,self.length,self.relDis1,self.relDis2,self.relRot1,self.relRot2=strength,length,relDis1,relDis2,relRot1,relRot2
        self.drawPos1x=e1.posAt(relDis1,relRot1).x
        self.drawPos1y=e1.posAt(relDis1,relRot1).y
        self.drawPos2x=e2.posAt(relDis2,relRot2).x
        self.drawPos2y=e2.posAt(relDis2,relRot2).x
        self.shape=self.canvas.create_line(self.drawPos1x,self.drawPos1y,self.drawPos2x,self.drawPos2y)
    def update(self,dt):
        e1=self.entityList[0]
        e2=self.entityList[1]
        sprintVertex1=e1.posAt(self.relDis1,self.relRot1)
        springVertex2=e2.posAt(self.relDis2,self.relRot2)
        self.drawPos1x=sprintVertex1.x
        self.drawPos1y=sprintVertex1.y
        self.drawPos2x=springVertex2.x
        self.drawPos2y=springVertex2.y
        springVector=springVertex2-sprintVertex1
        springVector_mag=springVector.mag()
        if springVector_mag:
            springVector_unit=springVector/springVector_mag
            dis=springVector_mag-self.length
            impulus=springVector_unit*dis*self.strength*dt
            e1.impuluses.append(Impulus(vec=impulus,pos=sprintVertex1))
            e2.impuluses.append(Impulus(vec=-impulus,pos=springVertex2))
        self.draw()
    def draw(self):
        self.canvas.coords(self.shape,self.drawPos1x,self.drawPos1y,self.drawPos2x,self.drawPos2y)
class Penetration:
    def __init__(self,e1,e2,normal,depth):
        self.e1=e1
        self.e2=e2
        self.normal=normal
        self.depth=depth
class CollisionInfo:
    def __init__(self,e1,e2,e,normal,penetration,collisionPos):
        self.e1,self.e2,self.e,self.normal,self.penetration,self.collisionPos=e1,e2,e,normal,penetration,collisionPos
class World:
    cursorMemorySize=10
    def __init__(self,width=800,height=600,gravaty=5,worldSpeedMultiplier=1,airResistence=0.1,color="#1E1E1E"):
        self.tk = Tk() #main window
        self.tk.title('')
        self.tk.wm_attributes('-fullscreen','true')
        self.width = width
        self.height = height
        self.color=color
        self.c = Canvas(self.tk,width=self.width,height=self.height,bg=color)
        self.c.pack()
        self.entityList = []
        self.interactionList = []
        self.penetrations = []
        if isinstance(gravaty,V):
            self.gravaty = gravaty
        else:
            self.gravaty = V(0,gravaty)
        self.g=self.gravaty.mag()
        self.worldSpeedMultiplier=worldSpeedMultiplier
        self.airResistence=airResistence
        self.timeOfLastFrame = None
        self.paused = False
        self.controlledEntity=None
        self.cursorPos=V(0,0)
        self.cursorVelocity=V(0,0)
        self.cursorPosHistory=[self.cursorPos]*World.cursorMemorySize
        self.cursorPosHistoryIndex=0
        self.updateCursorPos()
    def addEntity(self,entity):
        if entity:
            self.entityList.append(entity)
    def addCombinedEntity(self,index1,index2):
        subEntityList=[]
        if isinstance(self.entityList[index1],CombinedEntity):
            subEntityList+=self.entityList[index1].entities
        else:
            subEntityList.append(self.entityList[index1])
        if isinstance(self.entityList[index2],CombinedEntity):
            subEntityList+=self.entityList[index2].entities
        else:
            subEntityList.append(self.entityList[index2])
        self.addEntity(CombinedEntity(subEntityList))
        if index2>index1:
            index1,index2=index2,index1
        del self.entityList[index1]
        del self.entityList[index2]
    def addInteraction(self,interaction):
        if interaction:
            self.interactionList.append(interaction)
    def addRandomBallAtPos(self,pos,e=1,v=V(0,0)):
        self.addEntity(self.generateRandomBallAtPos(pos,e,v=v))
    def addRandomPolygonAtPos(self,pos,e=1,v=V(0,0)):
        self.addEntity(self.generateRandomPolygonAtPos(pos,e=e,v=v))
    def addRandomPenisAtPos(self,pos,e=1,v=V(0,0)):
        self.addEntity(self.generateRandomPenisAtPos(pos,e=e,v=v))
    def generateRandomBallAtPos(self,pos,e=1,v=V(0,0),density=0.1):
        radius=random.randrange(10, 100)
        mass=radius**2*math.pi*density
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
    def generateRandomPenisAtPos(self,pos,e=1,v=V(0,0),density=0.1):
        size=random.randint(20,150)
        width=size*1.5
        length=size*4
        mass=width*length*density
        rect=Polygon(self,self.c,m=mass,x=pos.x,y=pos.y,rot=0,dynamic=True,e=e,v=V(0,0),w=0,verteies=[V(0,0),V(width,0),V(width,length),V(0,length)],color=generateRandomColor())
        radius=size
        mass=size**2*math.pi*density
        leftBall= Ball(self,self.c,m=mass,x=pos.x-radius,y=pos.y+length/2,e=e,r=radius,color=generateRandomColor())
        rightBall=Ball(self,self.c,m=mass,x=pos.x+radius,y=pos.y+length/2,e=e,r=radius,color=generateRandomColor())
        penis=CombinedEntity([rect,leftBall,rightBall])
        penis.v=v
        penis.rot=random.uniform(-2,2)
        self.addEntity(penis)

    def IsPosOutOfBound(self,pos):
        return pos.x>self.width*2 or pos.y>self.height*2 or pos.x<-self.width or pos.y<-self.height
    def deleteOutOfBoundEntities(self):
        for e in self.entityList:
            if self.IsPosOutOfBound(e.pos):
                e.deleteMe=True
    def delEntityAt(self,index):
        if self.entityList[index]:
            self.entityList[index].destory()
            del self.entityList[index]
        else:
            raise Exception("index invalid")
    def delInteractionAt(self,index):
        if self.interactionList[index]:
            self.interactionList[index].destory()
            del self.interactionList[index]
        else:
            raise Exception("index invalid")
    def delLastEntity(self):
        if len(self.entityList):
            self.c.delete(self.entityList.pop().shape)
        else:
            raise Exception("entityList is empty")
    def start(self):
        self.timeOfLastFrame = time.clock()
        self.loop()
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
    def updateEntities(self,dt):#apply airResistence and gravaty, then call Entity.update(dt)
        for entity in self.entityList:
            if entity.dynamic:
                entity.v = entity.v*(1-self.airResistence*dt)
                entity.w = entity.w*(1-self.airResistence*dt)
                if self.gravaty:
                    entity.v = entity.v + self.gravaty*dt
                entity.update(dt)
                entity.draw()
    def updateInteractions(self,dt):
        for interaction in self.interactionList:
            interaction.update(dt)
    def solveCollision(self,dt):
        for i in range(0,len(self.entityList)):#check collision
            for j in range(i + 1,len(self.entityList)):
                if self.entityList[i].dynamic or self.entityList[j].dynamic:
                    collisionInfo=self.checkCollision(self.entityList[i],self.entityList[j])
                    if isinstance(collisionInfo,CollisionInfo) or isinstance(collisionInfo,list):
                        self.processCollisionInfo(collisionInfo)
        while self.penetrations:#un-penetrate
            self.unpenetrate(self.penetrations.pop())
    def getControlledledEntityIndex(self):
        if self.controlledEntity:
            for i in reversed(range(0,len(self.entityList))):
                if self.controlledEntity is self.entityList[i]:
                    return i
    def respondToGameControl(self,dt):
        if GetAsyncKeyState(16):#shift held
            if self.controlledEntity:
                if GetAsyncKeyState(1):#mouse left held
                    for i in reversed(range(0,len(self.entityList))):
                        if self.entityList[i] is not self.controlledEntity and self.checkContainPoint(self.entityList[i],self.cursorPos):
                            controlledEntityIndex=self.getControlledledEntityIndex()
                            self.addCombinedEntity(i,controlledEntityIndex)
                            self.controlledEntity=self.entityList[-1]
                            self.controlledEntity.setColor("#000000")
                            break
        else:
            if GetAsyncKeyState(71):#g held, remove entities
                for i in reversed(range(0,len(self.entityList))):
                    if self.checkContainPoint(self.entityList[i],self.cursorPos):
                        self.delEntityAt(i)

            if self.controlledEntity:
                if GetAsyncKeyState(104): #numpad 8 held, controlled entity acc up
                    self.controlledEntity.v = self.controlledEntity.v + 30 * V(0,-1) * dt
                if GetAsyncKeyState(102): #numpad 6 held, controlled entity acc right
                    self.controlledEntity.v = self.controlledEntity.v + 30 * V(1, 0) * dt
                if GetAsyncKeyState(101): #numpad 5 held, controlled entity acc down
                    self.controlledEntity.v = self.controlledEntity.v + 30 * V(0, 1) * dt
                if GetAsyncKeyState(100): #numpad 4 held, controlled entity acc left
                    self.controlledEntity.v = self.controlledEntity.v + 30 * V(-1,0) * dt
                if GetAsyncKeyState(105): #numpad 9 held, controlled entity acc clockwise spin
                    self.controlledEntity.w = self.controlledEntity.w + 0.3 * dt
                if GetAsyncKeyState(103): #numpad 7 held, controlled entity acc conter-clockwise spin
                    self.controlledEntity.w = self.controlledEntity.w - 0.3 * dt   
                if GetAsyncKeyState(96): #numpad 0 held, repel every other entity
                    for e in self.entityList:
                        if e.dynamic and e is not self.controlledEntity:
                            e.v = e.v - 0.1 * (self.controlledEntity.pos - e.pos) * dt
                if GetAsyncKeyState(97): #numpad 1 held, attract every other entity
                    for e in self.entityList:
                        if e.dynamic and e is not self.controlledEntity:
                            e.v = e.v + 0.1 * (self.controlledEntity.pos - e.pos) * dt
            if GetAsyncKeyState(65):#a held, acc left
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(-1,0) * dt
            if GetAsyncKeyState(83):#s held, acc down
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(0,1) * dt
            if GetAsyncKeyState(68):#d held, acc right
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(1,0) * dt
            if GetAsyncKeyState(87):#w held, acc up
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 10 * V(0,-1) * dt
            if GetAsyncKeyState(70) == -32767:#f pressed, velocity and angular velocity flip
                for e in self.entityList:
                    if e.dynamic:
                        e.v = -e.v
                        e.w = -e.w
            if GetAsyncKeyState(88):#x held, dec towards current velocity and angular velocity
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v * (1 - 0.4 * dt)
                        e.w = e.w * (1 - 0.4 * dt)
            if GetAsyncKeyState(81):#q held, acc towards current velocity and angular velocity
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v * (0.4 * dt + 1)
                        e.w = e.w * (0.4 * dt + 1)
            if GetAsyncKeyState(90):#z held, acc randomly
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 300 * V(random.uniform(-1,1),random.uniform(-1,1)) * dt
            if GetAsyncKeyState(82) == -32767:#r pressed, reverse gravity
                if self.gravaty == V(0,0):
                    self.gravaty = V(self.g,0)
                self.gravaty = self.gravaty.rot(math.pi/6)
            if GetAsyncKeyState(84) == -32767:#t pressed, enable/disable gravaty
                self.gravaty = V(0,self.g) if self.gravaty == V(0,0) else V(0,0)
            if GetAsyncKeyState(67) == -32767:#c pressed, add ball at cursor pos with cursor velocity
                self.addRandomBallAtPos(self.cursorPos,v=self.cursorVelocity)
            if GetAsyncKeyState(86) == -32767:#v pressed, add poly at cursor pos with cursor velocity
                self.addRandomPolygonAtPos(self.cursorPos,v=self.cursorVelocity)
            if GetAsyncKeyState(66) == -32767:#b pressed, add penis at cursor pos with cursor velocity
                self.addRandomPenisAtPos(self.cursorPos,v=self.cursorVelocity)
            if GetAsyncKeyState(1):#left-mouse down, attract entities
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v + 0.1 * (self.cursorPos - e.pos) * dt
            if GetAsyncKeyState(2):#right-mouse down, repel entities
                for e in self.entityList:
                    if e.dynamic:
                        e.v = e.v - 0.1 * (self.cursorPos - e.pos) * dt
            if GetAsyncKeyState(4):#mid-mouse down, select/de-select entity to be controlled
                foundEntity = False
                for i in reversed(range(0,len(self.entityList))):
                    if self.checkContainPoint(self.entityList[i],self.cursorPos):
                        if self.controlledEntity:
                            self.controlledEntity.setColor(self.controlledEntity.color)
                        self.controlledEntity = self.entityList[i]
                        self.controlledEntity.setColor("#000000")
                        foundEntity = True
                        break
                if not foundEntity:
                    if self.controlledEntity:
                        self.controlledEntity.setColor(self.controlledEntity.color)
                    self.controlledEntity = None
    def loop(self):
        while True:
            self.updateCursorPos()
            if GetAsyncKeyState(27)==-32767:#esc pressed, exit
                sys.exit()
            if GetAsyncKeyState(32)==-32767:#space-bar pressed, pause/unpause
                self.paused=not self.paused
                self.emptyCursorHistory()
            if self.paused:
                self.getDeltaTime()
                self.tk.update()
                continue
            dt = self.getDeltaTime()
            self.respondToGameControl(dt)
            self.deleteOutOfBoundEntities()
            self.removeDeletedEntities()
            self.removeDeletedInteractions()
            self.updateInteractions(dt)
            self.updateEntities(dt)
            self.solveCollision(dt)
            self.tk.update()
    def removeDeletedEntities(self):
        for i in reversed(range(len(self.entityList))):
            if self.entityList[i].deleteMe:
                self.delEntityAt(i)
    def removeDeletedInteractions(self):
        for i in reversed(range(len(self.interactionList))):
            if self.interactionList[i].deleteMe:
                self.delInteractionAt(i)
    def unpenetrate(self,penetration):#e1 will move towards normal
        e1=penetration.e1
        e2=penetration.e2
        correction = penetration.depth / (e1.rm + e2.rm) * penetration.normal *1
        e1.pos += e1.rm * correction
        e2.pos -= e2.rm * correction
    def checkCollision_ball_ball(self,b1,b2):
        dis = b1.pos - b2.pos
        if (b1.r + b2.r)**2 < dis.mag_sq():
            return False
        else:
            normal=dis.unitVector()
            return CollisionInfo(b1,b2,min(b1.e,b2.e),normal,Penetration(b1,b2,normal,b1.r+b2.r-dis.mag()),b1.pos+normal*b1.r)
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
        penetration=Penetration(b1,p1,minPenetrationNormal,minPenetrationDepth)
        return CollisionInfo(b1,p1,min(p1.e,b1.e),minPenetrationNormal,penetration,collisionPos)
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
        penetration=Penetration(p1,p2,minPenetrationNormal,minPenetrationDepth)
        return CollisionInfo(p1,p2,min(p1.e,p1.e),minPenetrationNormal,penetration,collisionPos)
    def checkCollision_combined_combined(self,c1,c2):
        atLeastOneCollision=False
        collisionInfoList=[]
        for e1 in c1.entities:
            for e2 in c2.entities:
                collisionInfo=self.checkCollision(e1,e2)
                if isinstance(collisionInfo,CollisionInfo):
                    if e1 is collisionInfo.penetration.e1:
                        collisionInfo.penetration.e1=c1
                        collisionInfo.penetration.e2=c2
                        collisionInfo.e1=c1
                        collisionInfo.e2=c2
                    else:
                        collisionInfo.penetration.e1=c2
                        collisionInfo.penetration.e2=c1
                        collisionInfo.e1=c2
                        collisionInfo.e2=c1
                    collisionInfoList.append(collisionInfo)
                elif collisionInfo:
                    atLeastOneCollision=True
        if len(collisionInfoList):
            return collisionInfoList
        else:
            return atLeastOneCollision
    def checkCollision_combined_single(self,c,e):
        atLeastOneCollision=False
        collisionInfoList=[]
        for entity in c.entities:
            collisionInfo=self.checkCollision(entity,e)
            if isinstance(collisionInfo,CollisionInfo):
                if entity is collisionInfo.penetration.e1:
                    collisionInfo.penetration.e1=c
                    collisionInfo.e1=c
                else:
                    collisionInfo.penetration.e2=c
                    collisionInfo.e2=c
                collisionInfoList.append(collisionInfo)
            elif collisionInfo:
                atLeastOneCollision=True
        if len(collisionInfoList):
            return collisionInfoList
        else:
            return atLeastOneCollision
    def checkCollision_ball_V(self,b,v):
        return (b.pos-v).mag_sq()<b.r2
    def checkCollision_poly_V(self,p,v):
        areaSum=0
        for i in range(0,p.verteiesCount):
            v1=p.verteies_real[i]
            v2=p.verteies_real[(i+1)%p.verteiesCount]
            areaSum+=V.CalculateTriangleArea(v1,v2,v)
        if areaSum>p.area*1.01:
            return False
        return True
    def checkCollision_combined_V(self,c,v):
        for entity in c.entities:
            if self.checkContainPoint(entity,v):
                return True
        return False
    def checkCollision_poly_V_by_verteies(vert,v,area):
        areaSum=0
        for i in range(0,len(vert)):
            v1=vert[i]
            v2=vert[(i+1)%len(vert)]
            areaSum+=V.CalculateTriangleArea(v1,v2,v)
        if areaSum>area:
            return False
        return True
    def checkContainPoint(self,e,v):
        if isinstance(e,Ball):
            return self.checkCollision_ball_V(e,v)
        if isinstance(e,Polygon):
            return self.checkCollision_poly_V(e,v)
        if isinstance(e,CombinedEntity):
            return self.checkCollision_combined_V(e,v)
    def checkCollision(self,e1,e2):
        if AABB.noIntersect(e1.aabb,e2.aabb):
            return False
        if isinstance(e1,CombinedEntity):
            if isinstance(e2,CombinedEntity):
                return self.checkCollision_combined_combined(e1,e2)
            else:
                return self.checkCollision_combined_single(e1,e2)
        if isinstance(e2,CombinedEntity):
            if isinstance(e1,CombinedEntity):
                return self.checkCollision_combined_combined(e1,e2)
            else:
                return self.checkCollision_combined_single(e2,e1)
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
    def processCollisionInfo(self,collisionInfo):
        if isinstance(collisionInfo,list):
            for CI in collisionInfo:
                self.processCollisionInfo(CI)
        else:
            self.penetrations.append(collisionInfo.penetration)
            r1=collisionInfo.collisionPos-collisionInfo.e2.pos
            r2=collisionInfo.collisionPos-collisionInfo.e1.pos
            relVel=collisionInfo.e2.velocityAt(collisionInfo.collisionPos)-collisionInfo.e1.velocityAt(collisionInfo.collisionPos)
            if relVel*collisionInfo.normal>0:
                impulus=(relVel*collisionInfo.normal*(1+collisionInfo.e)/(collisionInfo.e2.rm+collisionInfo.e1.rm+(((r1**collisionInfo.normal)**r1)*collisionInfo.e2.rI+((r2**collisionInfo.normal)**r2)*collisionInfo.e1.rI)*collisionInfo.normal))*collisionInfo.normal       
                collisionInfo.e1.impuluses.append(Impulus(vec=impulus,pos=collisionInfo.collisionPos))
                collisionInfo.e2.impuluses.append(Impulus(vec=-impulus,pos=collisionInfo.collisionPos))
def generateRandomColor(min=150,max=255):
    r=random.randint(min,max)
    g=random.randint(min,max)
    b=random.randint(min,max)
    if (r-g)**2+(g-b)**2+(b-r)**2*40<(max-min)**2:
        return generateRandomColor(min,max)
    return '#%02x%02x%02x' % tuple([r,g,b])
def createBalls(w,numBalls=100,e=1):
    screenWidth,screenHeight=w.width,w.height
    for i in range(0,numBalls):
        radius=random.randrange(10, 70)
        mass=radius**2*math.pi/10
        w.addEntity(Ball(w,w.c,m=mass,x=random.randrange(150, screenWidth - 150),y=random.randrange(150,screenHeight - 150),v=V(random.randrange(-3,3),random.randrange(-30,30)),r=radius,e=e,color=generateRandomColor()))
def createContainerBalls(w,screenWidth,screenHeight,color='#303030'):
    numberOfBalls=10
    radius=screenWidth / numberOfBalls/2
    for i in range(numberOfBalls+1):
        w.addEntity(Ball(w,w.c,m=0,x=screenWidth / numberOfBalls * i,y=0,v=V(0,0),r=radius,dynamic=FALSE,color=color))
        w.addEntity(Ball(w,w.c,m=0,x=screenWidth / numberOfBalls * i,y=screenHeight,v=V(0,0),r=radius,dynamic=FALSE,color=color))
    numberOfBalls=5
    radius=screenHeight / numberOfBalls/2
    for i in range(numberOfBalls+1):
        w.addEntity(Ball(w,w.c,m=0,x=0,y=screenHeight / numberOfBalls * i,v=V(0,0),r=radius,dynamic=FALSE,color=color))
        w.addEntity(Ball(w,w.c,m=0,x=screenWidth,y=screenHeight / numberOfBalls * i,v=V(0,0),r=radius,dynamic=FALSE,color=color))
def createContainerRects(w,screenWidth,screenHeight,color='#303030'):
    thickness=1000
    w.addEntity(Polygon(w,w.c,m=100,x=screenWidth/2,y=screenHeight+thickness,v=V(0,0),dynamic=False,verteies=[V(-screenWidth/2,-thickness),V(screenWidth/2,-thickness),V(screenWidth/2,thickness),V(-screenWidth/2,thickness)],color=color))#down
    w.addEntity(Polygon(w,w.c,m=100,x=screenWidth/2,y=-thickness,v=V(0,0),dynamic=False,verteies=[V(-screenWidth/2,-thickness),V(screenWidth/2,-thickness),V(screenWidth/2,thickness),V(-screenWidth/2,thickness)],color=color))#up
    w.addEntity(Polygon(w,w.c,m=100,x=-thickness,y=screenHeight/2,v=V(0,0),dynamic=False,verteies=[V(-thickness,-screenHeight/2),V(thickness,-screenHeight/2),V(thickness,screenHeight/2),V(-thickness,screenHeight/2)],color=color))#left
    w.addEntity(Polygon(w,w.c,m=100,x=screenWidth+thickness,y=screenHeight/2,v=V(0,0),dynamic=False,verteies=[V(-thickness,-screenHeight/2),V(thickness,-screenHeight/2),V(thickness,screenHeight/2),V(-thickness,screenHeight/2)],color=color))#right
def createSquares(w,numSquares=100,e=1):
    screenWidth,screenHeight=w.width,w.height
    for i in range(0,numSquares):
        a=random.randrange(10, 70)
        mass=a**2/10
        w.addEntity(Polygon(w,w.c,e=e,m=mass,x=random.randrange(150, screenWidth - 150),y=random.randrange(150,screenHeight - 150),rot=random.randrange(-3,3),v=V(random.randrange(-3,3),random.randrange(-30,30)),w=random.randrange(-3,3),color=generateRandomColor(),verteies=[V(-a,-a),V(a,-a),V(a,a),V(-a,a)]))
def createWeb(world,entitySize,webSize,strength):
    polygons=[None]*webSize
    for i in range(webSize):
        polygons[i]=[None]*webSize
        for j in range(webSize):
            polygons[i][j]=Polygon(world,world.c,m=10,x=100*(i+1),y=100*(j+1),rot=0,v=V(0,0),w=0,verteies=[V(-entitySize,-entitySize),V(entitySize,-entitySize),V(entitySize,entitySize),V(-entitySize,entitySize)])
            world.addEntity(polygons[i][j])
            if j!=0:
                world.addInteraction(Spring(world.c,polygons[i][j],polygons[i][j-1],strength,100,0,0,math.pi/4,math.pi/4))
            if i!=0:
                world.addInteraction(Spring(world.c,polygons[i][j],polygons[i-1][j],strength,100,0,0,math.pi/4,math.pi/4))
def createPenis(world,e=1):
    world.addRandomPenisAtPos(pos=V(random.randint(100,world.width-100),random.randint(100,world.height-100)))
def main():
    HEIGHT = 1080
    WIDTH = 1920
    PI=math.pi
    world = World(WIDTH,HEIGHT,gravaty=300,worldSpeedMultiplier=1,airResistence=0.1)
    #world.gravaty=V(0,0)
    createSquares(world,5)
    createBalls(world,5)
    #create(world,entitySize=5,webSize=5,strength=500)
    createPenis(world)


    #size=30
    #polygon0=Polygon(world,world.c,m=10,x=500,y=150,rot=0,v=V(0,0),w=0,verteies=[V(-size,-size),V(size,-size),V(size,size),V(-size,size)],color=generateRandomColor())
    #size=100
    #polygon1=Polygon(world,world.c,m=10000,x=500,y=300,rot=0,v=V(0,0),w=0,verteies=[V(-size,-size),V(size,-size),V(size,size),V(-size,size)])
    #combinedEntity=CombinedEntity([polygon0,polygon1])
    #world.addEntity(combinedEntity)
    #combinedEntity.w=1


    #size=30
    #polygon0=Polygon(world,world.c,m=10,x=500,y=150,rot=1,v=V(0,0),w=0,verteies=[V(-size,-size),V(size,-size),V(size,size),V(-size,size)],color=generateRandomColor())
    #size=100
    #polygon1=Polygon(world,world.c,m=10000,x=500,y=300,rot=0,v=V(0,0),w=0,verteies=[V(-size,-size),V(size,-size),V(size,size),V(-size,size)])
    #size=30
    #ball1=Ball(world,world.c,m=PI*size**2,x=500,y=450,rot=0,v=V(0,0),w=0,r=size)
    #combinedEntity=CombinedEntity([polygon0,polygon1,ball1])
    #world.addEntity(combinedEntity)
    #combinedEntity.w=1

    #entitySize=100
    #polygon0=Polygon(world,world.c,m=4*entitySize**2,x=500,y=150,rot=0,v=V(0,0),w=0,verteies=[V(-entitySize,-entitySize),V(entitySize,-entitySize),V(entitySize,entitySize),V(-entitySize,entitySize)],color=generateRandomColor())
    #combinedEntity=CombinedEntity([polygon0])
    #world.addEntity(combinedEntity)
    #world.addInteraction(Spring(world.c,polygon0,polygon1,50,300,0,0,PI/4,PI/4))
    
    createContainerBalls(world,WIDTH,HEIGHT)
    #createContainerRects(world,WIDTH,HEIGHT)
    world.start()
main()