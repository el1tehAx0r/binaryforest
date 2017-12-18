class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.entropy=None
        self.left=None
        self.right=None
    def setEntropy(self,ent):
        self.entropy=ent
    def getEntropy(self):
        return self.entropy
    def getData(self):
        return self.data    
    def getLeft(self):
        return self.left
    def getRight(self):
        return self.right
    def setData(self,newdata):
        self.data = newdata
    def setLeft(self,newLeft):
        self.left = newLeft
    def setRight(self,newRight):
        self.right=newRight
