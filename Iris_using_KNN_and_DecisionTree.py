from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def MarvellousDecision(data_train,data_test,target_train,target_test):
	
	cobj=tree.DecisionTreeClassifier()
	
	cobj.fit(data_train,target_train)
	
	output=cobj.predict(data_test)
	
	Accuracy=accuracy_score(target_test,output)
	
	return Accuracy
	
def MarvellousKNN(data_train,data_test,target_train,target_test):
	
	
	cobj=KNeighborsClassifier()
	
	cobj.fit(data_train,target_train)
	
	output=cobj.predict(data_test)
	
	Accuracy=accuracy_score(target_test,output)
	
	return Accuracy
	
	
def main():
	
	dataset=load_iris()
	
	#print(dataset)
	print(dataset.data)
	print(dataset.target)
	data=dataset.data
	target=dataset.target
	
	data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
	
	ret=MarvellousDecision(data_train,data_test,target_train,target_test)
	print("Accuracy Decision Tree is",ret*100,"%")
	
	ret=MarvellousKNN(data_train,data_test,target_train,target_test)
	print("Accuracy KNN is",ret*100,"%")
	
if __name__=="__main__":
	main()

#industrial : 	deployed ML we got more data we  created log file after a certain period of time log file is studdied by analysit then coder mmake changes in code and and code iss aagain deployeed	
#everytime new tree is created when we run program as it is not preserved
	
#donhi algorithm la jaanara data vegla ahe as split vegla vegla hotoy so  to gain accuracy on same data split in main and then send to both alfo


#accuracy depends on variety of training data given on algo and that does not depend on us it depends on train_test split

#and we not have to coompare algorithms
#random state vaparnya aaivaji main madhye split kara kasa te baghu 

#kNN is not clustering but it looks like clustering
#take k value ODD
#k value depends on dataset

#k=squareroot of no of dataset
#but dont follow this method start from 3
#if k=3 then 3 neighbours which are closed ie calcculaated from distance formula fromm grapph 
#are considered and then majority of entries of particular class iiss counted and that class is assigned to the entry which we want predict
#if k=5 then 5 nearest neighbours are considered by calculating distaance formula from graph and majority of class in that 5  
#testing chhya .......visarlo

#calulatte ecledian distance fill it in list sort the list ask for k and make sub array comapre it subarray and return 

#for aaccuracy prefer eucledian distance

#lazy learning mhanje question vicharlya var parat abhyas karun answer dena
#ani quick learning mhanje queation vicharlya var past experience var answer dena

