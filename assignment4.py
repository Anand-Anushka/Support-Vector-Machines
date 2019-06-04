import csv
from sklearn import svm
import matplotlib.pyplot as plt
import numpy

'''Due to problems faced in extracting .mat files in python, the files were 
extracted in matlab and stored in a .csv file using the following code

******* MATLAB CODE TO EXTRACT .MAT FILES *********
S1=load ('Pattern1.mat');
S2=load ('Pattern2.mat');
S3=load ('Pattern3.mat');

T1=load ('Test1.mat');
T2=load ('Test2.mat');
T3=load ('Test3.mat');

%A=cell2mat(S1.train_pattern_1);
B=zeros(600,120);
C=zeros(600,1);
T=zeros(300,120);
Y=zeros(300,1);


for i = 1:200
    B(i,:)=S1.train_pattern_1{i};
    B(i+200,:)=S2.train_pattern_2{i};
    B(i+400,:)=S3.train_pattern_3{i};
    
    C(i)=1;
    C(i+200)=2;
    C(i+400)=3;
end

for i = 1:100
    T(i,:)=T1.test_pattern_1{i};
    T(i+100,:)=T2.test_pattern_2{i};
    T(i+200,:)=T3.test_pattern_3{i};
    
    Y(i)=1;
    Y(i+100)=2;
    Y(i+200)=3;
 
end
csvwrite('B',B);
csvwrite('C',C);
csvwrite('T',T);
csvwrite('Y',Y);

******* MATLAB CODE TO EXTRACT .MAT FILES *********

'''

with open('B', 'rb') as f:
    reader = csv.reader(f)
    B = map(tuple, reader)
with open('C', 'rb') as f:
    reader = csv.reader(f)
    C = map(tuple, reader)
    C=numpy.ravel(C)
with open('T', 'rb') as f:
    reader = csv.reader(f)
    T = map(tuple, reader)
with open('Y', 'rb') as f:
    reader = csv.reader(f)
    Y = map(tuple, reader)
    Y=numpy.ravel(Y)

s=[]
t=[]
r1=numpy.arange(1,100000,100)
r2=numpy.arange(0.001,3,0.01)
for i in r1:
    svc = svm.SVC(kernel='rbf', C=i,gamma=0.03)
    svc.fit(B,C)#training
    s.append(svc.score(T,Y))#testing

for i in r2:
    svc = svm.SVC(kernel='rbf', C=1000,gamma=i)
    svc.fit(B,C)#training
    t.append(svc.score(T,Y))#testing
    
max1=max(s)   
max2=max(t)

maxt=max(max1,max2)
print('The maximum accuracy that was obtained on the test data: ')
print(maxt)
    
fig = plt.figure()
plt.subplot(222)
plt.semilogx(r1, s)

plt.grid(True)
plt.savefig('plot_varying_C')
plt.subplot(222)
plt.ylabel('Accuracy of model')
plt.xlabel('Value of C used')  
plt.title('Plot showing Variation of Accuracy with parameter C') 
plt.grid(True)
plt.show()
fig = plt.figure()

plt.semilogx(r2, t)
plt.grid(True)
plt.savefig('plot_varying_v')
plt.ylabel('Accuracy of model')
plt.xlabel('Value of gamma')  
plt.title('Plot showing Variation of Accuracy with parameter gamma') 
plt.grid(True)
plt.show()
