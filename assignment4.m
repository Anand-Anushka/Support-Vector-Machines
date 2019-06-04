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

%Mdl = fitcecoc(B,C)
%model = svmtrain(B,C);