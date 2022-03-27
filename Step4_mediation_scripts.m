[num,txt,mplus]=xlsread('C:\Users\92175\Desktop\CPVN\41subFigu\mediationFigure\41subMediation_Input_Final.xls','mplus4');

%neg-feq→vmPFC-hippo→child anxiety
X_data=num(:,9);
M_data=num(:,10);
Y_data=num(:,7);%internal 


cov_all_data=[num(:,3:6)]; %cov: child age, child gender, parent gender,site

b = regress(X_data,[cov_all_data ones(size(cov_all_data,1),1)]);
X_data_cov = X_data - (b(1:end-1)'*cov_all_data')';

b = regress(M_data,[cov_all_data ones(size(cov_all_data,1),1)]);
M_data_cov = M_data - (b(1:end-1)'*cov_all_data')';

b = regress(Y_data,[cov_all_data ones(size(cov_all_data,1),1)]);
Y_data_cov = Y_data - (b(1:end-1)'*cov_all_data')';


[paths, stats] = mediation(X_data_cov, Y_data_cov, M_data_cov, 'plots', 'verbose', 'boot', 'bootsamples', 10000);


