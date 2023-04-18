filename='covid-germ2.xlsx';
covid_data=xlsread(filename);
Y=covid_data(:,1);

%%%% düz multivariable regression

Y_train=Y(1:280);
Y_test= Y(length(Y_train)+1:end);

X=covid_data(:,2:30);

X_train=X(1:length(Y_train),:);
X_test=X(length(Y_train)+1:end,:);
 
model1=fitlm(X_train,Y_train)
 
Y_predict1= predict(model1,X_train);

figure
plot([Y_train,Y_predict1]);
 
MSE_mv_regression_train = immse(Y_train,Y_predict1); 
MAD_mv_regression_train=mean(abs(Y_train-Y_predict1));
MAPE_mv_regression_train= mean(abs(Y_train-Y_predict1)./Y_train);

%%%% mv test
Y_predict_test=predict(model1,X_test);

figure
plot([Y_test,Y_predict_test]);
 
MSE_mv_regression_test=immse(Y_test,Y_predict_test);
RMSE_mv_regression_test=sqrt(MSE_mv_regression_test);
MAD_mv_regression_test=mean(abs(Y_test-Y_predict_test));
MAPE_mv_regression_test= mean(abs(Y_test-Y_predict_test)./Y_test);
 
% %%%%%% elle dropladık dersek
 
X2=X(:,[1,2,4,6,7,8,9,12,14,19]);

X2_train=X2(1:length(Y_train),:);
X2_test=X2(length(Y_train)+1:end,:);

model2=fitlm(X2_train,Y_train)

Y_predict2= predict(model2,X2_train);

figure
plot([Y_train,Y_predict2]);

MSE_mv_regression_train2= immse(Y_train,Y_predict2); 
MAD_mv_regression_train2=mean(abs(Y_train-Y_predict2));
MAPE_mv_regression_train2= mean(abs(Y_train-Y_predict2)./Y_train);

Y_predict_test2=predict(model2,X2_test);

figure
plot([Y_test,Y_predict_test2]);

MSE_mv_regression_test2=immse(Y_test,Y_predict_test2);
RMSE_mv_regression_test2=sqrt(MSE_mv_regression_test2);
MAD_mv_regression_test2=mean(abs(Y_test-Y_predict_test2));
MAPE_mv_regression_test2= mean(abs(Y_test-Y_predict_test2)./Y_test);

%%%%%%%% lasso

[B,fitinfo]=lasso(X_train,Y_train)
MSE_lasso=fitinfo.MSE;
 
figure
plot(MSE_lasso);
 
% target=21; en iyi test mape veren 13.87%
target=80;
coeff=B(:,target);
 
%intercept=fitinfo.Intercept(target);
intercept=0;

Y_predict_train=X_train*coeff+intercept;
MSE_new_train=immse(Y_train,Y_predict_train);
RMSE_new_train=sqrt(MSE_new_train);
MAD_new_train = mean (abs (Y_train-Y_predict_train));
MAPE_new_train= mean(abs(Y_train-Y_predict_train)./Y_train);

figure
plot([Y_train,Y_predict_train]);

%%%% lasso test

Y_predict_test=X_test*coeff+intercept;
MSE_new_test=immse(Y_test,Y_predict_test);
RMSE_new_test=sqrt(MSE_new_test);
MAD_new_test = mean (abs (Y_test-Y_predict_test));
MAPE_new_test= mean(abs(Y_test-Y_predict_test)./Y_test);
 
figure
plot([Y_test,Y_predict_test]);

