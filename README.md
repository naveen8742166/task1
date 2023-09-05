# task1
bharat intern task 1 based on my topic house price prediction
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
df=pd.read_csv('C:\\Users\\P.Shanmukha\\Desktop\\houseprice.csv')
df.shape
(5000, 7)
df.sample(10)
Avg. Area Income	Avg. Area House Age	Avg. Area Number of Rooms	Avg. Area Number of Bedrooms	Area Population	Price	Address
1806	66426.893441	6.315633	6.737996	4.48	40761.351310	1.168445e+06	7065 Brown Cliffs\nEast Sydneybury, PW 07220
899	67335.757730	6.785460	7.377845	6.26	10311.001394	8.632972e+05	493 Beth Tunnel Apt. 276\nNew Mitchell, DC 10981
1358	71227.388815	6.996436	8.006614	4.21	20096.291417	1.455692e+06	10454 Gonzales Summit Apt. 065\nChambersstad, ...
3877	63477.257696	7.581828	7.023403	5.31	28894.497938	1.204401e+06	004 Jessica Rapid Apt. 549\nChelseyshire, MI 3...
1446	71301.007068	7.281558	8.452275	4.03	44869.506948	1.920528e+06	1039 Douglas Creek\nLake Sarah, RI 65939-0004
4345	69657.362791	6.352332	5.368804	3.08	29973.093565	8.842272e+05	44132 Brittany Forks Suite 881\nDavisview, IA ...
1699	62902.255492	6.577966	6.711163	3.40	27321.032931	9.846723e+05	18878 Harrison Mission Suite 409\nSouth Matthe...
3496	59453.318301	5.515337	6.271345	2.18	26601.921498	5.617038e+05	3006 Wheeler Roads\nSharonshire, HI 92850
947	69711.903933	6.073168	8.090747	3.45	36291.145204	1.433615e+06	29718 Simmons Shores Apt. 119\nAndreafurt, IN ...
2086	78583.264722	5.278237	7.428635	3.07	35271.719841	1.384328e+06	22992 Gonzales Crossroad Suite 385\nPort Jacob...
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 7 columns):
 #   Column                        Non-Null Count  Dtype  
---  ------                        --------------  -----  
 0   Avg. Area Income              5000 non-null   float64
 1   Avg. Area House Age           5000 non-null   float64
 2   Avg. Area Number of Rooms     5000 non-null   float64
 3   Avg. Area Number of Bedrooms  5000 non-null   float64
 4   Area Population               5000 non-null   float64
 5   Price                         5000 non-null   float64
 6   Address                       5000 non-null   object 
dtypes: float64(6), object(1)
memory usage: 273.6+ KB
df.describe()
Avg. Area Income	Avg. Area House Age	Avg. Area Number of Rooms	Avg. Area Number of Bedrooms	Area Population	Price
count	5000.000000	5000.000000	5000.000000	5000.000000	5000.000000	5.000000e+03
mean	68583.108984	5.977222	6.987792	3.981330	36163.516039	1.232073e+06
std	10657.991214	0.991456	1.005833	1.234137	9925.650114	3.531176e+05
min	17796.631190	2.644304	3.236194	2.000000	172.610686	1.593866e+04
25%	61480.562388	5.322283	6.299250	3.140000	29403.928702	9.975771e+05
50%	68804.286404	5.970429	7.002902	4.050000	36199.406689	1.232669e+06
75%	75783.338666	6.650808	7.665871	4.490000	42861.290769	1.471210e+06
max	107701.748378	9.519088	10.759588	6.500000	69621.713378	2.469066e+06
df.columns
Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
      dtype='object')
df.isnull().values.any()
False
numerical_features = [feature for feature in df.columns if df[feature].dtype!='O']
print(numerical_features)
print('No.of numerical features: ',len(numerical_features))
df[numerical_features].head()
['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population', 'Price']
No.of numerical features:  6
Avg. Area Income	Avg. Area House Age	Avg. Area Number of Rooms	Avg. Area Number of Bedrooms	Area Population	Price
0	79545.458574	5.682861	7.009188	4.09	23086.800503	1.059034e+06
1	79248.642455	6.002900	6.730821	3.09	40173.072174	1.505891e+06
2	61287.067179	5.865890	8.512727	5.13	36882.159400	1.058988e+06
3	63345.240046	7.188236	5.586729	3.26	34310.242831	1.260617e+06
4	59982.197226	5.040555	7.839388	4.23	26354.109472	6.309435e+05
sns.pairplot(df)
<seaborn.axisgrid.PairGrid at 0x22fb3dece50>

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
<Axes: >

X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
Y=df['Price']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=101)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
((3000, 5), (2000, 5), (3000,), (2000,))
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
print(lr.intercept_)
-2640159.7968519107
coeff_df=pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
coeff_df
Coefficient
Avg. Area Income	21.528276
Avg. Area House Age	164883.282027
Avg. Area Number of Rooms	122368.678027
Avg. Area Number of Bedrooms	2233.801864
Area Population	15.150420
predictions=lr.predict(X_test)
plt.scatter(Y_test,predictions,c='green')
<matplotlib.collections.PathCollection at 0x22fb85af070>

sns.histplot((Y_test-predictions),bins=50,kde=True)
<Axes: xlabel='Price', ylabel='Count'>

from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(Y_test,predictions))
MAE: 82288.22251914955
print('MSE:',metrics.mean_squared_error(Y_test,predictions))
MSE: 10460958907.209503
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
RMSE: 102278.82922291153
