import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy import spatial
from scipy import signal
from scipy import misc



#Import files
file=pd.read_csv('listings.csv',dtype={'weekly_price':str,'monthly_price':str,'license':str,'jurisdiction_names':str})
house=pd.read_csv('UK Average House Price.csv')

#Drop irrelevant files
airbnb=file.loc[0:,['id', 'name','neighbourhood_cleansed', 'city', 'zipcode', 'smart_location', 'property_type',
                    'room_type', 'accommodates', 'price','weekly_price', 'monthly_price', 'security_deposit',
                    'cleaning_fee', 'extra_people','first_review','last_review']]
airbnb=airbnb.dropna(subset=['last_review'])

#Cleaning airbnb dataset
#Change the dtype
airbnb[['price','weekly_price','monthly_price','security_deposit','cleaning_fee',
        'extra_people']]=airbnb[['price','weekly_price','monthly_price','security_deposit','cleaning_fee',
                                 'extra_people']].replace('[\$,]','',regex=True).astype(float)

#Fill the missing price values
airbnb['price']=airbnb['price'].fillna(airbnb['price'].mean())
airbnb['cleaning_fee']=airbnb['cleaning_fee'].fillna(airbnb['cleaning_fee'].mean())

#Convert dollar to pound
airbnb[['price','weekly_price','monthly_price']]=airbnb[['price','weekly_price','monthly_price']]*0.78

#Find the average days of weekly and monthly price columns
weekly_avg=(airbnb['weekly_price']/airbnb['price']).mean()
monthly_avg=(airbnb['monthly_price']/airbnb['price']).mean()

#Fill in the missing values with the average days
airbnb['weekly_price']=airbnb['weekly_price'].fillna(airbnb['price']*weekly_avg)
airbnb['monthly_price']=airbnb['monthly_price'].fillna(airbnb['price']*monthly_avg)

#Create new columns of average 7 days a week, 30 days a month and 365 days a year
weekly=airbnb['price']*7
airbnb.insert(11,'weekly',weekly)

monthly=airbnb['price']*30
airbnb.insert(13,'monthly',monthly)

Price_airbnb=airbnb['price']*365
airbnb.insert(14,'Price_airbnb',Price_airbnb)

#Create new columns where total price includes other fees
total_price=airbnb['Price_airbnb']+airbnb['cleaning_fee']
airbnb.insert(17,'Total_airbnb',total_price)

#Change the format of date
airbnb['first_review']=pd.to_datetime(airbnb['first_review']).dt.strftime('%d/%m/%y')
airbnb['first_review']=pd.to_datetime(airbnb['first_review'])

airbnb['last_review']=pd.to_datetime(airbnb['last_review']).dt.strftime('%d/%m/%y')
airbnb['last_review']=pd.to_datetime(airbnb['last_review'])
airbnb=airbnb.sort_values(by=['last_review'],ascending=[True])

airbnb['property_type'].value_counts()
airbnb['neighbourhood_cleansed'].value_counts()

#Plot Number of Airbnb Listings per London Borough
plt.hist(airbnb['neighbourhood_cleansed'],bins=30,rwidth=0.8,color='black',align='left')
plt.title('Number of Airbnb Listings per London Borough')
plt.xlabel('Borough of London')
plt.ylabel('Number of Airbnb Listings')
plt.xticks(rotation=65,horizontalalignment='right')
plt.tight_layout()
plt.show()

#Plot Number of Property Types per London Borough
plt.hist(airbnb['property_type'],bins=30,rwidth=0.8,color='black',align='left',log=True)
plt.title('Number of Property Types per London Borough')
plt.xticks(rotation=65,horizontalalignment='right')
plt.tight_layout()
plt.show()

#Condence airbnb dataset further
airbnb1=airbnb

#Obtain the year from the date column
airbnb1['Year']=airbnb1['last_review'].dt.year

#Rename the coluns
airbnb1=airbnb1.rename(columns={'neighbourhood_cleansed':'Area'})

#Group the area and year columns and reset idex
airbnb1=airbnb1.groupby(['Area','Year']).mean().reset_index()

#Drop further irrelevant columns
airbnb1=airbnb1.drop(columns=['id','accommodates','weekly_price','weekly','monthly_price','monthly',
                              'security_deposit','cleaning_fee','extra_people'])

#Final cleansed airbnb dataset
airbnb1=airbnb1[['Price_airbnb','Total_airbnb','Area','Year']]

airbnb1.to_csv('test_airbnb.csv')

#Plot Number of Airbnb Listings per Borough throughout the Year
sns.lineplot(x=airbnb1['Area'],y=airbnb1['Year'])
plt.xticks(rotation=65,horizontalalignment='right')
plt.title('Number of Airbnb Listings per Borough throughout the Year')
plt.tight_layout()
plt.show()

#Cleaning house dataset
#Drop irrelevant columns
house=house.drop(axis=0,index=0)
house=house.loc[:,:'Westminster']

#Change the format of date
house['Date']=pd.to_datetime(house['Date'])

#Change the dataset from wide to long format
house=house.melt(id_vars=['Date'],var_name='Area',value_name='Price')

#Obtain the year from the date column
house['Year']=house['Date'].dt.year

#Rename columns
house=house.rename(columns={'Price':'Price_house'})
house['Price_house']=house['Price_house'].replace(',','',regex=True).astype(int)

#Group area and year columns and reset index
house=house.groupby(['Area','Year']).mean().reset_index()

#Final cleansed house dataset
house=house[['Year','Area','Price_house']]

house=house.replace('Barking & Dagenham','Barking and Dagenham')
house=house.replace('Kensington & Chelsea','Kensington and Chelsea')
house=house.replace('Hammersmith & Fulham','Hammersmith and Fulham')

#Merge two datasets on year
house=house[house.Year>=2011]
merged=pd.merge(house,airbnb1,how='outer',on=['Year','Area'])

merged['Price_airbnb']=merged['Price_airbnb'].fillna(merged['Price_airbnb'].mean())
merged['Total_airbnb']=merged['Total_airbnb'].fillna(merged['Total_airbnb'].mean())


#Plot Average Airbnb Listing Price per Borough
sns.barplot(x=merged['Area'],y=merged['Price_house'],palette='vlag')
plt.xticks(rotation=65,horizontalalignment='right')
plt.title('Average Airbnb Listing Price per Borough')
plt.tight_layout()
plt.show()

#Plot Average London Housing Price per Borough
sns.barplot(x=merged['Area'],y=merged['Price_airbnb'],palette='rocket')
plt.xticks(rotation=65,horizontalalignment='right')
plt.title('Average London Housing Price per Borough')
plt.tight_layout()
plt.show()

#Plot Total airbnb price vs House price
plt.scatter(merged['Total_airbnb'],merged['Price_house'],alpha=0.4)
plt.xlabel('Total Airbnb Price')
plt.ylabel('House Price')
plt.title('Total airbnb price vs House price')
plt.show()

#Plot Airbnb price vs House price
plt.scatter(merged['Price_airbnb'],merged['Price_house'],alpha=0.4)
plt.xlabel('Airbnb Price')
plt.ylabel('House Price')
plt.title('Airbnb price vs House price')
plt.show()

#identify and plot the outliers with mahalanobis distances
column_values=merged[['Total_airbnb','Price_house']].values
mean_vector=np.asarray([merged['Total_airbnb'].mean(),merged['Price_house'].mean()]).reshape((1,2))
mahalanobis_distances=spatial.distance.cdist(column_values,mean_vector,'mahalanobis')[:,0]
plt.scatter(merged['Total_airbnb'],merged['Price_house'],c=mahalanobis_distances,cmap=plt.cm.Blues)
plt.xlabel('Total Airbnb Price')
plt.ylabel('House Price')
plt.title('Airbnb Price vs House Price')
plt.show()


column_values1=merged[['Price_airbnb','Price_house']].values
mean_vector1=np.asarray([merged['Price_airbnb'].mean(),merged['Price_house'].mean()]).reshape((1,2))
mahalanobis_distances=spatial.distance.cdist(column_values,mean_vector,'mahalanobis')[:,0]
plt.scatter(merged['Price_airbnb'],merged['Price_house'],c=mahalanobis_distances,cmap=plt.cm.Greens)
plt.xlabel('Airbnb Price')
plt.ylabel('House Price')
plt.title('Airbnb Price vs House Price')
plt.show()

sns.residplot(merged['Price_airbnb'],merged['Price_house'],lowess=True,color='g')
plt.show()
#therefore, a non-linear regression model might be more approriate.

sns.residplot(merged['Total_airbnb'],merged['Price_house'],lowess=True,color='g')
plt.show()
#therefore, a non-linear regression model might be more approriate.

#Pearson correlation analysis
pearson_corr,p_val_pearson=stats.pearsonr(merged['Total_airbnb'],merged['Price_house'])
print('Pearson correlation between Total airbnb and Price house is: ',pearson_corr,' with p value of: ',p_val_pearson)

pearson_corr1,p_val_pearson1=stats.pearsonr(merged['Price_airbnb'],merged['Price_house'])
print('Pearson correlation between Price airbnb and Price house is: ',pearson_corr1,' with p value of: ',p_val_pearson1)

if pearson_corr<=0.29 and pearson_corr1<=0.29:
    print('The degree of Pearson correlation is low.')
elif 0.30<=pearson_corr<=0.49 and 0.30<=pearson_corr<=0.49:
    print('The degree of Pearson correlation is medium.')
elif 0.50<=pearson_corr<=1 and 0.50<=pearson_corr<=1:
    print('The degree of Pearson correlation is high.')
else:
    print('There is no Pearson correlation')

#Spearman correlation analysis
spearman_corr,p_val_spearman=stats.spearmanr(merged['Total_airbnb'],merged['Price_house'])
print('Spearman correlation between Total airbnb and Price house is: ',spearman_corr,' with p value of: ',p_val_spearman)

spearman_corr1,p_val_spearman1=stats.spearmanr(merged['Price_airbnb'],merged['Price_house'])
print('Spearman correlation between Price airbnb and Price house is: ',spearman_corr1,' with p value of: ',p_val_spearman1)

if spearman_corr<=0.19 and spearman_corr1<=0.19:
    print('The degree of Spearman correlation is very weak')
elif 0.20<=spearman_corr<=0.39 and .20<=spearman_corr1<=0.39:
    print('The degree of Spearman correlation is weak')
elif 0.40<=spearman_corr<=0.59 and 0.40<=spearman_corr1<=0.59:
    print('The degree of Spearman correlation is moderate')
elif 0.60<=spearman_corr<=0.79 and 0.60<=spearman_corr1<=0.79:
    print('The degree of Spearman correlation is strong')
elif 0.80<=spearman_corr<=1 and 0.80<=spearman_corr1<=1:
    print('The degree of Spearman correlation is very strong')
else:
    print('There is no Spearman correlation')

#Plot the correlation
correlation=merged.corr()
plt.figure(figsize=(6,6))
sns.heatmap(correlation,vmin=-1,vmax=1,cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.xticks(horizontalalignment='right')
plt.show()

#Plot a correlation heatmap
correlation_house=correlation[['Price_house']]
correlation_house=correlation_house.sort_values(by='Price_house',ascending=False)
plt.figure(figsize=(3,6))
sns.heatmap(correlation_house,vmin=-1,vmax=1,cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.xticks(horizontalalignment='right')
plt.show()

#Convert the columns to numpy arrays to allow PCA analysis
column_values_np=np.asarray(merged.columns.values)
merged_np=merged.to_numpy()
merged_np=merged_np[:,2::]

#PCA=2
pca=PCA(n_components=2)
pca.fit(merged_np)
print(pca.explained_variance_ratio_)

pca_transform=pca.transform(merged_np)

#Plot the PCA transformation
plt.figure(figsize=(6,6))
plt.scatter(pca_transform[:,0],pca_transform[:,1],alpha=0.4,s=50,linewidths=0)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('First two components ')
plt.show()

#Cross correlation with scipy
cross_corr = signal.correlate(merged_np[2],merged_np[3])
print(cross_corr)

plt.plot(cross_corr)
plt.xlabel('House Price')
plt.ylabel('Airbnb Price')
plt.title('Cross correlation between house and airbnb price')
plt.show()

merged['Prev_Year_House_Price'] = merged['Price_airbnb'].shift(1)
merged = merged[['Year','Area','Price_house','Prev_Year_House_Price','Price_airbnb','Total_airbnb']]

merged.to_csv('test.csv')

plt.subplot(2,1,1)
plt.scatter(merged['Year'],merged['Price_house'])

plt.subplot(2,1,2)
plt.scatter(merged['Year'],merged['Prev_Year_House_Price'])
plt.show()



#spurious correlation
#lasso regression moddel