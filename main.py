import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import spatial
from scipy import signal
from scipy import misc
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Import house dataset
house = pd.read_csv('pp-complete.csv',header=None)

#Check everything is in place
print(house.shape)
house.head()

#Import airbnb listing dataset
file   = pd.read_csv('listings.csv',dtype={'weekly_price':str,'monthly_price':str,'license':str,'jurisdiction_names':str})

#Drop irrelevant columns
airbnb = file.loc[0:,['id', 'name','neighbourhood_cleansed', 'city', 'zipcode', 'smart_location', 'property_type',
                    'room_type', 'accommodates', 'price','weekly_price', 'monthly_price', 'security_deposit',
                    'cleaning_fee', 'extra_people','first_review','last_review']]

#Drop empty cells with no last review date as this column will be taken for the time stamp
airbnb = airbnb.dropna(subset=['last_review'])

#Check if everything is in place
print(airbnb.shape)
print(airbnb.head())

#Change the data type
airbnb[['price','weekly_price','monthly_price','security_deposit','cleaning_fee',
        'extra_people']]= airbnb[['price','weekly_price','monthly_price','security_deposit','cleaning_fee',
                                 'extra_people']].replace('[\$,]','',regex=True).astype(float)

#Convert dollar to pound
airbnb[['price','weekly_price','monthly_price']]=airbnb[['price','weekly_price','monthly_price']]*0.78

#Find the average days of weekly and monthly price columns
weekly_avg  = (airbnb['weekly_price']/airbnb['price']).mean()
monthly_avg = (airbnb['monthly_price']/airbnb['price']).mean()

print(weekly_avg,monthly_avg)

#Fill in the missing values with the average days
airbnb['weekly_price']  = airbnb['weekly_price'].fillna(airbnb['price']*weekly_avg)
airbnb['monthly_price'] = airbnb['monthly_price'].fillna(airbnb['price']*monthly_avg)

#Calculate the yearly price based on the given monthly price
Price_airbnb_original = airbnb['monthly_price']*12
airbnb.insert(17,'Price_airbnb_original',Price_airbnb_original)

#Create new columns of average 365 days a year
Price_airbnb_cal = airbnb['price']*365
airbnb.insert(18,'Price_airbnb_cal',Price_airbnb_cal)

#Create new columns of total prices which include other fees
airbnb['cleaning_fee'] = airbnb['cleaning_fee'].fillna(0)

total_airbnb_original = airbnb['Price_airbnb_original']+airbnb['cleaning_fee']*365
airbnb.insert(19,'Total_airbnb_original',total_airbnb_original)

total_airbnb_cal = airbnb['Price_airbnb_cal']+airbnb['cleaning_fee']*365
airbnb.insert(20,'Total_airbnb_cal',total_airbnb_cal)

#Drop rows with no price values
null_price = airbnb.loc[airbnb['price']==0]
null_price['neighbourhood_cleansed'].value_counts()

#There are only 15 rows with missing price values, these will be dropped
airbnb = airbnb[airbnb['price']!= 0]

print(airbnb.shape)

#Change the format of date
airbnb['first_review'] = pd.to_datetime(airbnb['first_review'],format='%Y-%m-%d')
#airbnb['first_review'] = pd.to_datetime(airbnb['first_review'])

airbnb['last_review']  = pd.to_datetime(airbnb['last_review'],format='%Y-%m-%d')
#airbnb['last_review']  = pd.to_datetime(airbnb['last_review'])

#Sort the dates from the oldest to newest
airbnb = airbnb.sort_values(by=['last_review'],ascending=[True])

#Check if everything is in place
print(airbnb.shape)
print(airbnb.head())

#Set 'last_review' as the index for the airbnb dataset to allow temporal analysis
airbnb.set_index(airbnb['last_review'],inplace=True)

# Inspect the temporal distribution by year
yearly_count = airbnb.resample('Y').count()[['last_review']]

# Visualize the evolution
plt.figure(figsize = (15, 5))
sns.barplot(yearly_count.index, yearly_count.values.flatten())
plt.xlabel('Date by Year')
plt.ylabel('Number of Airbnb Listings')
plt.title('Yearly Evolution of the Number of Airbnb Listings')
plt.xticks(rotation=90)
plt.tight_layout()

plt.hist(airbnb['neighbourhood_cleansed'],bins=32,rwidth=0.8,color='black',align='left')
plt.title('Number of Airbnb Listings per London Borough')
plt.xlabel('Borough of London')
plt.ylabel('Number of Airbnb Listings')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.hist(airbnb['property_type'],bins=40,rwidth=0.8,color='black',log=True,align='left')
plt.xlabel('Property Types')
plt.ylabel('Number of Property Types')
plt.title('Number of Airbnb Property Types')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.hist(airbnb['room_type'],color='black',align='left')
plt.title('Number of Airbnb Room Type ')
plt.xlabel('Room Types')
plt.ylabel('Number of Room Types')
plt.tight_layout()
plt.show()

#Creat a new file to avoid overlapping
airbnb1 = airbnb

#Extract the year and month from the date column
airbnb1['YearMonth'] = airbnb1['last_review'].map(lambda x: 100*x.year + x.month)

#Extract the first part (postcode districts) from Postcodes
airbnb1['Postcode'] = airbnb1['zipcode'].str[0:3]

#Group the postcode and year columns and reset idex
airbnb1 = airbnb1.groupby(['YearMonth','Postcode']).mean().reset_index()

#Drop further irrelevant columns
airbnb1 = airbnb1.drop(columns=['id','accommodates','weekly_price','monthly_price','security_deposit','cleaning_fee','extra_people'])

#Rearrange the columns
airbnb1 = airbnb1[['Price_airbnb_original','Price_airbnb_cal','Total_airbnb_original','Total_airbnb_cal','Postcode','YearMonth']]

print(airbnb1.shape)
print(airbnb1.head())

#Remove special characters and symbols in the Postcode column
def preprocess(Postcode):
    Postcode = Postcode.str.upper()
    Postcode = Postcode.dropna()
    Postcode = Postcode.str.replace('[','')
    Postcode = Postcode.str.replace('€','')
    Postcode = Postcode.str.replace('.','')
    Postcode = Postcode.str.replace('$', '')
    return(Postcode)


# Extract the first part (postcode district) from postcodes
airbnb1['Postcode'] = preprocess(airbnb1['Postcode'])

airbnb1 = airbnb1[~airbnb1['Postcode'].astype(str).str.startswith('1')]
airbnb1 = airbnb1[~airbnb1['Postcode'].astype(str).str.startswith('3')]

# Spliting the Postcode areas into East, West, North, North Weset, South East, South West, Central and Outer
East_airbnb = airbnb1[airbnb1['Postcode'].str[0] == 'E']
East_airbnb = East_airbnb[East_airbnb['Postcode'].str[1] != 'C']
East_airbnb = East_airbnb[East_airbnb['Postcode'].str[1] != 'N']

if East_airbnb['Postcode'].all() <= 'E18':
 East_airbnb['Postcode'] = East_airbnb['Postcode']
else:
 East_airbnb['Postcode'] = East_airbnb['Postcode'].str[0:2]

East_airbnb.reset_index()

West_airbnb = airbnb1[airbnb1['Postcode'].str[0] == 'W']
West_airbnb = West_airbnb[West_airbnb['Postcode'].str[1] != 'C']

if West_airbnb['Postcode'].all() <= 'W13':
 West_airbnb['Postcode'] = West_airbnb['Postcode']
else:
 West_airbnb['Postcode'] = West_airbnb['Postcode'].str[0:2]

West_airbnb.reset_index()

North_airbnb = airbnb1[airbnb1['Postcode'].str[0] == 'N']
North_West_airbnb = North_airbnb[North_airbnb['Postcode'].str[1] == 'W']
North_airbnb = North_airbnb[North_airbnb['Postcode'].str[1] != 'W']

if North_airbnb['Postcode'].all() <= 'N22':
 North_airbnb['Postcode'] = North_airbnb['Postcode']
else:
 North_airbnb['Postcode'] = North_airbnb['Postcode'].str[0:2]

North_airbnb.reset_index()
North_West_airbnb.reset_index()

South_airbnb = airbnb1[airbnb1['Postcode'].str[0] == 'S']
South_East_airbnb = South_airbnb[South_airbnb['Postcode'].str[1] == 'E']
South_West_airbnb = South_airbnb[South_airbnb['Postcode'].str[1] == 'W']

South_East_airbnb.reset_index()
South_West_airbnb.reset_index()

Central_airbnb = airbnb1[airbnb1['Postcode'].str[1] == 'C']

Central_airbnb.reset_index()

IG = airbnb1[airbnb1['Postcode'].str[0] == 'I']
RM = airbnb1[airbnb1['Postcode'].str[0] == 'R']
EN = airbnb1[airbnb1['Postcode'].str[1] == 'N']
DA = airbnb1[airbnb1['Postcode'].str[0] == 'D']
BR = airbnb1[airbnb1['Postcode'].str[0] == 'B']
CR = airbnb1[airbnb1['Postcode'].str[0] == 'C']
SM = airbnb1[airbnb1['Postcode'].str[1] == 'M']
KT = airbnb1[airbnb1['Postcode'].str[0] == 'K']
TW = airbnb1[airbnb1['Postcode'].str[0] == 'T']
UB = airbnb1[airbnb1['Postcode'].str[0] == 'U']
HA = airbnb1[airbnb1['Postcode'].str[0] == 'H']
WD = airbnb1[airbnb1['Postcode'].str[1] == 'D']

Outer_airbnb = pd.concat([IG, RM, EN, DA, BR, CR, SM, KT, TW, UB, HA, WD]).reset_index()

East_count = East_airbnb['Postcode'].value_counts()
East_count = East_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(East_count.index,East_count.values,palette='Set2')
plt.title('Top 10 East London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

West_count = West_airbnb['Postcode'].value_counts()
West_count = West_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(West_count.index,West_count.values,palette='Set2')
plt.title('Top 10 West London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

North_count = North_airbnb['Postcode'].value_counts()
North_count = North_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(North_count.index,North_count.values,palette='Set2')
plt.title('Top 10 North London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

NW_count = North_West_airbnb['Postcode'].value_counts()
NW_count = NW_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(NW_count.index,NW_count.values,palette='Set2')
plt.title('Top 10 North West London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

SE_count = South_East_airbnb['Postcode'].value_counts()
SE_count = SE_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(SE_count.index,SE_count.values,palette='Set2')
plt.title('Top 10 South East London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

SW_count = South_West_airbnb['Postcode'].value_counts()
SW_count = SW_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(SW_count.index,SW_count.values,palette='Set2')
plt.title('Top 10 South West London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

Central_count = Central_airbnb['Postcode'].value_counts()
Central_count = Central_count[:5,]
plt.figure(figsize=(15,6))
sns.barplot(Central_count.index,Central_count.values,palette='Set2')
plt.title('Central London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

Outer_count = Outer_airbnb['Postcode'].value_counts()
Outer_count = Outer_count[:10,]
plt.figure(figsize=(15,6))
sns.barplot(Outer_count.index,Outer_count.values,palette='Set2')
plt.title('Top 10 Outer London Postcode Districts with most Airbnb Listings')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Airbnb Listings')

#Drop irrelevant columns
house.columns=['ID','Price_house','Time','Postcode','a','b','c','First','Number','Street','Locality','City','Area','County','A','A1']
house = house.drop(columns={'ID','a','b','c','First','Number','Street','Locality','A','A1'})
print(house.shape)
print(house.head())

#Filter the dataset to London only
house = house.loc[house['County']=='GREATER LONDON']
house = house.loc[house['City']=='LONDON']

#Change the date format
Clean_Time = pd.to_datetime(house['Time'],format='%Y-%m-%d %H:%M')
house.insert(1,'Clean_Time',Clean_Time)

#Create another dataframe to avoid overlapping
house1 = house

#Extract the year and month from date column
house1['YearMonth'] = house1['Clean_Time'].map(lambda x: 100*x.year + x.month)

#Extracting postcode districts
house1['Postcode'] = house1['Postcode'].str[0:3]

#Group the dataset by YearMonth and Postcode
house1 = house1.groupby(['YearMonth','Postcode']).mean().reset_index()

#Rearranging the columns
house1 = house1[['YearMonth','Postcode','Price_house']]

print(house1.shape)
print(house1.head())

# Extract the postocde ditricts from postcode
house1['Postcode'] = preprocess(house1['Postcode'])

# Spliting the Postcode areas into East, West, North, North Weset, South East, South West, Central
# and Outer
East_house = house1[house1['Postcode'].str[0] == 'E']
East_house = East_house[East_house['Postcode'].str[1] != 'C']
East_house = East_house[East_house['Postcode'].str[1] != 'N']

if East_house['Postcode'].all() <= 'E18':
 East_house['Postcode'] = East_house['Postcode']
else:
 East_house['Postcode'] = East_house['Postcode'].str[0:2]

East_house.reset_index()

West_house = house1[house1['Postcode'].str[0] == 'W']
West_house = West_house[West_house['Postcode'].str[1] != 'C']

if West_house['Postcode'].all() <= 'W13':
 West_house['Postcode'] = West_house['Postcode']
else:
 West_house['Postcode'] = West_house['Postcode'].str[0:2]

West_house.reset_index()

North_house = house1[house1['Postcode'].str[0] == 'N']
North_West_house = North_house[North_house['Postcode'].str[1] == 'W']
North_house = North_house[North_house['Postcode'].str[1] != 'W']

if North_house['Postcode'].all() <= 'N22':
 North_house['Postcode'] = North_house['Postcode']
else:
 North_house['Postcode'] = North_house['Postcode'].str[0:2]

North_house.reset_index()
North_West_house.reset_index()

South_house = house1[house1['Postcode'].str[0] == 'S']

South_East_house = South_house[South_house['Postcode'].str[1] == 'E']
South_West_house = South_house[South_house['Postcode'].str[1] == 'W']

South_East_house.reset_index()
South_West_house.reset_index()

Central_house = house1[house1['Postcode'].str[1] == 'C']

Central_house.reset_index()

IG = house1[house1['Postcode'].str[0] == 'I']
RM = house1[house1['Postcode'].str[0] == 'R']
EN = house1[house1['Postcode'].str[1] == 'N']
DA = house1[house1['Postcode'].str[0] == 'D']
BR = house1[house1['Postcode'].str[0] == 'B']
CR = house1[house1['Postcode'].str[0] == 'C']
SM = house1[house1['Postcode'].str[1] == 'M']
KT = house1[house1['Postcode'].str[0] == 'K']
TW = house1[house1['Postcode'].str[0] == 'T']
UB = house1[house1['Postcode'].str[0] == 'U']
HA = house1[house1['Postcode'].str[0] == 'H']
WD = house1[house1['Postcode'].str[1] == 'D']

Outer_house = pd.concat([IG, RM, EN, DA, BR, CR, SM, KT, TW, UB, HA, WD]).reset_index()

#Set 'YearMonth' as the index for the airbnb dataset to allow temporal analysis
house.set_index(house['Clean_Time'],inplace=True)

# Inspect the temporal distribution by year
yearly_count1 = house.resample('Y').count()[['Clean_Time']]

# Visualize the evolution
plt.figure(figsize = (15, 5))
sns.barplot(yearly_count1.index, yearly_count1.values.flatten())
plt.xlabel('Date by Year')
plt.ylabel('Number of Houses')
plt.title('Yearly Evolution of the Number of Houses')
plt.xticks(rotation=90)
plt.tight_layout()

East_count_house = East_house['Postcode'].value_counts()
East_count_house = East_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(East_count_house.index,East_count_house.values,palette='Set2')
plt.title('Top 10 East London Postcode Districs with most Houses')
plt.xlabel('London Postcode Districs')
plt.ylabel('Number of Houses')

West_count_house = West_house['Postcode'].value_counts()
West_count_house = West_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(West_count_house.index,West_count_house.values,palette='Set2')
plt.title('Top 10 West London Postcode District with most Houses')
plt.xlabel('London Postcode Districts')
plt.ylabel('Number of Houses')

North_count_house = North_house['Postcode'].value_counts()
North_count_house = North_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(North_count_house.index,North_count_house.values,palette='Set2')
plt.title('Top 10 North London Postcode Districts with most Houses')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Houses')

NW_count_house = North_West_house['Postcode'].value_counts()
NW_count_house = NW_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(NW_count_house.index,NW_count_house.values,palette='Set2')
plt.title('Top 10 North West London Postcode Districts with most Houses')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Houses')

SW_count_house = South_West_house['Postcode'].value_counts()
SW_count_house = SW_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(SW_count_house.index,SW_count_house.values,palette='Set2')
plt.title('Top 10 South West London Postcode Districts with most Houses')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Houses')

SE_count_house = South_East_house['Postcode'].value_counts()
SE_count_house = SE_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(SE_count_house.index,SE_count_house.values,palette='Set2')
plt.title('Top 10 South East London Postcode Districts with most Houses')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Houses')

Central_count_house = Central_house['Postcode'].value_counts()
Central_count_house = Central_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(Central_count_house.index,Central_count_house.values,palette='Set2')
plt.title('Top 10 Central London Postcode Districts with most Houses')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Houses')

Outer_count_house = Outer_house['Postcode'].value_counts()
Outer_count_house = Outer_count_house[:10,]
plt.figure(figsize=(15,6))
sns.barplot(Outer_count_house.index,Outer_count_house.values,palette='Set2')
plt.title('Top 10 Outer London Postcode Districts with most Houses')
plt.xlabel('London Postcode District')
plt.ylabel('Number of Houses')

#Merge two datasets on Year and Postcode
merged = pd.merge(house1,airbnb1,how='outer',on=['YearMonth','Postcode'])

#Sort both columns
merged = merged.sort_values(by=['Postcode','YearMonth'],ascending=[True,True]).reset_index()

#Rearrange columns
merged = merged[['YearMonth','Postcode','Price_house','Price_airbnb_original','Price_airbnb_cal','Total_airbnb_original','Total_airbnb_cal']]

print(merged.head())

#Set YearMonth as the datetime index of the merged dataset
datetime_series = pd.to_datetime(merged['YearMonth'].astype(str), format='%Y%m')
datetime_index = pd.DatetimeIndex(datetime_series.values)
merged1 = merged.set_index(datetime_index)
merged1 = merged1.drop('YearMonth',axis=1)

#Drop missing rows with both house prices and airbnb prices
merged1 = merged1.dropna()

#Sort the index from oldest to newest
merged1 = merged1.sort_index()

print(merged1.head())

# Split the Postcode areas into East, West, North, North Weset, South East, South West, Central and Outer
East_merged = merged1[merged1['Postcode'].str[0] == 'E']
East_merged = East_merged[East_merged['Postcode'].str[1] != 'C']
East_merged = East_merged[East_merged['Postcode'].str[1] != 'N']

if East_merged['Postcode'].all() <= 'E18':
 East_merged['Postcode'] = East_merged['Postcode']
else:
 East_merged['Postcode'] = East_merged['Postcode'].str[0:2]

East_merged.drop_duplicates(subset=['Postcode']).reset_index()

West_merged = merged1[merged1['Postcode'].str[0] == 'W']
West_merged = West_merged[West_merged['Postcode'].str[1] != 'C']

West_merged.drop_duplicates(subset=['Postcode']).reset_index()

North_merged = merged1[merged1['Postcode'].str[0] == 'N']
North_West_merged = North_merged[North_merged['Postcode'].str[1] == 'W']
North_merged = North_merged[North_merged['Postcode'].str[1] != 'W']

North_merged.drop_duplicates(subset=['Postcode']).reset_index()
North_West_merged.drop_duplicates(subset=['Postcode']).reset_index()

South_merged = merged1[merged1['Postcode'].str[0] == 'S']
South_East_merged = South_merged[South_merged['Postcode'].str[1] == 'E']
South_West_merged = South_merged[South_merged['Postcode'].str[1] == 'W']

South_East_merged.drop_duplicates(subset=['Postcode']).reset_index()
South_West_merged.drop_duplicates(subset=['Postcode']).reset_index()

Central_merged = merged1[merged1['Postcode'].str[1] == 'C']

Central_merged.drop_duplicates(subset=['Postcode']).reset_index()

IG = merged1[merged1['Postcode'].str[0] == 'I']
RM = merged1[merged1['Postcode'].str[0] == 'R']
EN = merged1[merged1['Postcode'].str[1] == 'N']
DA = merged1[merged1['Postcode'].str[0] == 'D']
BR = merged1[merged1['Postcode'].str[0] == 'B']
CR = merged1[merged1['Postcode'].str[0] == 'C']
SM = merged1[merged1['Postcode'].str[1] == 'M']
KT = merged1[merged1['Postcode'].str[0] == 'K']
TW = merged1[merged1['Postcode'].str[0] == 'T']
UB = merged1[merged1['Postcode'].str[0] == 'U']
HA = merged1[merged1['Postcode'].str[0] == 'H']
WD = merged1[merged1['Postcode'].str[1] == 'D']

Outer_merged = pd.concat([IG, RM, EN, DA, BR, CR, SM, KT, TW, UB, HA, WD]).reset_index()

print(len(East_merged)+len(West_merged)+len(North_merged)+len(North_West_merged)+
     len(South_West_merged)+len(South_East_merged)+len(Central_merged)+len(Outer_merged))

plt.figure(figsize=(15,8))
sns.barplot(x=merged1['Postcode'],y=merged1['Price_house'],palette='vlag')
plt.xticks(rotation=65,horizontalalignment='right')
plt.title('Average London House Prices per Postcode District')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=merged1['Postcode'],y=merged1['Price_airbnb_original'],palette='rocket')
plt.xticks(rotation=65,horizontalalignment='right')
plt.title('Average London Airbnb Listing Price (original)per Postcode District')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=merged1['Postcode'],y=merged1['Price_airbnb_cal'],palette='rocket')
plt.xticks(rotation=65,horizontalalignment='right')
plt.title('Average London Airbnb Listing Price (calculated) per Postcode District')
plt.tight_layout()
plt.show()

#Visualise average house and aribnb listing prices in East London
plt.figure(figsize=(15,8))
sns.barplot(x=East_merged['Postcode'],y=East_merged['Price_house'],palette='vlag')
plt.title('Average House Price in East London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=East_merged['Postcode'],y=East_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original) in East London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=East_merged['Postcode'],y=East_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in East London')
plt.tight_layout()

#Visualise average house and aribnb listing prices in West London
plt.figure(figsize=(15,8))
sns.barplot(x=West_merged['Postcode'],y=West_merged['Price_house'],palette='vlag')
plt.title('Average House Price in West London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=West_merged['Postcode'],y=West_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original) in West London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=West_merged['Postcode'],y=West_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in West London')
plt.tight_layout()

#Visualise average house and aribnb listing prices in North London
plt.figure(figsize=(15,8))
sns.barplot(x=North_merged['Postcode'],y=North_merged['Price_house'],palette='vlag')
plt.title('Average House Price in North London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=North_merged['Postcode'],y=North_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original)in North London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=North_merged['Postcode'],y=North_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in North London')
plt.tight_layout()

#Visualise average house and aribnb listing prices in North West London
plt.figure(figsize=(15,8))
sns.barplot(x=North_West_merged['Postcode'],y=North_West_merged['Price_house'],palette='vlag')
plt.title('Average House Price in North West London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=North_West_merged['Postcode'],y=North_West_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original) in North West London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=North_West_merged['Postcode'],y=North_West_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in North West London')
plt.tight_layout()

#Visualise average house and aribnb listing prices in South West London
plt.figure(figsize=(15,8))
sns.barplot(x=South_West_merged['Postcode'],y=South_West_merged['Price_house'],palette='vlag')
plt.title('Average Hous3 Price in South West London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=South_West_merged['Postcode'],y=South_West_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original) in South West London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=South_West_merged['Postcode'],y=South_West_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in South West London')
plt.tight_layout()

#Visualise average house and aribnb listing prices in South East London
plt.figure(figsize=(15,8))
sns.barplot(x=South_East_merged['Postcode'],y=South_East_merged['Price_house'],palette='vlag')
plt.title('Average House Price in South East London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=South_East_merged['Postcode'],y=South_East_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original) in South East London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=South_East_merged['Postcode'],y=South_East_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in South East London')
plt.tight_layout()

#Visualise average house and aribnb listing prices in Central London
plt.figure(figsize=(15,8))
sns.barplot(x=Central_merged['Postcode'],y=Central_merged['Price_house'],palette='vlag')
plt.title('Average House Price in Central London')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(x=Central_merged['Postcode'],y=Central_merged['Price_airbnb_original'],palette='vlag')
plt.title('Average Airbnb Listing Price (Original)in Central London')
plt.tight_layout()

plt.figure(figsize=(15,8))
sns.barplot(x=Central_merged['Postcode'],y=Central_merged['Price_airbnb_cal'],palette='vlag')
plt.title('Average Airbnb Listing Price (Calculated) in Central London')
plt.tight_layout()

#Visualise the datasets with histograms to see if log transformation is needed
plt.hist(merged1['Price_house'])
plt.title('Price_house')
plt.xticks(rotation=45)
plt.show()

plt.subplot(1,2,1)
plt.hist(merged1['Total_airbnb_original'])
plt.title('Total_airbnb_original')
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.hist(merged1['Total_airbnb_cal'])
plt.title('Total_airbnb_cal')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.hist(merged1['Price_airbnb_original'])
plt.title('Price_airbnb_original')
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.hist(merged1['Price_airbnb_cal'])
plt.title('Price_airbnb_cal')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Applying log transformation
merged1['Price_house'].apply(np.log).hist(label='Price House')
merged1['Total_airbnb_original'].apply(np.log).hist(label='Total Price Airbnb Original')
plt.legend(loc='upper left')
plt.show()

merged1['Price_house'].apply(np.log).hist(label='Price House')
merged1['Total_airbnb_cal'].apply(np.log).hist(label='Total Price Airbnb Calculated')
plt.legend(loc='upper left')
plt.show()

merged1['Price_house'].apply(np.log).hist(label='Price House')
merged1['Price_airbnb_original'].apply(np.log).hist(label='Price Airbnb Original')
plt.legend(loc='upper left')
plt.show()

#Insert log transformed columns to the dataframe
log_Price_house           = np.log(merged1.Price_house)
log_Price_airbnb_original = np.log(merged1.Price_airbnb_original)
log_Price_airbnb_cal      = np.log(merged1.Price_airbnb_cal)
log_Total_airbnb_original = np.log(merged1.Total_airbnb_original)
log_Total_airbnb_cal      = np.log(merged1.Total_airbnb_cal)

merged1.insert(2,'log_Price_house',log_Price_house)
merged1.insert(4,'log_Price_airbnb_original',log_Price_airbnb_original)
merged1.insert(6,'log_Price_airbnb_cal',log_Price_airbnb_cal)
merged1.insert(8,'log_Total_airbnb_original',log_Total_airbnb_original)
merged1.insert(10,'log_Total_airbnb_cal',log_Total_airbnb_cal)

print(merged1.head())

# Applying log transformation to each postcode districts
East_merged = merged1[merged1['Postcode'].str[0] == 'E']
East_merged = East_merged[East_merged['Postcode'].str[1] != 'C']
East_merged = East_merged[East_merged['Postcode'].str[1] != 'N']

if East_merged['Postcode'].all() <= 'E18':
 East_merged['Postcode'] = East_merged['Postcode']
else:
 East_merged['Postcode'] = East_merged['Postcode'].str[0:2]

East_merged.drop_duplicates(subset=['Postcode']).reset_index()

West_merged = merged1[merged1['Postcode'].str[0] == 'W']
West_merged = West_merged[West_merged['Postcode'].str[1] != 'C']

West_merged.drop_duplicates(subset=['Postcode']).reset_index()

North_merged = merged1[merged1['Postcode'].str[0] == 'N']
North_West_merged = North_merged[North_merged['Postcode'].str[1] == 'W']
North_merged = North_merged[North_merged['Postcode'].str[1] != 'W']

North_merged.drop_duplicates(subset=['Postcode']).reset_index()
North_West_merged.drop_duplicates(subset=['Postcode']).reset_index()

South_merged = merged1[merged1['Postcode'].str[0] == 'S']
South_East_merged = South_merged[South_merged['Postcode'].str[1] == 'E']
South_West_merged = South_merged[South_merged['Postcode'].str[1] == 'W']

South_East_merged.drop_duplicates(subset=['Postcode']).reset_index()
South_West_merged.drop_duplicates(subset=['Postcode']).reset_index()

Central_merged = merged1[merged1['Postcode'].str[1] == 'C']

Central_merged.drop_duplicates(subset=['Postcode']).reset_index()

IG = merged1[merged1['Postcode'].str[0] == 'I']
RM = merged1[merged1['Postcode'].str[0] == 'R']
EN = merged1[merged1['Postcode'].str[1] == 'N']
DA = merged1[merged1['Postcode'].str[0] == 'D']
BR = merged1[merged1['Postcode'].str[0] == 'B']
CR = merged1[merged1['Postcode'].str[0] == 'C']
SM = merged1[merged1['Postcode'].str[1] == 'M']
KT = merged1[merged1['Postcode'].str[0] == 'K']
TW = merged1[merged1['Postcode'].str[0] == 'T']
UB = merged1[merged1['Postcode'].str[0] == 'U']
HA = merged1[merged1['Postcode'].str[0] == 'H']
WD = merged1[merged1['Postcode'].str[1] == 'D']

Outer_merged = pd.concat([IG, RM, EN, DA, BR, CR, SM, KT, TW, UB, HA, WD]).reset_index()

f, ax = plt.subplots(1, 2)
ax[0].boxplot(merged1['Price_house'])
ax[1].hist(merged1['Price_house'])
plt.xticks(rotation=65)
plt.tight_layout()

print('mean:', merged1['Price_house'].mean(),'median:', merged1['Price_house'].median(),
      '\n std:', merged1['Price_house'].std(),'iqr:', stats.iqr(merged1['Price_house']),
      '\n std:', merged1['Price_house'].std(),'mad:', sm.robust.scale.mad(merged1['Price_house']))

f, ax = plt.subplots(1, 2)
ax[0].boxplot(merged1['Price_airbnb_original'])
ax[1].hist(merged1['Price_airbnb_original'])
plt.xticks(rotation=65)
plt.tight_layout()

print('mean:', merged1['Price_airbnb_original'].mean(),'median:', merged1['Price_airbnb_original'].median(),
      '\n std:', merged1['Price_airbnb_original'].std(),'iqr:', stats.iqr(merged1['Price_airbnb_original']),
      '\n std:', merged1['Price_airbnb_original'].std(),'mad:', sm.robust.scale.mad(merged1['Price_airbnb_original']))

f, ax = plt.subplots(1, 2)
ax[0].boxplot(merged1['Price_airbnb_cal'])
ax[1].hist(merged1['Price_airbnb_cal'])
plt.xticks(rotation=65)
plt.tight_layout()

print('mean:', merged1['Price_airbnb_cal'].mean(),'median:', merged1['Price_airbnb_cal'].median(),
      '\n std:', merged1['Price_airbnb_cal'].std(),'iqr:', stats.iqr(merged1['Price_airbnb_cal']),
      '\n std:', merged1['Price_airbnb_cal'].std(),'mad:', sm.robust.scale.mad(merged1['Price_airbnb_cal']))

f, ax = plt.subplots(1, 2)
ax[0].boxplot(merged1['Total_airbnb_original'])
ax[1].hist(merged1['Total_airbnb_original'])
plt.xticks(rotation=65)
plt.tight_layout()

print('mean:', merged1['Total_airbnb_original'].mean(),'median:', merged1['Total_airbnb_original'].median(),
      '\n std:', merged1['Total_airbnb_original'].std(),'iqr:', stats.iqr(merged1['Total_airbnb_original']),
      '\n std:', merged1['Total_airbnb_original'].std(),'mad:', sm.robust.scale.mad(merged1['Total_airbnb_original']))

f, ax = plt.subplots(1, 2)
ax[0].boxplot(merged1['Total_airbnb_cal'])
ax[1].hist(merged1['Total_airbnb_cal'])
plt.xticks(rotation=65)
plt.tight_layout()

print('mean:', merged1['Total_airbnb_cal'].mean(),'median:', merged1['Total_airbnb_cal'].median(),
      '\n std:', merged1['Total_airbnb_cal'].std(),'iqr:', stats.iqr(merged1['Total_airbnb_cal']),
      '\n std:', merged1['Total_airbnb_cal'].std(),'mad:', sm.robust.scale.mad(merged1['Total_airbnb_cal']))

#Visualise a simple correlation between log transformed house prices and airbnb prices comparing
#with non log transformed datasets
sns.lmplot(x='log_Total_airbnb_original',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Log Transfomred Total Airbnb Price vs House Price (Original)')
sns.lmplot(x='Total_airbnb_original',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb Price vs House Price (Original)')

#Visualise a simple correlation between log transformed house prices and airbnb prices comparing
#with non log transformed datasets
sns.lmplot(x='log_Total_airbnb_cal',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Log Transfomred Total Airbnb Price vs House Price (Calculated)')
sns.lmplot(x='Total_airbnb_cal',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb Price vs House Price (Calculated)')

sns.lmplot(x='log_Price_airbnb_original',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Log Transformed Airbnb Price vs House Price (Original)')
sns.lmplot(x='Price_airbnb_original',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Airbnb Price vs House Price (Original)')

sns.lmplot(x='log_Price_airbnb_cal',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Log Transformed Airbnb Price vs House Price (Calculated)')
sns.lmplot(x='Price_airbnb_cal',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Airbnb Price vs House Price (Calculated)')

#1D outliers for each individual columns are already shown above, now 2D outliers will be shown
#by comparing HP and ALP.
#Calculate the mean and the std of each feature and identify the outliers
mean_house                 = merged1['log_Price_house'].mean()
mean_airbnb_original       = merged1['log_Price_airbnb_original'].mean()
mean_airbnb_cal            = merged1['log_Price_airbnb_cal'].mean()
mean_total_airbnb_original = merged1['log_Total_airbnb_original'].mean()
mean_total_airbnb_cal      = merged1['log_Total_airbnb_cal'].mean()

std_house                  = merged1['log_Price_house'].std()
std_airbnb_original        = merged1['log_Price_airbnb_original'].std()
std_airbnb_cal             = merged1['log_Price_airbnb_cal'].std()
std_total_airbnb_original  = merged1['log_Total_airbnb_original'].std()
std_total_airbnb_cal       = merged1['log_Total_airbnb_cal'].std()

is_outlier_house                 = np.where(abs(merged1['log_Price_house'] - mean_house) > 2 * std_house,
                                            1, 0)
is_outlier_airbnb_original       = np.where(abs(merged1['log_Price_airbnb_original'] - mean_airbnb_original)
                                            > 2 * std_airbnb_original, 1, 0)
is_outlier_airbnb_cal            = np.where(abs(merged1['log_Price_airbnb_cal'] - mean_airbnb_cal)
                                            > 2 * std_airbnb_cal, 1, 0)
is_outlier_total_airbnb_original = np.where(abs(merged1['log_Total_airbnb_original'] - mean_total_airbnb_original)
                                            > 2 * std_total_airbnb_original, 1, 0)
is_outlier_total_airbnb_cal      = np.where(abs(merged1['log_Total_airbnb_cal'] - mean_total_airbnb_cal)
                                            > 2 * std_total_airbnb_cal, 1, 0)

merged1.insert(11,'is_outlier_house',is_outlier_house)
merged1.insert(12,'is_outlier_airbnb_original',is_outlier_airbnb_original)
merged1.insert(13,'is_outlier_airbnb_cal',is_outlier_airbnb_cal)
merged1.insert(14,'is_outlier_total_airbnb_original',is_outlier_total_airbnb_original)
merged1.insert(15,'is_outlier_total_airbnb_cal',is_outlier_total_airbnb_cal)

from collections import Counter
print('Outliers for log transformed Price_house are: ',Counter(merged1['is_outlier_house']))
print('Outliers for log transformed Price_airbnb_original are: ',Counter(merged1['is_outlier_airbnb_original']))
print('Outliers for log transformed Price_airbnb_cal are: ',Counter(merged1['is_outlier_airbnb_cal']))
print('Outliers for log transformed Total_house_original are: ',Counter(merged1['is_outlier_total_airbnb_original']))
print('Outliers for log transformed Total_house_cal are: ',Counter(merged1['is_outlier_total_airbnb_cal']))

#Calculate the mean and the std of each feature and identify the outliers
mean_house1                 = merged1['Price_house'].mean()
mean_airbnb_original1       = merged1['Price_airbnb_original'].mean()
mean_airbnb_cal1            = merged1['Price_airbnb_cal'].mean()
mean_total_airbnb_original1 = merged1['Total_airbnb_original'].mean()
mean_total_airbnb_cal1      = merged1['Total_airbnb_cal'].mean()

std_house1                  = merged1['Price_house'].std()
std_airbnb_original1        = merged1['Price_airbnb_original'].std()
std_airbnb_cal1             = merged1['Price_airbnb_cal'].std()
std_total_airbnb_original1  = merged1['Total_airbnb_original'].std()
std_total_airbnb_cal1       = merged1['Total_airbnb_cal'].std()

is_outlier_house1                 = np.where(abs(merged1['Price_house'] - mean_house1) > 2 * std_house1,
                                            1, 0)
is_outlier_airbnb_original1       = np.where(abs(merged1['Price_airbnb_original'] - mean_airbnb_original1)
                                            > 2 * std_airbnb_original1, 1, 0)
is_outlier_airbnb_cal1            = np.where(abs(merged1['Price_airbnb_cal'] - mean_airbnb_cal1)
                                            > 2 * std_airbnb_cal1, 1, 0)
is_outlier_total_airbnb_original1 = np.where(abs(merged1['Total_airbnb_original'] - mean_total_airbnb_original1)
                                            > 2 * std_total_airbnb_original1, 1, 0)
is_outlier_total_airbnb_cal1      = np.where(abs(merged1['Total_airbnb_cal'] - mean_total_airbnb_cal1)
                                            > 2 * std_total_airbnb_cal1, 1, 0)

merged1.insert(16,'is_outlier_house1',is_outlier_house1)
merged1.insert(17,'is_outlier_airbnb_original1',is_outlier_airbnb_original1)
merged1.insert(18,'is_outlier_airbnb_cal1',is_outlier_airbnb_cal1)
merged1.insert(19,'is_outlier_total_airbnb_original1',is_outlier_total_airbnb_original1)
merged1.insert(20,'is_outlier_total_airbnb_cal1',is_outlier_total_airbnb_cal1)

from collections import Counter
print('Outliers for  transformed Price_house are: ',Counter(merged1['is_outlier_house1']))
print('Outliers for  transformed Price_airbnb_original are: ',Counter(merged1['is_outlier_airbnb_original1']))
print('Outliers for  transformed Price_airbnb_cal are: ',Counter(merged1['is_outlier_airbnb_cal1']))
print('Outliers for  transformed Total_house_original are: ',Counter(merged1['is_outlier_total_airbnb_original1']))
print('Outliers for  transformed Total_house_cal are: ',Counter(merged1['is_outlier_total_airbnb_cal1']))

#Plot the outliers for with mahalnobis distance
column_values_log = merged1[['log_Total_airbnb_original','log_Price_house']].values
mean_vector_log   = np.asarray([merged1['log_Total_airbnb_original'].mean(),merged1['log_Price_house'].mean()]).reshape((1,2))
mahalanobis_distances_log = spatial.distance.cdist(column_values_log,mean_vector_log,'mahalanobis')[:,0]

column_values = merged1[['Total_airbnb_original','Price_house']].values
mean_vector   = np.asarray([merged1['Total_airbnb_original'].mean(),merged1['Price_house'].mean()]).reshape((1,2))
mahalanobis_distances = spatial.distance.cdist(column_values,mean_vector,'mahalanobis')[:,0]

plt.subplot(1,2,1)
plt.scatter(merged1['log_Total_airbnb_original'],merged1['log_Price_house'],c=mahalanobis_distances_log,cmap=plt.cm.Blues)
plt.xlabel('Log Transformed Total Airbnb Price')
plt.ylabel('Log Transformed House Price')
plt.title('Total Airbnb Price vs House Price \nwith Log Transformation \n(Original)')

plt.subplot(1,2,2)
plt.scatter(merged1['Total_airbnb_original'],merged1['Price_house'],c=mahalanobis_distances,cmap=plt.cm.Blues)
plt.xlabel('Total Airbnb Price')
plt.ylabel('House Price')
plt.title('Total Airbnb Price vs House Price \nwithout Log Transformation \n(Original)')
plt.tight_layout()
plt.show()

#Plot the outliers for with mahalnobis distance
column_values_log1 = merged1[['log_Total_airbnb_cal','log_Price_house']].values
mean_vector_log1   = np.asarray([merged1['log_Total_airbnb_cal'].mean(),merged1['log_Price_house'].mean()]).reshape((1,2))
mahalanobis_distances_log1 = spatial.distance.cdist(column_values_log1,mean_vector_log1,'mahalanobis')[:,0]

column_values1 = merged1[['Total_airbnb_cal','Price_house']].values
mean_vector1   = np.asarray([merged1['Total_airbnb_cal'].mean(),merged1['Price_house'].mean()]).reshape((1,2))
mahalanobis_distances1 = spatial.distance.cdist(column_values1,mean_vector1,'mahalanobis')[:,0]

plt.subplot(1,2,1)
plt.scatter(merged1['log_Total_airbnb_cal'],merged1['log_Price_house'],c=mahalanobis_distances_log1,cmap=plt.cm.Blues)
plt.xlabel('Log Transformed Total Airbnb Price')
plt.ylabel('Log Transformed House Price')
plt.title('Total Airbnb Price vs House Price \nwith Log Transformation \n(Calculated)')

plt.subplot(1,2,2)
plt.scatter(merged1['Total_airbnb_cal'],merged1['Price_house'],c=mahalanobis_distances1,cmap=plt.cm.Blues)
plt.xlabel('Total Airbnb Price')
plt.ylabel('House Price')
plt.title('Total Airbnb Price vs House Price \nwithout Log Transformation \n(Calculated)')
plt.tight_layout()
plt.show()

#Plot the outliers with mahalnobis distance
column_values_log2 = merged1[['log_Price_airbnb_original','log_Price_house']].values
mean_vector_log2   = np.asarray([merged1['log_Price_airbnb_original'].mean(),merged1['log_Price_house'].mean()]).reshape((1,2))
mahalanobis_distances_log2 = spatial.distance.cdist(column_values_log2,mean_vector_log2,'mahalanobis')[:,0]

column_values2 = merged1[['Price_airbnb_original','Price_house']].values
mean_vector2   = np.asarray([merged1['Price_airbnb_original'].mean(),merged1['Price_house'].mean()]).reshape((1,2))
mahalanobis_distances2 = spatial.distance.cdist(column_values1,mean_vector1,'mahalanobis')[:,0]

plt.subplot(1,2,1)
plt.scatter(merged1['log_Price_airbnb_original'],merged1['log_Price_house'],c=mahalanobis_distances_log2,cmap=plt.cm.Blues)
plt.xlabel('Log Transformed Airbnb Price')
plt.ylabel('Log Transformed House Price')
plt.title('Airbnb Price vs House Price \nwith Log Transformation \n(Original)')

plt.subplot(1,2,2)
plt.scatter(merged1['Price_airbnb_original'],merged1['Price_house'],c=mahalanobis_distances2,cmap=plt.cm.Blues)
plt.xlabel('Airbnb Price')
plt.ylabel('House Price')
plt.title('Airbnb Price vs House Price \nwithout Log Transformation \n(Original)')
plt.tight_layout()
plt.show()

#Plot the outliers with mahalnobis distance
column_values_log3 = merged1[['log_Price_airbnb_cal','log_Price_house']].values
mean_vector_log3  = np.asarray([merged1['log_Price_airbnb_cal'].mean(),merged1['log_Price_house'].mean()]).reshape((1,2))
mahalanobis_distances_log3 = spatial.distance.cdist(column_values_log3,mean_vector_log3,'mahalanobis')[:,0]

column_values3 = merged1[['Price_airbnb_cal','Price_house']].values
mean_vector3   = np.asarray([merged1['Price_airbnb_cal'].mean(),merged1['Price_house'].mean()]).reshape((1,2))
mahalanobis_distances3 = spatial.distance.cdist(column_values1,mean_vector1,'mahalanobis')[:,0]

plt.subplot(1,2,1)
plt.scatter(merged1['log_Price_airbnb_cal'],merged1['log_Price_house'],c=mahalanobis_distances_log3,cmap=plt.cm.Blues)
plt.xlabel('Log Transformed Airbnb Price')
plt.ylabel('Log Transformed House Price')
plt.title('Airbnb Price vs House Price \nwith Log Transformation \n(Calculated)')

plt.subplot(1,2,2)
plt.scatter(merged1['Price_airbnb_cal'],merged1['Price_house'],c=mahalanobis_distances3,cmap=plt.cm.Blues)
plt.xlabel('Airbnb Price')
plt.ylabel('House Price')
plt.title('Airbnb Price vs House Price \nwithout Log Transformation \n(Calculated)')
plt.tight_layout()
plt.show()

#A new dataframe will be created with outliers and the results can be compared later.
merged_with_outlier = merged1.copy(deep=False)

#Remove the outliers
merged1 = merged1[merged1['is_outlier_house'] == 0]
merged1 = merged1[merged1['is_outlier_airbnb_original'] == 0]
merged1 = merged1[merged1['is_outlier_airbnb_cal'] == 0]
merged1 = merged1[merged1['is_outlier_total_airbnb_original'] == 0]
merged1 = merged1[merged1['is_outlier_total_airbnb_cal'] == 0]
merged1 = merged1[merged1['is_outlier_house1'] == 0]
merged1 = merged1[merged1['is_outlier_airbnb_original1'] == 0]
merged1 = merged1[merged1['is_outlier_airbnb_cal1'] == 0]
merged1 = merged1[merged1['is_outlier_total_airbnb_original1'] == 0]
merged1 = merged1[merged1['is_outlier_total_airbnb_cal1'] == 0]

#Visualise the regressions of all sub-dataframes
plt.subplot(1,2,1)
sns.regplot(x='log_Price_airbnb_original',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin ORLT (Original)')

plt.subplot(1,2,2)
sns.regplot(x='Price_airbnb_original',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin ORW/LT (Original)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Price_airbnb_original',y='log_Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin OLT (Original)')

plt.subplot(1,2,2)
sns.regplot(x='Price_airbnb_original',y='Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin OW/LT (Original)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Price_airbnb_cal',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin ORLT (Calculated)')

plt.subplot(1,2,2)
sns.regplot(x='Price_airbnb_cal',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin ORW/LT (Calculated)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Price_airbnb_cal',y='log_Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin OLT (Calculated)')

plt.subplot(1,2,2)
sns.regplot(x='Price_airbnb_cal',y='Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Airbnb price vs \nHouse price \nin OW/LT (Calculated)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Total_airbnb_original',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \nHouse price \n in ORLT (Original)')

plt.subplot(1,2,2)
sns.regplot(x='Total_airbnb_original',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \n House price \nin ORW/LT (Original)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Total_airbnb_original',y='log_Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \nHouse price \n in OLT (Original)')

plt.subplot(1,2,2)
sns.regplot(x='Total_airbnb_original',y='Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \n House price \nin OW/LT (Original)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Total_airbnb_cal',y='log_Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \nHouse price \n in ORLT (Calculated)')

plt.subplot(1,2,2)
sns.regplot(x='Total_airbnb_cal',y='Price_house',data=merged1,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \n House price \nin ORW/LT (Calculated)')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.regplot(x='log_Total_airbnb_cal',y='log_Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \nHouse price \n in OLT (Calculated)')

plt.subplot(1,2,2)
sns.regplot(x='Total_airbnb_cal',y='Price_house',data=merged_with_outlier,scatter_kws={'alpha':0.4})
plt.title('Total Airbnb price vs \n House price \nin OW/LT (Calculated)')
plt.tight_layout()
plt.show()

#PCA Analysis
#Convert the columns to numpy arrays
merged_PCA = merged1[['Postcode','Total_airbnb_original','Price_airbnb_original','Total_airbnb_cal',
                      'Price_airbnb_cal','Price_house','log_Price_house','log_Total_airbnb_original',
                      'log_Price_airbnb_original','log_Total_airbnb_cal','log_Price_airbnb_cal']]
column_values_np = np.asarray(merged_PCA.columns.values)
merged_np = merged_PCA.to_numpy()
merged_np_filtered = merged_np[:,1::]

#Convert the columns to numpy arrays
merged_with_outlier_PCA = merged_with_outlier[['Postcode','Total_airbnb_original','Price_airbnb_original',
                                               'Total_airbnb_cal', 'Price_airbnb_cal','Price_house',
                                               'log_Price_house','log_Total_airbnb_original',
                                               'log_Price_airbnb_original','log_Total_airbnb_cal',
                                               'log_Price_airbnb_cal']]
column_values_np = np.asarray(merged_with_outlier_PCA.columns.values)
merged_outlier_np = merged_with_outlier_PCA.to_numpy()
merged__outlier_np_filtered = merged_np[:,1::]

#Assign 10 principle components
pca=PCA(n_components=10)
pca.fit(merged_np_filtered)
merged_projected=pca.transform(merged_np_filtered)
print(pca.explained_variance_ratio_)

pca1=PCA(n_components=10)
pca1.fit(merged__outlier_np_filtered)
merged_outlier_projected=pca.transform(merged__outlier_np_filtered)
print(pca1.explained_variance_ratio_)

print(pca.components_)
print(pca1.components_)

print(merged_projected.shape)
print(merged_outlier_projected.shape)

#Plot the visualisation of PCA with colour mapping of the dataset
coulour_mapping  = np.asarray(merged_projected[:,-1],'f')
coulour_mapping1 = np.asarray(merged_outlier_projected[:,-1],'f')

coulour_mapping_log  = np.asarray(merged_projected[:,-1],'f')
coulour_mapping_log1 = np.asarray(merged_outlier_projected[:,-1],'f')

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,0],merged_projected[:,4],s=50,c=coulour_mapping,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Original) \nin ORW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,0],merged_outlier_projected[:,4],s=50,c=coulour_mapping1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_aribnb \n(Original) \nin OW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,6],merged_projected[:,5],s=50,c=coulour_mapping_log,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Original)\n in ORLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,6],merged_outlier_projected[:,5],s=50,c=coulour_mapping_log1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Original)\n in OLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,2],merged_projected[:,4],s=50,c=coulour_mapping,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Calculated) \nin ORW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,2],merged_outlier_projected[:,4],s=50,c=coulour_mapping1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_aribnb \n(Calculated) \nin OW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,8],merged_projected[:,5],s=50,c=coulour_mapping_log,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Calculated)\n in ORLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,8],merged_outlier_projected[:,5],s=50,c=coulour_mapping_log1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Calculated)\n in OLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

#Plot the residual plots
plt.subplot(1,2,1)
sns.residplot(merged_projected[:,0],merged_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \nTotal_aribnb (Original) \n in ORW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,0],merged_outlier_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \nTotal_aribnb (Original)\n in OW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_projected[:,6],merged_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \nTotal_aribnb (Original) \n in ORLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,6],merged_outlier_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \nTotal_aribnb (Original)\n in OLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_projected[:,2],merged_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \nTotal_aribnb (Calculated) \n in OW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,2],merged_outlier_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \n Total_aribnb (Calculated) \n in OW/LT')
plt.xlabel('Total_aribnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_projected[:,8],merged_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \nTotal_aribnb (Calculated) \n in OLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,8],merged_outlier_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \n Total_aribnb (Calculated) \n in OLT')
plt.xlabel('Log Transformed Total_aribnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

#Plot the visualisation of PCA with colour mapping of the dataset
plt.subplot(1,2,1)
plt.scatter(merged_projected[:,1],merged_projected[:,4],s=50,c=coulour_mapping,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Original) \nin ORW/LT')
plt.xlabel('Price_Airbnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,1],merged_outlier_projected[:,4],s=50,c=coulour_mapping1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Original) \nin OW/LT')
plt.xlabel('Price_Airbnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,7],merged_projected[:,5],s=50,c=coulour_mapping_log,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Original)\n in ORLT')
plt.xlabel('Log Transformed Price_Airbnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,7],merged_outlier_projected[:,5],s=50,c=coulour_mapping_log1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Total_Airbnb \n(Original)\n in OLT')
plt.xlabel('Log Transformed Price_Airbnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,3],merged_projected[:,4],s=50,c=coulour_mapping,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Calculated) \nin ORW/LT')
plt.xlabel('Price_Airbnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,3],merged_outlier_projected[:,4],s=50,c=coulour_mapping1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Calculated) \nin OW/LT')
plt.xlabel('Price_Airbnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
plt.scatter(merged_projected[:,9],merged_projected[:,5],s=50,c=coulour_mapping_log,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Calculated)\n in ORLT')
plt.xlabel('Log Transformed Price_Airbnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
plt.scatter(merged_outlier_projected[:,9],merged_outlier_projected[:,5],s=50,c=coulour_mapping_log1,linewidths=0,cmap=plt.cm.Blues)
plt.title('Price_house vs Price_Airbnb \n(Calculated)\n in OLT')
plt.xlabel('Log Transformed Price_Airbnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

#Plot the residual plots
plt.subplot(1,2,1)
sns.residplot(merged_projected[:,1],merged_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Original) \n in ORW/LT')
plt.xlabel('Price_airbnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,1],merged_outlier_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Original)\n in OW/LT')
plt.xlabel('Price_airbnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_projected[:,7],merged_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Original) \n in ORLT')
plt.xlabel('Log Transformed Price_airbnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,7],merged_outlier_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Original)\n in OLT')
plt.xlabel('Log Transformed Price_airbnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_projected[:,3],merged_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Calculated) \n in OW/LT')
plt.xlabel('Price_airbnb')
plt.ylabel('Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,3],merged_outlier_projected[:,4],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Calculated) \n in OW/LT')
plt.xlabel('Price_airbnb')
plt.ylabel('Price_house')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_projected[:,9],merged_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Calculated) \n in OLT')
plt.xlabel('Log Transformed Price_airbnb')
plt.ylabel('Log Transformed Price_house')

plt.subplot(1,2,2)
sns.residplot(merged_outlier_projected[:,9],merged_outlier_projected[:,5],lowess=True,color='b')
plt.title('Price_house vs \n Price_airbnb (Calculated) \n in OLT')
plt.xlabel('Log Transformed Price_airbnb')
plt.ylabel('Log Transformed Price_house')
plt.tight_layout()
plt.show()

#Visualise all the features in both O and OR sub-dataframes.
sns.pairplot(merged1,hue='Postcode')
plt.title('Pairplot with outliers removed')

sns.pairplot(merged_with_outlier,hue='Postcode')
plt.title('Pairplot with outliers')

#Plot the residual plots for all sub-dataframes
plt.subplot(1,2,1)
sns.residplot(merged1['log_Total_airbnb_original'],merged1['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Original)\nand Price_house in ORLT')

plt.subplot(1,2,2)
sns.residplot(merged1['log_Total_airbnb_cal'],merged1['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Calculated) \nand Price_house in ORLT')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged1['Total_airbnb_original'],merged1['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Original) \nand Price_house in ORW/LT')

plt.subplot(1,2,2)
sns.residplot(merged1['Total_airbnb_cal'],merged1['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Calculated)\nand Price_house in ORW/LT')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged1['log_Price_airbnb_original'],merged1['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Original)\nand Price_house in ORLT')

plt.subplot(1,2,2)
sns.residplot(merged1['log_Price_airbnb_cal'],merged1['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Calculated)\nand Price_house in ORLT')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged1['Price_airbnb_original'],merged1['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Original)\nand Price_house in ORW/LT')

plt.subplot(1,2,2)
sns.residplot(merged1['Price_airbnb_cal'],merged1['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Calculated) \nand Price_house in ORW/LT')
plt.tight_layout()
plt.show()

#Plot the residual plots for all sub-dataframes
plt.subplot(1,2,1)
sns.residplot(merged_with_outlier['log_Total_airbnb_original'],merged_with_outlier['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Original)\nand Price_house in ORLT')

plt.subplot(1,2,2)
sns.residplot(merged_with_outlier['log_Total_airbnb_cal'],merged_with_outlier['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Calculated) \nand Price_house in ORLT')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_with_outlier['Total_airbnb_original'],merged_with_outlier['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Original) \nand\n Price_house in ORW/LT')

plt.subplot(1,2,2)
sns.residplot(merged_with_outlier['Total_airbnb_cal'],merged_with_outlier['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Total_airbnb (Calculated)\nand\n Price_house in ORW/LT')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_with_outlier['log_Price_airbnb_original'],merged_with_outlier['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Original)\nand Price_house in ORLT')

plt.subplot(1,2,2)
sns.residplot(merged_with_outlier['log_Price_airbnb_cal'],merged_with_outlier['log_Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Calculated)\nand Price_house in ORLT')
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.residplot(merged_with_outlier['Price_airbnb_original'],merged_with_outlier['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Original)\nand\n Price_house in ORW/LT')

plt.subplot(1,2,2)
sns.residplot(merged_with_outlier['Price_airbnb_cal'],merged_with_outlier['Price_house'],lowess=True,color='g')
plt.title('Residual plot for \n Price_airbnb (Calculated) \nand\n Price_house in ORW/LT')
plt.tight_layout()
plt.show()

#Correlation heatmap
correlation  = merged1.corr()
correlation1 = merged_with_outlier.corr()

plt.figure(figsize=(6,6))
sns.heatmap(correlation,vmin=-1,vmax=1,cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.title('Correlation Heatmap in OR')
plt.xticks(horizontalalignment='right')
plt.show()

plt.figure(figsize=(6,6))
sns.heatmap(correlation1,vmin=-1,vmax=1,cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.title('Correlation Heatmap in O')
plt.xticks(horizontalalignment='right')
plt.show()

#Restricting to house price and sort the values from the most to the least
correlation_house = correlation[['log_Price_house']]
correlation_house = correlation_house.sort_values(by='log_Price_house',ascending=False)

correlation_house1 = correlation1[['log_Price_house']]
correlation_house1 = correlation_house1.sort_values(by='log_Price_house',ascending=False)

plt.subplot(1,2,1)
sns.heatmap(correlation_house,vmin=-1,vmax=1,cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.title('Correlation Heatmap \nof OR')

plt.subplot(1,2,2)
sns.heatmap(correlation_house1,vmin=-1,vmax=1,cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.title('Correlation Heatmap \nof O')
plt.tight_layout()
plt.show()

#Drop the is_outliers columns
merged1 = merged1[['Postcode','Total_airbnb_original','Price_airbnb_original','Total_airbnb_cal',
                      'Price_airbnb_cal','Price_house','log_Price_house','log_Total_airbnb_original',
                      'log_Price_airbnb_original','log_Total_airbnb_cal','log_Price_airbnb_cal']]

#Drop the is_outliers columns
merged_with_outlier = merged_with_outlier[['Postcode','Total_airbnb_original','Price_airbnb_original','Total_airbnb_cal',
                      'Price_airbnb_cal','Price_house','log_Price_house','log_Total_airbnb_original',
                      'log_Price_airbnb_original','log_Total_airbnb_cal','log_Price_airbnb_cal']]

#Pearson correlation
pearsoncorr  = merged1.corr(method='pearson')
pearsoncorr1 = merged_with_outlier.corr(method='pearson')

#Plot the Pearson correlation heatmap
plt.subplots(figsize=(20,20))
sns.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)

#Plot the Pearson correlation heatmap
plt.subplots(figsize=(20,20))
sns.heatmap(pearsoncorr1,
            xticklabels=pearsoncorr1.columns,
            yticklabels=pearsoncorr1.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)

#Obtain the p-values of ORLT
pearson_corr,p_val_pearson   = stats.pearsonr(merged1['log_Total_airbnb_original'],merged1['log_Price_house'])
print('Pearson correlation between Total_airbnb (Original) and Price_house is in ORLT is: ',pearson_corr,
      '\n with p value of: ',p_val_pearson)

pearson_corr1,p_val_pearson1 = stats.pearsonr(merged1['log_Price_airbnb_original'],merged1['log_Price_house'])
print('Pearson correlation between Price_airbnb (Original) and Price_house in ORLT is: ',pearson_corr1,
      '\n with p value of: ',p_val_pearson1)

pearson_corr2,p_val_pearson2 = stats.pearsonr(merged1['log_Total_airbnb_cal'],merged1['log_Price_house'])
print('Pearson correlation between Total_airbnb (Calculated) and Price_house in ORLT is: ',pearson_corr2,
      '\n with p value of: ',p_val_pearson2)

pearson_corr3,p_val_pearson3 = stats.pearsonr(merged1['log_Price_airbnb_cal'],merged1['log_Price_house'])
print('Pearson correlation between Price_airbnb (Calculated) and Price_house is in ORLT is: ',pearson_corr3,
      '\n with p value of: ',p_val_pearson3)

#Obtain the p-values of OLT
pearson_corr4,p_val_pearson4 = stats.pearsonr(merged_with_outlier['log_Total_airbnb_original'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Total_airbnb (Original) and Price_house in OLT is: ',pearson_corr4,
      '\n with p value of: ',p_val_pearson4)

pearson_corr5,p_val_pearson5 = stats.pearsonr(merged_with_outlier['log_Total_airbnb_cal'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Total_airbnb (Calculated) and Price_house in OLT is: ',pearson_corr5,
      '\n with p value of: ',p_val_pearson5)

pearson_corr6,p_val_pearson6 = stats.pearsonr(merged_with_outlier['log_Price_airbnb_original'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Price_airbnb (Original) and Price_house in OLT is: ',pearson_corr6,
      '\n with p value of: ',p_val_pearson6)

pearson_corr7,p_val_pearson7 = stats.pearsonr(merged_with_outlier['log_Price_airbnb_cal'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Price_airbnb (Calculated) and Price_house is in OLT is: ',pearson_corr7,
      '\n with p value of: ',p_val_pearson7)

#Spearman correlation
spearmancorr  = merged1.corr(method='spearman')
spearmancorr1 = merged_with_outlier.corr(method='spearman')

#Plot Spearman correlation heatmap
plt.subplots(figsize=(20,20))
sns.heatmap(spearmancorr,
            xticklabels=spearmancorr.columns,
            yticklabels=spearmancorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)

#Plot Spearman correlation heatmap
plt.subplots(figsize=(20,20))
sns.heatmap(spearmancorr1,
            xticklabels=spearmancorr1.columns,
            yticklabels=spearmancorr1.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)

#Obtain p-values of ORLT
spearman_corr,p_val_spearman=stats.spearmanr(merged1['log_Total_airbnb_original'],merged1['log_Price_house'])
print('Pearson correlation between Total_airbnb (Original) and Price_house is in ORLT is: ',spearman_corr,
      '\n with p value of: ',p_val_spearman)

spearman_corr1,p_val_spearman1=stats.spearmanr(merged1['log_Total_airbnb_cal'],merged1['log_Price_house'])
print('Pearson correlation between Total_airbnb (Calculated) and Price_house in ORLT is: ',spearman_corr1,
      '\n with p value of: ',p_val_spearman1)

spearman_corr2,p_val_spearman2=stats.spearmanr(merged1['log_Price_airbnb_original'],merged1['log_Price_house'])
print('Pearson correlation between Price_airbnb (Original) and Price_house in ORLT is: ',spearman_corr2,
      '\n with p value of: ',p_val_spearman2)

spearman_corr3,p_val_spearman3=stats.spearmanr(merged1['log_Price_airbnb_cal'],merged1['log_Price_house'])
print('Pearson correlation between Price_airbnb (Calculated) and Price_house is in ORLT is: ',spearman_corr3,
      '\n with p value of: ',p_val_spearman3)

#Obtain p-values of OLT
spearman_corr4,p_val_spearman4=stats.spearmanr(merged_with_outlier['log_Total_airbnb_original'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Total_airbnb (Original) and Price_house in OLT is: ',spearman_corr4,
      '\n with p value of: ',p_val_spearman4)

spearman_corr5,p_val_spearman5=stats.spearmanr(merged_with_outlier['log_Total_airbnb_cal'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Total_airbnb (Calculated) and Price_house in OLT is: ',spearman_corr5,
      '\n with p value of: ',p_val_spearman5)

spearman_corr6,p_val_spearman6=stats.spearmanr(merged_with_outlier['log_Price_airbnb_original'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Price_airbnb (Original) and Price_house in OLT is: ',spearman_corr6,
      '\n with p value of: ',p_val_spearman6)

spearman_corr7,p_val_spearman7=stats.spearmanr(merged_with_outlier['log_Price_airbnb_cal'],
                                            merged_with_outlier['log_Price_house'])
print('Pearson correlation between Price_airbnb (Calculated) and Price_house is in OLT is: ',spearman_corr7,
      '\n with p value of: ',p_val_spearman7)

#Plot the time series graph for both OLT and ORLT and resample to 3 years
merged1[["log_Price_house", "log_Price_airbnb_original",'log_Price_airbnb_cal']].resample("3y").median().plot(figsize=(15,4))
plt.title('Time Series for House Prices and Airbnb Prices in ORLT with resampling to 3 years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

merged_with_outlier[["log_Price_house", "log_Price_airbnb_original",'log_Price_airbnb_cal']].resample("3y").median().plot(figsize=(15,4))
plt.title('Time Series for House Prices and Airbnb Prices in OLT with resampling to 3 years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

#Plot the rolling windwo time series graph for both OLT and ORLT with length of 24
merged1[["log_Price_house", "log_Price_airbnb_original",'log_Price_airbnb_cal']].rolling(24).mean().plot(figsize=(15,4))
plt.title('Rolling Window Time Series for House Prices and Airbnb Prices in ORLT')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

merged_with_outlier[["log_Price_house", "log_Price_airbnb_original",'log_Price_airbnb_cal']].rolling(24).mean().plot(figsize=(15,4))
plt.title('Rolling Window Time Series for House Prices and Airbnb Prices in OLT')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

#The Pearson correlation is a good place to start to find the global synchrony of two datasets.
#Plot the rolling windwo time series graph with Pearson correlation for both ORLT and OLT
f,ax = plt.subplots(figsize=(15,3))
merged1[["log_Price_house", "log_Price_airbnb_original",'log_Price_airbnb_cal']].rolling(window=30,center=True).median().plot(ax=ax)
ax.set(xlabel='Time',ylabel='Pearson r',title='Time Series with Rolling Window for House Prices and Airbnb Prices in ORLT')
plt.legend(loc='upper left')
plt.show()

f,ax = plt.subplots(figsize=(15,3))
merged_with_outlier[["log_Price_house", "log_Price_airbnb_original",'log_Price_airbnb_cal']].rolling(window=30,center=True).median().plot(ax=ax)
ax.set(xlabel='Time',ylabel='Pearson r',title='Time Series with Rolling Window for House Prices and Airbnb Prices in OLT')
plt.legend(loc='upper left')
plt.show()


# Use pandas own functions to implement a cross correlation function
def crosscorr(datax, datay, lag=0, wrap=False):
 if wrap:
  shiftedy = datay.shift(lag)
  shiftedy.iloc[:lag] = datay.iloc[-lag:].values
  return datax.corr(shiftedy)
 else:
  return datax.corr(datay.shift(lag))

#TLCC for ORLT
d1 = merged1['log_Price_airbnb_original']
d2 = merged1['log_Price_house']

years  = 1
fps    = 50

rs     = [crosscorr(d1,d2,lag) for lag in range(-int(years*fps),int(years*fps+1))]
offset = np.ceil(len(rs)/2)-np.argmax(rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in ORLT \n Offset = {offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

d3 = merged1['log_Price_airbnb_cal']
d4 = merged1['log_Price_house']

rs1     = [crosscorr(d3,d4,lag) for lag in range(-int(years*fps),int(years*fps+1))]
offset1 = np.ceil(len(rs1)/2)-np.argmax(rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(rs1)
ax.axvline(np.ceil(len(rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in ORLT \n Offset = {offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

#TLCC for OLT
d4 = merged_with_outlier['log_Price_airbnb_original']
d5 = merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

rs3     = [crosscorr(d4,d5,lag) for lag in range(-int(years*fps),int(years*fps+1))]
offset3 = np.ceil(len(rs3)/2)-np.argmax(rs3)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(rs3)
ax.axvline(np.ceil(len(rs3)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs3),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in OLT \n Offset = {offset3} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

d6 = merged_with_outlier['log_Price_airbnb_cal']
d7 = merged_with_outlier['log_Price_house']

rs4     = [crosscorr(d6,d7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
offset4 = np.ceil(len(rs4)/2)-np.argmax(rs4)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(rs4)
ax.axvline(np.ceil(len(rs4)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs4),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in OLT \n Offset = {offset4} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

# Split the Postcode areas into East, West, North, North Weset, South East, South West, Central and Outer
East_merged_with_outlier = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'E']
East_merged_with_outlier = East_merged_with_outlier[East_merged_with_outlier['Postcode'].str[1] != 'C']
East_merged_with_outlier = East_merged_with_outlier[East_merged_with_outlier['Postcode'].str[1] != 'N']

if East_merged_with_outlier['Postcode'].all() <= 'E18':
 East_merged_with_outlier['Postcode'] = East_merged_with_outlier['Postcode']
else:
 East_merged_with_outlier['Postcode'] = East_merged_with_outlier['Postcode'].str[0:2]

East_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()

West_merged_with_outlier = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'W']
West_merged_with_outlier = West_merged_with_outlier[West_merged_with_outlier['Postcode'].str[1] != 'C']

West_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()

North_merged_with_outlier = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'N']
NW_merged_with_outlier = North_merged_with_outlier[North_merged_with_outlier['Postcode'].str[1] == 'W']
North_merged = North_merged_with_outlier[North_merged_with_outlier['Postcode'].str[1] != 'W']

North_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()
NW_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()

South_merged_with_outlier = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'S']
SE_merged_with_outlier = South_merged_with_outlier[South_merged_with_outlier['Postcode'].str[1] == 'E']
SW_merged_with_outlier = South_merged_with_outlier[South_merged_with_outlier['Postcode'].str[1] == 'W']

SE_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()
SW_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()

Central_merged_with_outlier = merged_with_outlier[merged_with_outlier['Postcode'].str[1] == 'C']

Central_merged_with_outlier.drop_duplicates(subset=['Postcode']).reset_index()

IG = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'I']
RM = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'R']
EN = merged_with_outlier[merged_with_outlier['Postcode'].str[1] == 'N']
DA = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'D']
BR = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'B']
CR = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'C']
SM = merged_with_outlier[merged_with_outlier['Postcode'].str[1] == 'M']
KT = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'K']
TW = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'T']
UB = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'U']
HA = merged_with_outlier[merged_with_outlier['Postcode'].str[0] == 'H']
WD = merged_with_outlier[merged_with_outlier['Postcode'].str[1] == 'D']

Outer_merged_with_outlier = pd.concat([IG, RM, EN, DA, BR, CR, SM, KT, TW, UB, HA, WD]).reset_index()

e1 = East_merged_with_outlier['log_Price_airbnb_original']
e2 = East_merged_with_outlier['log_Price_airbnb_cal']
e3 = East_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

east_rs     = [crosscorr(e1,e3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
east_offset = np.ceil(len(east_rs)/2)-np.argmax(east_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(east_rs)
ax.axvline(np.ceil(len(east_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(east_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in East London in OLT \nOffset = {east_offset} frames\nE1 leads <> E2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

east_rs1     = [crosscorr(e2,e3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
east_offset1 = np.ceil(len(east_rs1)/2)-np.argmax(east_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(east_rs1)
ax.axvline(np.ceil(len(east_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(east_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in East Londonin OLT \nOffset = {east_offset1} frames\nE1 leads <> E2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

e5 = East_merged['log_Price_airbnb_original']
e6 = East_merged['log_Price_airbnb_cal']
e7 = East_merged['log_Price_house']

years  = 1
fps    = 50

east_rs5     = [crosscorr(e5,e7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
east_offset5 = np.ceil(len(east_rs5)/2)-np.argmax(east_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(east_rs5)
ax.axvline(np.ceil(len(east_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(east_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in East London in ORLT \nOffset = {east_offset5} frames\nE1 leads <> E2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

east_rs6     = [crosscorr(e6,e7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
east_offset6 = np.ceil(len(east_rs6)/2)-np.argmax(east_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(east_rs6)
ax.axvline(np.ceil(len(east_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(east_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in East London in ORLT \nOffset = {east_offset6} frames\nE1 leads <> E2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

w1 = West_merged_with_outlier['log_Price_airbnb_original']
w2 = West_merged_with_outlier['log_Price_airbnb_cal']
w3 = West_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

west_rs     = [crosscorr(w1,w3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
west_offset = np.ceil(len(west_rs)/2)-np.argmax(west_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(west_rs)
ax.axvline(np.ceil(len(west_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(west_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in West London in OLT \nOffset = {west_offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

west_rs1     = [crosscorr(w2,w3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
west_offset1 = np.ceil(len(west_rs1)/2)-np.argmax(west_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(west_rs1)
ax.axvline(np.ceil(len(west_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(west_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in West Londonin OLT \nOffset = {west_offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

w5 = West_merged['log_Price_airbnb_original']
w6 = West_merged['log_Price_airbnb_cal']
w7 = West_merged['log_Price_house']

years  = 1
fps    = 50

west_rs5     = [crosscorr(w5,w7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
west_offset5 = np.ceil(len(west_rs5)/2)-np.argmax(west_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(west_rs5)
ax.axvline(np.ceil(len(west_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(west_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in West London in ORLT \nOffset = {west_offset5} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

west_rs6     = [crosscorr(w6,w7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
west_offset6 = np.ceil(len(west_rs6)/2)-np.argmax(west_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(west_rs6)
ax.axvline(np.ceil(len(west_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(west_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in West London in ORLT \nOffset = {west_offset6} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

n1 = North_merged_with_outlier['log_Price_airbnb_original']
n2 = North_merged_with_outlier['log_Price_airbnb_cal']
n3 = North_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

north_rs     = [crosscorr(n1,n3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
north_offset = np.ceil(len(north_rs)/2)-np.argmax(north_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(north_rs)
ax.axvline(np.ceil(len(north_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(north_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in North London in OLT \nOffset = {north_offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

north_rs1     = [crosscorr(n2,n3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
north_offset1 = np.ceil(len(north_rs1)/2)-np.argmax(north_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(north_rs1)
ax.axvline(np.ceil(len(north_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(north_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in North Londonin OLT \nOffset = {north_offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

n5 = North_merged['log_Price_airbnb_original']
n6 = North_merged['log_Price_airbnb_cal']
n7 = North_merged['log_Price_house']

years  = 1
fps    = 50

north_rs5     = [crosscorr(n5,n7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
north_offset5 = np.ceil(len(north_rs5)/2)-np.argmax(north_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(north_rs5)
ax.axvline(np.ceil(len(north_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(north_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in North London in ORLT \nOffset = {north_offset5} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

north_rs6     = [crosscorr(n6,n7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
north_offset6 = np.ceil(len(north_rs6)/2)-np.argmax(north_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(north_rs6)
ax.axvline(np.ceil(len(north_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(north_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in North London in ORLT \nOffset = {north_offset6} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

nw1 = NW_merged_with_outlier['log_Price_airbnb_original']
nw2 = NW_merged_with_outlier['log_Price_airbnb_cal']
nw3 = NW_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

nw_rs     = [crosscorr(nw1,nw3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
nw_offset = np.ceil(len(nw_rs)/2)-np.argmax(nw_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(nw_rs)
ax.axvline(np.ceil(len(nw_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(nw_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in North West London in OLT \nOffset = {nw_offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

nw_rs1     = [crosscorr(nw2,nw3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
nw_offset1 = np.ceil(len(nw_rs1)/2)-np.argmax(nw_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(nw_rs1)
ax.axvline(np.ceil(len(nw_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(nw_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in North West Londonin OLT \nOffset = {nw_offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

nw5 = North_West_merged['log_Price_airbnb_original']
nw6 = North_West_merged['log_Price_airbnb_cal']
nw7 = North_West_merged['log_Price_house']

years  = 1
fps    = 50

nw_rs5     = [crosscorr(nw5,nw7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
nw_offset5 = np.ceil(len(nw_rs5)/2)-np.argmax(nw_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(nw_rs5)
ax.axvline(np.ceil(len(nw_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(nw_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in North West London in ORLT \nOffset = {nw_offset5} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

nw_rs6     = [crosscorr(nw6,nw7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
nw_offset6 = np.ceil(len(nw_rs6)/2)-np.argmax(nw_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(nw_rs6)
ax.axvline(np.ceil(len(nw_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(nw_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in North West London in ORLT \nOffset = {nw_offset6} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

sw1 = SW_merged_with_outlier['log_Price_airbnb_original']
sw2 = SW_merged_with_outlier['log_Price_airbnb_cal']
sw3 = SW_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

sw_rs     = [crosscorr(sw1,sw3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
sw_offset = np.ceil(len(sw_rs)/2)-np.argmax(sw_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(sw_rs)
ax.axvline(np.ceil(len(sw_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(sw_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in South West London in OLT \nOffset = {sw_offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

sw_rs1     = [crosscorr(sw2,sw3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
sw_offset1 = np.ceil(len(sw_rs1)/2)-np.argmax(sw_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(sw_rs1)
ax.axvline(np.ceil(len(sw_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(sw_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in South West Londonin OLT \nOffset = {sw_offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

sw5 = South_West_merged['log_Price_airbnb_original']
sw6 = South_West_merged['log_Price_airbnb_cal']
sw7 = South_West_merged['log_Price_house']

years  = 1
fps    = 50

sw_rs5     = [crosscorr(sw5,sw7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
sw_offset5 = np.ceil(len(sw_rs5)/2)-np.argmax(sw_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(sw_rs5)
ax.axvline(np.ceil(len(sw_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(sw_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in South West London in ORLT \nOffset = {sw_offset5} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

sw_rs6     = [crosscorr(sw6,sw7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
sw_offset6 = np.ceil(len(sw_rs6)/2)-np.argmax(sw_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(sw_rs6)
ax.axvline(np.ceil(len(sw_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(sw_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in South West London in ORLT \nOffset = {sw_offset6} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

se1 = SE_merged_with_outlier['log_Price_airbnb_original']
se2 = SE_merged_with_outlier['log_Price_airbnb_cal']
se3 = SE_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

se_rs     = [crosscorr(se1,se3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
se_offset = np.ceil(len(se_rs)/2)-np.argmax(se_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(se_rs)
ax.axvline(np.ceil(len(se_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(se_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in South East London in OLT \nOffset = {se_offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

se_rs1     = [crosscorr(se2,se3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
se_offset1 = np.ceil(len(se_rs1)/2)-np.argmax(se_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(se_rs1)
ax.axvline(np.ceil(len(se_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(se_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in South East Londonin OLT \nOffset = {se_offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

se5 = South_East_merged['log_Price_airbnb_original']
se6 = South_East_merged['log_Price_airbnb_cal']
se7 = South_East_merged['log_Price_house']

years  = 1
fps    = 50

se_rs5     = [crosscorr(se5,se7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
se_offset5 = np.ceil(len(se_rs5)/2)-np.argmax(se_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(se_rs5)
ax.axvline(np.ceil(len(se_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(se_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in South East London in ORLT \nOffset = {se_offset5} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

se_rs6     = [crosscorr(se6,se7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
se_offset6 = np.ceil(len(se_rs6)/2)-np.argmax(se_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(se_rs6)
ax.axvline(np.ceil(len(se_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(se_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in South East London in ORLT \nOffset = {se_offset6} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

c1 = Central_merged_with_outlier['log_Price_airbnb_original']
c2 = Central_merged_with_outlier['log_Price_airbnb_cal']
c3 = Central_merged_with_outlier['log_Price_house']

years  = 1
fps    = 50

c_rs     = [crosscorr(c1,c3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
c_offset = np.ceil(len(c_rs)/2)-np.argmax(c_rs)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(c_rs)
ax.axvline(np.ceil(len(c_rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(c_rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in Central London in OLT \nOffset = {c_offset} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

c_rs1     = [crosscorr(c2,c3,lag) for lag in range(-int(years*fps),int(years*fps+1))]
c_offset1 = np.ceil(len(c_rs1)/2)-np.argmax(c_rs1)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(c_rs1)
ax.axvline(np.ceil(len(c_rs1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(c_rs1),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in Central Londonin OLT \nOffset = {c_offset1} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

c5 = Central_merged['log_Price_airbnb_original']
c6 = Central_merged['log_Price_airbnb_cal']
c7 = Central_merged['log_Price_house']

years  = 1
fps    = 50

c_rs5     = [crosscorr(c5,c7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
c_offset5 = np.ceil(len(c_rs5)/2)-np.argmax(c_rs5)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(c_rs5)
ax.axvline(np.ceil(len(c_rs5)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(c_rs5),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Original) vs House Price in Central London in ORLT \nOffset = {c_offset5} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

c_rs6     = [crosscorr(c6,c7,lag) for lag in range(-int(years*fps),int(years*fps+1))]
c_offset6 = np.ceil(len(c_rs6)/2)-np.argmax(c_rs6)

f,ax = plt.subplots(figsize=(14,3))
ax.plot(c_rs6)
ax.axvline(np.ceil(len(c_rs6)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(c_rs6),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Airbnb Price (Calculated) vs House Price in Central London in ORLT \nOffset = {c_offset6} frames\nD1 leads <> D2 leads',
       ylim=[-.2,1],xlim=[0,101],xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 20, 40, 51, 61, 81, 101])
ax.set_xticklabels([-50, -30, -10, 0, 10, 30, 50]);
plt.legend()
plt.show()

#Regression Models
#A simple linear regression will be build first using the most correlated column: OALP from OLT.
#Set Postcode as the index for the dataframe to allow visualisation of each postcode district
merged2 = merged1.set_index('Postcode')
merged_with_outlier1 = merged_with_outlier.set_index('Postcode')

#Set depedent (X) and independent (y) variables
X = pd.DataFrame(merged_with_outlier1['log_Price_house'])
y = pd.DataFrame(merged_with_outlier1['log_Price_airbnb_original'])

#First try the simple linear regression model
lm = sm.OLS(X,y).fit()
linear_predictions = lm.predict()

lm.summary()

print(lm.params)

#Calculating dependent variable values from parameters directly
Dep_value  = merged_with_outlier1['log_Price_airbnb_original']*lm.params['log_Price_airbnb_original']
Dep_value_predict = linear_predictions

#Calculate the residuals 3 ways
residuals_direct  = merged_with_outlier1['log_Price_house']-Dep_value
residuals_predict = merged_with_outlier1['log_Price_house']-Dep_value_predict
residuals_fromlib = lm.resid

f, ax = plt.subplots(1, 3)
ax[0].set_title("Residuals \n(direct)")
ax[0].hist(residuals_direct,20)
ax[1].set_title("Residuals \n(predict)")
ax[1].hist(residuals_predict,20)
ax[2].set_title("Residuals \n(from residuals)")
ax[2].hist(residuals_fromlib,20)
plt.tight_layout()

f,ax = plt.subplots(figsize=(12,8))
f = sm.graphics.influence_plot(lm,ax=ax,criterion='cooks')

print("R-squared:",lm.rsquared)
print("MSE model:",lm.mse_model)
print("MSE residuals:",lm.mse_resid)
print("MSE total:",lm.mse_total)

#Drop the outliers in W1D
Xa = X.drop('W1D')
ya = y.drop('W1D')

lm_a = sm.OLS(Xa,ya).fit()
linear_predictions_a = lm_a.predict()

lm_a.summary()

print(lm_a.params)

print("R-squared:",lm_a.rsquared)
print("MSE model:",lm_a.mse_model)
print("MSE residuals:",lm_a.mse_resid)
print("MSE total:",lm_a.mse_total)

# 10-fold cross-validation linear regression model
model = linear_model.LinearRegression()
scores = []
rmse = []
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X, y)):
 model.fit(X.iloc[train, :], y.iloc[train, :])
 prediction = model.predict(X.iloc[test, :])
 score = r2_score(y.iloc[test, :], prediction)
 scores.append(score)
 test_rmse = (np.sqrt(mean_squared_error(y.iloc[test, :], prediction)))
 rmse.append(test_rmse)

print(scores)
print(sum(scores) / len(scores))
print(rmse)
print(sum(rmse) / len(rmse))

#A MLRM will be build using both OALP and CALP and see if results will improve.
#Set depedent (X) and independent (y) variables
X1 = merged_with_outlier1['log_Price_house'];
y1 = sm.add_constant(merged_with_outlier1[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm1 = sm.OLS(X1,y1).fit()
linear_predictions1 = lm1.predict()

lm1.summary()

f,ax = plt.subplots(figsize=(12,8))
f = sm.graphics.influence_plot(lm1,ax=ax,criterion='cooks')

print(lm1.params)

#Calculating dependent variable values from parameters directly
Dep_value1  = merged_with_outlier1['log_Price_airbnb_original']*lm1.params['log_Price_airbnb_original']+merged_with_outlier1['log_Price_airbnb_cal']*lm1.params['log_Price_airbnb_cal']+lm1.params['const']
Dep_value_predict1 = linear_predictions1

#Calculate the residuals 3 ways
residuals_direct1  = merged_with_outlier1['log_Price_house']-Dep_value1
residuals_predict1 = merged_with_outlier1['log_Price_house']-Dep_value_predict1
residuals_fromlib1 = lm1.resid

f, ax = plt.subplots(1, 3)
ax[0].set_title("Residuals \n(direct)")
ax[0].hist(np.isfinite(residuals_direct1),20)
ax[1].set_title("Residuals \n(predict)")
ax[1].hist(residuals_predict1,20)
ax[2].set_title("Residuals \n(from residuals)")
ax[2].hist(residuals_fromlib1,20)
plt.tight_layout()

residuals_direct_mean  = residuals_direct1.mean()
residuals_predict_mean = residuals_predict1.mean()
residuals_fromlib_mean = residuals_fromlib1.mean()

print(residuals_direct_mean,residuals_predict_mean,residuals_fromlib_mean)

residuals_direct_sum  = residuals_direct1.sum()
residuals_predict_sum = residuals_predict1.sum()
residuals_fromlib_sum = residuals_fromlib1.sum()

print(residuals_direct_sum,residuals_predict_sum,residuals_fromlib_sum)

#create instance of influence
influence = lm1.get_influence()

#leverage (hat values)
leverage = influence.hat_matrix_diag

#Cook's D values (and p-values) as tuple of arrays
cooks_d = influence.cooks_distance

#standardized residuals
studentised_residuals_int = influence.resid_studentized_internal

#studentized residuals
studentised_residuals_ext = influence.resid_studentized_external

print(studentised_residuals_int.mean(),studentised_residuals_int.sum(),studentised_residuals_ext.mean(),studentised_residuals_ext.sum())

#Plot the regression plot
sns.regplot(x=X1,y=linear_predictions1)
plt.title('Regression plot of the Multiple Linear Regression Model')

#Plot the residual plot
sns.residplot(X1,linear_predictions1)
plt.title('Residual Plot of the Multiple Linear Regression Model')

#Apply exponential transform the the price house to allow interpretaion
exp_linear_predictions1 = np.expm1(linear_predictions1)

#Plot the regression plot with exponential transformed house price
sns.regplot(x=X1,y=exp_linear_predictions1)
plt.title('Regression plot of the Multiple Linear Regression Model \n with Reverse Log Transformation')

print("R-squared:",lm1.rsquared)
print("MSE model:",lm1.mse_model)
print("MSE residuals:",lm1.mse_resid)
print("MSE total:",lm1.mse_total)

#Drop the outliers of E12 which has more leverage than W1D
X1 = merged_with_outlier1['log_Price_house'];
y1 = sm.add_constant(merged_with_outlier1[['log_Price_airbnb_original','log_Price_airbnb_cal']])

Xb = X1.drop('E12')
yb = y1.drop('E12')

lm_b = sm.OLS(Xb,yb).fit()
linear_predictions_b = lm_b.predict()

lm_b.summary()

print("R-squared:",lm_b.rsquared)
print("MSE model:",lm_b.mse_model)
print("MSE residuals:",lm_b.mse_resid)
print("MSE total:",lm_b.mse_total)

# 10-fold cross-validated mulitple regression model
X1 = np.asarray(merged_with_outlier1['log_Price_house']).reshape(-1, 1)
y1 = sm.add_constant(merged_with_outlier1[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model1 = linear_model.LinearRegression()
scores1 = []
rmse1 = []
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X1, y1)):
 model1.fit(X1[train, :], y1.iloc[train, :])
 prediction1 = model1.predict(X1[test, :])
 score1 = r2_score(y1.iloc[test, :], prediction1)
 scores1.append(score1)
 test_rmse1 = (np.sqrt(mean_squared_error(y1.iloc[test, :], prediction1)))
 rmse1.append(test_rmse1)

print(scores1)
print(sum(scores1) / len(scores1))
print(rmse1)
print(sum(rmse1) / len(rmse1))

#Set dependent (X) and independent (y) variables
X2 = merged2['log_Price_house'];
y2 = sm.add_constant(merged2[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm2 = sm.OLS(X2,y2).fit()
linear_predictions2 = lm2.predict()

lm2.summary()

f,ax = plt.subplots(figsize=(12,8))
f = sm.graphics.influence_plot(lm2,ax=ax,criterion='cooks')

print(lm2.params)

#Calculating dependent variable values from parameters directly
Dep_value2  = merged2['log_Price_airbnb_original']*lm1.params['log_Price_airbnb_original']+merged2['log_Price_airbnb_cal']*lm1.params['log_Price_airbnb_cal']+lm1.params['const']
Dep_value_predict2 = linear_predictions2

#Calculate the residuals 3 ways
residuals_direct2  = merged2['log_Price_house']-Dep_value2
residuals_predict2 = merged2['log_Price_house']-Dep_value_predict2
residuals_fromlib2 = lm2.resid

f, ax = plt.subplots(1, 3)
ax[0].set_title("Residuals \n(direct)")
ax[0].hist(residuals_direct2,20)
ax[1].set_title("Residuals \n(predict)")
ax[1].hist(residuals_predict2,20)
ax[2].set_title("Residuals \n(from residuals)")
ax[2].hist(residuals_fromlib2,20)
plt.tight_layout()

residuals_direct_mean1  = residuals_direct2.mean()
residuals_predict_mean1 = residuals_predict2.mean()
residuals_fromlib_mean1 = residuals_fromlib2.mean()

print(residuals_direct_mean1,residuals_predict_mean1,residuals_fromlib_mean1)

residuals_direct_sum1  = residuals_direct2.sum()
residuals_predict_sum1 = residuals_predict2.sum()
residuals_fromlib_sum1 = residuals_fromlib2.sum()

print(residuals_direct_sum1,residuals_predict_sum1,residuals_fromlib_sum1)

#create instance of influence
influence1 = lm2.get_influence()

#leverage (hat values)
leverage1 = influence1.hat_matrix_diag

#Cook's D values (and p-values) as tuple of arrays
cooks_d1 = influence1.cooks_distance

#standardized residuals
studentised_residuals_int1 = influence1.resid_studentized_internal

#studentized residuals
studentised_residuals_ext1 = influence1.resid_studentized_external

print(studentised_residuals_int1.mean(),studentised_residuals_int1.sum(),studentised_residuals_ext1.mean(),studentised_residuals_ext1.sum())

#Plot the regression plot
sns.regplot(x=X2,y=linear_predictions2)
plt.title('Regression plot of the Multiple Linear Regression Model')

#Plot the residual plot
sns.residplot(X2,linear_predictions2)
plt.title('Residual Plot of the Multiple Linear Regression Model')

#Exponentially transform the house prices
exp_linear_predictions2 = np.expm1(linear_predictions2)

#Plot the regression plot with exponentially transformed house prices
sns.regplot(x=X2,y=exp_linear_predictions2)
plt.title('Regression plot of the Multiple Linear Regression Model \n with Reverse Log Transformation')

print("R-squared:",lm2.rsquared)
print("MSE model:",lm2.mse_model)
print("MSE residuals:",lm2.mse_resid)
print("MSE total:",lm2.mse_total)

# 10-fold cross-validated multiple regression model
X2 = np.asarray(merged2['log_Price_house']).reshape(-1, 1)
y2 = sm.add_constant(merged2[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model2 = linear_model.LinearRegression()
scores = []
rmse = []
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(X2, y2)):
 model.fit(X2[train, :], y2.iloc[train, :])
 prediction = model.predict(X2[test, :])
 score = r2_score(y2.iloc[test, :], prediction)
 scores.append(score)
 test_rmse = (np.sqrt(mean_squared_error(y2.iloc[test, :], prediction)))
 rmse.append(test_rmse)

print(scores)
print(sum(scores) / len(scores))
print(rmse)
print(sum(rmse) / len(rmse))

#Therefore, ORLT will be used to build regression models for each postcode districts
Xe = East_merged['log_Price_house']
ye = sm.add_constant(East_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_e = sm.OLS(Xe,ye).fit()
linear_predictions_e = lm_e.predict()

lm_e.summary()

Xe = np.asarray(East_merged['log_Price_house']).reshape(-1, 1)
ye = sm.add_constant(East_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_e = linear_model.LinearRegression()
scores_e = []
rmse_e = []
kfold_e = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold.split(Xe, ye)):
 model_e.fit(Xe[train, :], ye.iloc[train, :])
 prediction_e = model_e.predict(Xe[test, :])
 score_e = r2_score(ye.iloc[test, :], prediction_e)
 scores_e.append(score_e)
 test_rmse_e = (np.sqrt(mean_squared_error(ye.iloc[test, :], prediction_e)))
 rmse_e.append(test_rmse_e)

print(sum(scores_e) / len(scores))
print(sum(rmse_e) / len(rmse))

exp_linear_predictions_e = np.expm1(linear_predictions_e)

sns.regplot(x=Xe,y=exp_linear_predictions_e)
plt.title('Regression plot of the Multiple Linear Regression Model \n in East London')

Xn = North_merged['log_Price_house']
yn = sm.add_constant(North_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_n = sm.OLS(Xn,yn).fit()
linear_predictions_n = lm_n.predict()

lm_n.summary()

Xn = np.array(North_merged['log_Price_house']).reshape(-1, 1)
yn = sm.add_constant(North_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_n = linear_model.LinearRegression()
scores_n = []
rmse_n = []
kfold_n = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold_n.split(Xn, yn)):
 model_n.fit(Xn[train, :], yn.iloc[train, :])
 prediction_n = model_n.predict(Xn[test, :])
 score_n = r2_score(yn.iloc[test, :], prediction_n)
 scores_n.append(score_n)
 test_rmse_n = (np.sqrt(mean_squared_error(yn.iloc[test, :], prediction_n)))
 rmse_n.append(test_rmse_n)

print(sum(scores_n) / len(scores_n))
print(sum(rmse_n) / len(rmse_n))

exp_linear_predictions_n = np.expm1(linear_predictions_n)

sns.regplot(x=Xn,y=exp_linear_predictions_n)
plt.title('Regression plot of the Multiple Linear Regression Model \n in North London')

Xc = Central_merged['log_Price_house'];
yc = sm.add_constant(Central_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_c = sm.OLS(Xc,yc).fit()
linear_predictions_c = lm_c.predict()

lm_c.summary()

Xc = np.asarray(Central_merged['log_Price_house']).reshape(-1, 1)
yc = sm.add_constant(Central_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_c = linear_model.LinearRegression()
scores_c = []
rmse_c = []
kfold_c = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold_c.split(Xc, yc)):
 model_c.fit(Xc[train, :], yc.iloc[train, :])
 prediction_c = model_c.predict(Xc[test, :])
 score_c = r2_score(yc.iloc[test, :], prediction_c)
 scores_c.append(score_c)
 test_rmse_c = (np.sqrt(mean_squared_error(yc.iloc[test, :], prediction_c)))
 rmse_c.append(test_rmse_c)

print(sum(scores_c) / len(scores_c))
print(sum(rmse_c) / len(rmse_c))

exp_linear_predictions_c = np.expm1(linear_predictions_c)

sns.regplot(x=Xc,y=exp_linear_predictions_c)
plt.title('Regression plot of the Multiple Linear Regression Model \n in Central London')

Xw = West_merged['log_Price_house'];
yw = sm.add_constant(West_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_w = sm.OLS(Xw,yw).fit()
linear_predictions_w = lm_w.predict()

lm_w.summary()

Xw = np.asarray(West_merged['log_Price_house']).reshape(-1, 1)
yw = sm.add_constant(West_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_w = linear_model.LinearRegression()
scores_w = []
rmse_w = []
kfold_w = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold_w.split(Xw, yw)):
 model_w.fit(Xw[train, :], yw.iloc[train, :])
 prediction_w = model_w.predict(Xw[test, :])
 score_w = r2_score(yw.iloc[test, :], prediction_w)
 scores_w.append(score_w)
 test_rmse_w = (np.sqrt(mean_squared_error(yw.iloc[test, :], prediction_w)))
 rmse_w.append(test_rmse_w)

print(sum(scores_w) / len(scores_w))
print(sum(rmse_w) / len(rmse_w))

exp_linear_predictions_w = np.expm1(linear_predictions_w)

sns.regplot(x=Xw,y=exp_linear_predictions_w)
plt.title('Regression plot of the Multiple Linear Regression Model \n in West London')

Xnw = North_West_merged['log_Price_house'];
ynw = sm.add_constant(North_West_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_nw = sm.OLS(Xnw,ynw).fit()
linear_predictions_nw = lm_nw.predict()

lm_nw.summary()

Xnw = np.asarray(North_West_merged['log_Price_house']).reshape(-1, 1)
ynw = sm.add_constant(North_West_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_nw = linear_model.LinearRegression()
scores_nw = []
rmse_nw = []
kfold_nw = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold_nw.split(Xnw, ynw)):
 model_nw.fit(Xnw[train, :], ynw.iloc[train, :])
 prediction_nw = model_nw.predict(Xnw[test, :])
 score_nw = r2_score(ynw.iloc[test, :], prediction_nw)
 scores_nw.append(score_nw)
 test_rmse_nw = (np.sqrt(mean_squared_error(ynw.iloc[test, :], prediction_nw)))
 rmse_nw.append(test_rmse_nw)

print(sum(scores_nw) / len(scores_nw))
print(sum(rmse_nw) / len(rmse_nw))

exp_linear_predictions_nw = np.expm1(linear_predictions_nw)

sns.regplot(x=Xnw,y=exp_linear_predictions_nw)
plt.title('Regression plot of the Multiple Linear Regression Model \n in North West London')

Xsw = South_West_merged['log_Price_house'];
ysw = sm.add_constant(South_West_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_sw = sm.OLS(Xsw,ysw).fit()
linear_predictions_sw = lm_sw.predict()

lm_sw.summary()

Xsw = np.asarray(South_West_merged['log_Price_house']).reshape(-1, 1)
ysw = sm.add_constant(South_West_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_sw = linear_model.LinearRegression()
scores_sw = []
rmse_sw = []
kfold_sw = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold_sw.split(Xsw, ysw)):
 model_sw.fit(Xsw[train, :], ysw.iloc[train, :])
 prediction_sw = model_sw.predict(Xsw[test, :])
 score_sw = r2_score(ysw.iloc[test, :], prediction_sw)
 scores_sw.append(score_sw)
 test_rmse_sw = (np.sqrt(mean_squared_error(ysw.iloc[test, :], prediction_sw)))
 rmse_sw.append(test_rmse_sw)

print(sum(scores_sw) / len(scores_sw))
print(sum(rmse_sw) / len(rmse_sw))

exp_linear_predictions_sw = np.expm1(linear_predictions_sw)

sns.regplot(x=Xw,y=exp_linear_predictions_w)
plt.title('Regression plot of the Multiple Linear Regression Model \n in South West London')

Xse = South_East_merged['log_Price_house'];
yse = sm.add_constant(South_East_merged[['log_Price_airbnb_original','log_Price_airbnb_cal']])

lm_se = sm.OLS(Xse,yse).fit()
linear_predictions_se = lm_se.predict()

lm_se.summary()

Xse = np.asarray(South_East_merged['log_Price_house']).reshape(-1, 1)
yse = sm.add_constant(South_East_merged[['log_Price_airbnb_original', 'log_Price_airbnb_cal']])

model_se = linear_model.LinearRegression()
scores_se = []
rmse_se = []
kfold_se = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (train, test) in enumerate(kfold_se.split(Xse, yse)):
 model_se.fit(Xse[train, :], yse.iloc[train, :])
 prediction_se = model_se.predict(Xse[test, :])
 score_se = r2_score(yse.iloc[test, :], prediction_se)
 scores_se.append(score_se)
 test_rmse_se = (np.sqrt(mean_squared_error(yse.iloc[test, :], prediction_se)))
 rmse_se.append(test_rmse_se)

print(sum(scores_se) / len(scores_se))
print(sum(rmse_se) / len(rmse_se))

exp_linear_predictions_se = np.expm1(linear_predictions_se)

sns.regplot(x=Xse,y=exp_linear_predictions_se)
plt.title('Regression plot of the Multiple Linear Regression Model \n in South East London')













