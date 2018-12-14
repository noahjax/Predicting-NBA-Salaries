import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

#Load Players
players = pd.read_csv('Data/Raw Data/Players.csv')
players = players.loc[:,['Player','height', 'weight']]

#Load season stats
season_stats = pd.read_csv('Data/Raw Data/Seasons_Stats.csv')
season_stats = season_stats.loc[:, ['Year', 'Player', 'Age', 'Pos', 'G', 'GS', 'MP', 'PER', 
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB',
    'TRB', 'AST','STL','BLK','TOV','PF','PTS','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%',
    'USG%','WS']]

season_stats = season_stats.loc[season_stats.Year >= 1990]
season_stats = season_stats.loc[season_stats.Year <= 2017]
season_stats = season_stats.dropna(axis=0)

#Parse out positions
season_stats['isPG'] = season_stats.apply(lambda row: 'PG' in row['Pos'], axis=1)
season_stats['isSG'] = season_stats.apply(lambda row: 'SG' in row['Pos'], axis=1)
season_stats['isSF'] = season_stats.apply(lambda row: 'SF' in row['Pos'], axis=1)
season_stats['isPF'] = season_stats.apply(lambda row: 'PF' in row['Pos'], axis=1)
season_stats['isC'] = season_stats.apply(lambda row: 'C' in row['Pos'], axis=1)


#Load salary info
salaries = pd.read_csv('Data/Raw Data/nba_salaries_1990_to_2018.csv')
salaries = salaries.loc[:, ['player','salary','season_start']]
salaries = salaries.rename(index=str, columns={'player':'Player','season_start':'Year'})
#Filter to only the desired years
salaries = salaries.loc[salaries.Year >= 1990]
salaries = salaries.loc[salaries.Year <= 2017]

#Merge datasets, clean a bit
merged = pd.merge(season_stats, salaries, how='inner', left_on=['Player','Year'], right_on=['Player','Year'])
merged = pd.merge(merged, players, how='inner', left_on=['Player'], right_on='Player')
merged = merged.dropna(axis=0)

#Merging for players who were traded that year
merged['Weighted PER'] = merged.apply(lambda row: row['MP']*row['PER'], axis=1)
merged['Weighted ORB%'] = merged.apply(lambda row: row['MP']*row['ORB%'], axis=1)
merged['Weighted DRB%'] = merged.apply(lambda row: row['MP']*row['DRB%'], axis=1)
merged['Weighted TRB%'] = merged.apply(lambda row: row['MP']*row['TRB%'], axis=1)
merged['Weighted AST%'] = merged.apply(lambda row: row['MP']*row['AST%'], axis=1)
merged['Weighted STL%'] = merged.apply(lambda row: row['MP']*row['STL%'], axis=1)
merged['Weighted BLK%'] = merged.apply(lambda row: row['MP']*row['BLK%'], axis=1)
merged['Weighted TOV%'] = merged.apply(lambda row: row['MP']*row['TOV%'], axis=1)
merged['Weighted USG%'] = merged.apply(lambda row: row['MP']*row['USG%'], axis=1)
merged['Weighted WS'] = merged.apply(lambda row: row['G']*row['WS'], axis=1)

grouped = merged.groupby(by=['Year','Player'], sort=False)
sum_agg = grouped['G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P','2PA','FT','FTA','ORB','DRB',
    'TRB', 'AST','STL','BLK','TOV','PF','PTS','Weighted PER','Weighted ORB%', 'Weighted DRB%',
    'Weighted TRB%','Weighted AST%','Weighted STL%','Weighted BLK%','Weighted TOV%', 'Weighted USG%',
    'Weighted WS'].agg(np.sum)
mean_agg = grouped['salary','height','weight','isPG','isSG','isSF','isPF','isC'].agg(np.mean)

sum_agg['2P%'] = sum_agg.apply(lambda row: row['2P']/row['2PA'], axis=1)    
sum_agg['3P%'] = sum_agg.apply(lambda row: row['3P']/row['3PA'], axis=1)    
sum_agg['FT%'] = sum_agg.apply(lambda row: row['FT']/row['FTA'], axis=1)    
sum_agg['PER'] = sum_agg.apply(lambda row: row['Weighted PER']/row['MP'], axis=1)  
sum_agg['TS%'] = sum_agg.apply(lambda row: row['PTS']/(2*(row['FGA'] + .44*row['FTA'])), axis=1)  
sum_agg['ORB%'] = sum_agg.apply(lambda row: row['Weighted ORB%']/row['MP'], axis=1)  
sum_agg['DRB%'] = sum_agg.apply(lambda row: row['Weighted DRB%']/row['MP'], axis=1)  
sum_agg['TRB%'] = sum_agg.apply(lambda row: row['Weighted TRB%']/row['MP'], axis=1)  
sum_agg['AST%'] = sum_agg.apply(lambda row: row['Weighted AST%']/row['MP'], axis=1)  
sum_agg['STL%'] = sum_agg.apply(lambda row: row['Weighted STL%']/row['MP'], axis=1)  
sum_agg['BLK%'] = sum_agg.apply(lambda row: row['Weighted BLK%']/row['MP'], axis=1)  
sum_agg['TOV%'] = sum_agg.apply(lambda row: row['Weighted TOV%']/row['MP'], axis=1)  
sum_agg['USG%'] = sum_agg.apply(lambda row: row['Weighted USG%']/row['MP'], axis=1)  
sum_agg['WS'] = sum_agg.apply(lambda row: row['Weighted WS']/row['G'], axis=1)  
sum_agg = sum_agg.drop(['Weighted PER','Weighted ORB%', 'Weighted DRB%',
    'Weighted TRB%','Weighted AST%','Weighted STL%','Weighted BLK%','Weighted TOV%',
    'Weighted USG%','Weighted WS'], axis=1)  

final_data = pd.merge(sum_agg, mean_agg, how='inner', left_on=['Player','Year'], right_on=['Player','Year'])
final_data = final_data.reset_index()
final_data = final_data.drop('Player', axis=1)

#Split into train, dev and test sets
rand=14
train, test = train_test_split(final_data, test_size=0.2, random_state=rand)
train, dev = train_test_split(train, test_size=.25, random_state=rand)

#Save min_year and scale factor for salary
min_year = train['Year'].min()
max_year = train['Year'].max()
year_group = train.groupby(by='Year')
yearly_salary = year_group['salary'].agg(np.mean).reset_index()
scale_factor = (yearly_salary['salary'].max()/yearly_salary['salary'].min())**(1/(max_year-min_year))
np.savetxt('Data/min_year_scale_factor.csv', np.array([min_year, scale_factor]))

# print(train.shape)
train_x = train.drop(['salary'], axis=1)
train_y = train.loc[:,['salary']]
train_x.to_csv('Data/train_x.csv')
train_y.to_csv('Data/train_y.csv')

dev_x = dev.drop(['salary'], axis=1)
dev_y = dev.loc[:,['salary']]
dev_x.to_csv('Data/dev_x.csv')
dev_y.to_csv('Data/dev_y.csv')

test_x = test.drop(['salary'], axis=1)
test_y = test.loc[:,['salary']]
test_x.to_csv('Data/test_x.csv')
test_y.to_csv('Data/test_y.csv')

