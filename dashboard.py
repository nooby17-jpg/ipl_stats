import streamlit as st
import glob
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objects as go
from PIL import Image
import numpy as np
# Load data
matches =  pd.read_csv("IPL_Matches_2008_2022.csv")
# Count the number of wins for each team
most_wins = matches['WinningTeam'].value_counts().sort_values(ascending=False)

# Create a new DataFrame with 'team' as the index and 'Wins' as the column
most_wins_df = pd.DataFrame({'Wins': most_wins})
most_wins_df.index.name = 'Team'
most_wins_df.reset_index(inplace=True)
most_wins_df.rename(columns={'index': 'Team'}, inplace=True)
most_wins_df.reset_index(drop=True, inplace=True)


image = Image.open('ipl-logo.png')
st.image(image, width=200)
st.title('Indian Premier League')
st.subheader('Statistics & Analysis')

#Plotly - Visualisation Function (BAR PLOT)
def generate_plot(x,y, title,x_title, y_title):
    fig = go.Figure(go.Bar(
                x=x.values,
                y=y.values,
                marker_color='#1D267D'
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(
            family="sans-serif",
            color="#ffffff"
        )
    )
    st.plotly_chart(fig)

#Display data in Dataframe
def show_df(data):
    st.dataframe(data, use_container_width=True)


#Display the Most Wins Section
def most_wins_plot(data, title, x, y):
    fig = go.Figure(go.Bar(
                    x=data.index,
                    y=data.values,
                    marker_color='#1D267D'
                ))

    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(
            family="sans-serif",
            color="#ffffff"
        )
    )
    st.plotly_chart(fig)






# Define the directory path where the CSV files are located
directory = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Runs'

# Get a list of all CSV file paths in the directory
csv_files = glob.glob(directory + '/*.csv')

# Create an empty dictionary to store the total and mean values for each player
batting_stats = {}

# Iterate through each CSV file
for file in csv_files:
    # Read the CSV file and extract the required columns
    data = pd.read_csv(file)
    columns = ['Player', 'Mat', 'Inns','Runs', 'NO', 'BF', '100', '50', '4s', '6s', 'Avg', 'SR', 'HS']
    subset_data = data[columns]

    # Iterate through each row and update the batting_stats dictionary
    for row in subset_data.itertuples(index=False):
        player = row[0]
        if player in batting_stats:
            # Update the values for the player
            batting_stats[player][0] += row[1]
            batting_stats[player][1] += row[2]
            batting_stats[player][2] += row[3]
            batting_stats[player][3] += row[4]
            batting_stats[player][4] += row[5]
            batting_stats[player][5] += row[6]
            batting_stats[player][6] += row[7]
            batting_stats[player][7] += row[8]
            batting_stats[player][8] += row[9]
            if row[10] != '-' and pd.notna(row[10]):  # Check if 'Avg' is not '-' or NaN
                batting_stats[player][9].append(float(row[10]))  # Convert 'Avg' to float and append

            if pd.notna(row[11]):  # Check if 'SR' is not NaN
                batting_stats[player][10].append(float(row[11]))  # Convert 'SR' to float and append
            batting_stats[player][11] = max(batting_stats[player][11], row[12])
        else:
            # Add a new entry for the player
            batting_stats[player] = [row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], [], [], row[12]]

# Calculate the mean values for 'Avg' and 'SR' for each player
for player in batting_stats:
    avg_values = batting_stats[player][9]
    sr_values = batting_stats[player][10]
    avg_mean = np.mean(avg_values) if avg_values else np.nan
    sr_mean = np.mean(sr_values) if sr_values else np.nan
    batting_stats[player][9] = avg_mean
    batting_stats[player][10] = sr_mean

# Convert the batting_stats dictionary to a DataFrame
columns = ['Mat', 'Inns','Runs', 'NO', 'BF', '100', '50', '4s', '6s', 'Avg', 'SR', 'HS']
batting_stats_df = pd.DataFrame.from_dict(batting_stats, orient='index', columns=columns)
# Reset index and rename the 'index' column to 'Player'
batting_stats_df.reset_index(inplace=True)
batting_stats_df.rename(columns={'index': 'Player'}, inplace=True)
batting_stats_df.reset_index(drop=True, inplace=True)
batting_stats_df.set_index('Player', inplace=True)
    




# Define the directory path where the CSV files are located
directory_bowl = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Wickets'

# Get a list of all CSV file paths in the directory
csv_files_bowl = glob.glob(directory_bowl + '/*.csv')

# Create an empty dictionary to store the total and mean values for each player
bowling_stats = {}

# Iterate through each CSV file
for file in csv_files_bowl:
    # Read the CSV file and extract the required columns
    data = pd.read_csv(file)
    columns = ['Player', 'Mat', 'Inns', 'Ov', 'Runs', 'Wkts', '4w', '5w', 'Avg', 'Econ', 'SR','BBI']
    subset_data = data[columns]

    # Iterate through each row and update the bowling_stats dictionary
    for row in subset_data.itertuples(index=False):
        player = row[0]
        if player in bowling_stats:
            # Update the values for the player
            bowling_stats[player][0] += row[1]
            bowling_stats[player][1] += row[2]
            bowling_stats[player][2] += row[3]
            bowling_stats[player][3] += row[4]
            bowling_stats[player][4] += row[5]
            bowling_stats[player][5] += row[6]
            bowling_stats[player][6] += row[7]
            if row[8] != '-' and pd.notna(row[8]):  # Check if 'Avg' is not '-' or NaN
                bowling_stats[player][7].append(float(row[8]))  # Convert 'Avg' to float and append
            if pd.notna(row[9]):  # Check if 'Econ' is not NaN
                bowling_stats[player][8].append(float(row[9]))  # Convert 'Econ' to float and append
            if pd.notna(row[10]):  # Check if 'SR' is not NaN
                if isinstance(bowling_stats[player][9], list):
                    bowling_stats[player][9].append(float(row[10]))  # Convert 'SR' to float and append
                else:
                    bowling_stats[player][9] = [float(row[10])]  # Create a new list with the 'SR' value
            bowling_stats[player][10] = max(bowling_stats[player][10], row[11])
        else:
            # Add a new entry for the player
            bowling_stats[player] = [row[1], row[2], row[3], row[4], row[5], row[6], row[7], [], [], [], row[11]]

# Calculate the mean values for 'Avg', 'Econ', and 'SR' for each player
for player in bowling_stats:
    avg_values = bowling_stats[player][7]
    econ_values = bowling_stats[player][8]
    sr_values = bowling_stats[player][9]

    avg_mean = np.mean(avg_values) if avg_values else np.nan
    econ_mean = np.mean(econ_values) if econ_values else np.nan
    sr_mean = np.mean(sr_values) if sr_values else np.nan
    bowling_stats[player][7] = avg_mean
    bowling_stats[player][8] = econ_mean
    bowling_stats[player][9] = sr_mean

# Convert the bowling_stats dictionary to a DataFrame
columns = ['Mat', 'Inns', 'Ov', 'Runs', 'Wkts', '4w', '5w', 'Avg', 'Econ', 'SR','BBI']
bowling_stats_df = pd.DataFrame.from_dict(bowling_stats, orient='index', columns=columns)
# Reset index and rename the 'index' column to 'Player'
bowling_stats_df.reset_index(inplace=True)
bowling_stats_df.rename(columns={'index': 'Player'}, inplace=True)
bowling_stats_df.reset_index(drop=True, inplace=True)
bowling_stats_df.set_index('Player', inplace=True)



# filters
col1, col2 = st.columns(2)


with col1:

    year = st.selectbox('Season', ['Select Year', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'])
with col2:
    team = st.selectbox('Team', ['All Teams', 'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Kings XI Punjab', 'Rajasthan Royals', 'Deccan Chargers', 'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Gujarat Lions', 'Rising Pune Supergiant', 'Delhi Capitals'])






col3, col4 = st.columns(2)

with col3:
    filter_options = st.radio(
    "Filter option", ('BATTING', 'BOWLING'), 
    horizontal=True)

with col4: 
    if filter_options == 'BATTING':
        filter_batter = st.selectbox('Batting', ['HIGHEST RUNS', 'MOST SIXES', 'MOST FOURS', 'MOST SIXES (INNINGS)', 'MOST FOURS (INNINGS)', 'MOST FIFTIES', 'MOST CENTURIES', 'FASTEST FIFTIES', 'FASTEST CENTURIES'])
    else:
        filter_bowler = st.selectbox('Bowling', ['HIGHEST WICKETS', 'MOST DOT BALLS', 'MOST DOT BALLS (INNINGS)', 'BEST BOWLING AVERAGE', 'BEST BOWLING ECONOMY', 'BEST BOWLING ECONOMY (INNINGS)', 'BEST BOWLING STRIKE RATE', 'BEST BOWLING STRIKE RATE (INNINGS)', 'MOST RUNS CONCEDED (INNINGS)'])

#Short Feature for display
stats_feature = ['Team1', 'Team2','City', 'Venue','Date','WinningTeam','WonBy','Margin','Player_of_Match']
run_feature = ['Player','Mat', 'Inns', 'SR','Avg', '100','50','HS','Runs']
eco_feature = ['Player', 'Ov', 'Runs', 'Wkts', 'Dots', 'Econ', 'Against']
bowling_feature = ['Mat', 'Wkts', 'Avg', 'Econ','SR', 'BBI']
batting_feature = ['Mat', 'Inns', 'SR','Avg', '100','50','Runs']
global_six=['Mat','Inns', 'Runs','SR','6s']
global_four =['Mat','Inns', 'Runs','SR','4s']


#Conditional Statement Start
if year != 'Select Year':
    #This code will show the output when the value of year is selected as one of the season
    st.header(f"Indian Premier League {year}")
    data = matches[matches['Season'] == year]
    
    if team != 'All Teams':
        #This code will show the output when the value of year is selected as one of the season and the value of team is selected as one of team
        data = data[(data['Team1'] == team) | (data['Team2'] == team)]
        show_df(data[stats_feature])
       
    else:
        #This code will show the output when the value of team is selected as 'ALL Teams' and the value of year is selected as one of the seasons
        if filter_options == 'BATTING':
            if filter_batter == 'HIGHEST RUNS':
                #Most Runs in IPL Chart and Data
                st.subheader(f'{filter_batter}')
                run_path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Runs/Most Runs - ' + year + '.csv'
                most_runs = pd.read_csv(run_path)
                show_df(most_runs[run_feature].head(10))
                generate_plot(x=most_runs['Player'].head(10),y=most_runs['Runs'].head(10), title=f"Most Runs in IPL {year}", x_title=' ', y_title='Runs')
            elif filter_batter == 'MOST SIXES':
                # Most Sixes in IPL Chart and Data
                st.subheader(f'{filter_batter}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Sixes Innings/Most Sixes Innings - ' + year + '.csv'
                sixes = pd.read_csv(path)
                most_six = sixes.groupby('Player')['6s'].sum().sort_values(ascending=False)
                show_df(most_six.head(10))

                most_wins_plot(data=most_six.head(10), title=f"Most Sixes by a player in IPL - {year}", x='Sixes', y='Players')
            elif filter_batter == 'MOST FOURS':
                #Most Fours in IPL Chart and Data
                st.subheader(f'{filter_batter}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Fours Innings/Most Fours Innings - ' + year + '.csv'
                fours = pd.read_csv(path)
                most_four = fours.groupby('Player')['4s'].sum().sort_values(ascending=False)
                show_df(most_four.head(10))
                most_wins_plot(data=most_four.head(10), title=f"Most Fours by a player in IPL - {year}", x='FOurs', y='Players')
            elif filter_batter == 'MOST SIXES (INNINGS)':
                st.subheader(f'{filter_batter}')
                #Most Sixes (INNINGS) in IPL Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Sixes Innings/Most Sixes Innings - ' + year + '.csv'
                six_feature = ['POS','Player','Runs','SR','6s','Against']
                most_six_innings = pd.read_csv(path)
                show_df(most_six_innings[six_feature].head(10))
                generate_plot(x=most_six_innings['Player'].head(10),y=most_six_innings['6s'].head(10), title=f"Highest Sixes in single Innings in IPL - {year}", x_title=' ', y_title='Sixes')
            elif filter_batter == 'MOST FOURS (INNINGS)':
                st.subheader(f'{filter_batter}')
                #Most Runs in IPL Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Fours Innings/Most Fours Innings - ' + year + '.csv'
                four_feature = ['POS','Player','Runs','SR','4s','Against']
                most_four_innings = pd.read_csv(path)
                show_df(most_four_innings[four_feature].head(10))
                generate_plot(x=most_four_innings['Player'].head(10),y=most_four_innings['4s'].head(10), title=f"Highest Fours in single Innings in IPL - {year}", x_title=' ', y_title='Sixes')
            elif filter_batter == 'MOST FIFTIES':
                st.subheader(f'{filter_batter}')
                #Most Fifties in IPL Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Fastest Fifties/Fastest Fifties - ' + year + '.csv'
                fifties = pd.read_csv(path)
                most_fifties = fifties['Player'].value_counts().sort_values(ascending=False)
                show_df(most_fifties.head(10))

                most_wins_plot(data=most_fifties, title=f'Most Fifities in IPL {year}', x='Players', y='Fifties')
            elif filter_batter == 'MOST CENTURIES':
                st.subheader(f'{filter_batter}')
                 #Most Centuries in IPL Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Fastest Centuries/Fastest Centuries - ' + year + '.csv'
                centuries = pd.read_csv(path)
                most_centuries = centuries['Player'].value_counts().sort_values(ascending=False)
                show_df(most_centuries.head(10))
                most_wins_plot(data=most_centuries, title=f'Most Centuries in IPL {year}', x='Players', y='Centuries')
            elif filter_batter == 'FASTEST FIFTIES':
                st.subheader(f'{filter_batter}')
                #Fastest Fifties in IPL Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Fastest Fifties/Fastest Fifties - ' + year + '.csv'
                fastest_feature = ['POS','Player','Runs','BF', 'Against']
                fastest_fifties = pd.read_csv(path)
                show_df(fastest_fifties[fastest_feature].head(10))
                generate_plot(x=fastest_fifties['Player'].head(10),y=fastest_fifties['BF'].head(10), title=f"Fastest FIfties in IPL - {year}", x_title=' ', y_title='Balls')
            elif filter_batter == 'FASTEST CENTURIES':
                st.subheader(f'{filter_batter}')
                #Fastes Fifties in IPL Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Fastest Centuries/Fastest Centuries - ' + year + '.csv'
                fastest_feature = ['POS','Player','Runs','BF', 'Against']
                fastest_centuries = pd.read_csv(path)
                show_df(fastest_centuries[fastest_feature].head(10))
                generate_plot(x=fastest_centuries['Player'].head(10),y=fastest_centuries['BF'].head(10), title=f"Fastest Centuries in IPL - {year}", x_title=' ', y_title='Sixes')
        else:
            if  filter_bowler == 'HIGHEST WICKETS':
                st.subheader(f'{filter_bowler}')
                #Most Wickets by bowler in a selected season  Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Wickets/Most Wickets - ' + year + '.csv'
                wkt_feature = ['POS','Player','Mat', 'Avg', 'Econ','SR', 'BBI', 'Wkts']
                most_wickets = pd.read_csv(path)
                show_df(most_wickets[wkt_feature].head(10))
                generate_plot(x=most_wickets['Player'].head(10), y= most_wickets['Wkts'].head(10), title=f"Most Wickets in IPL {year}",x_title='Player', y_title='Wickets')

            elif filter_bowler == 'BEST BOWLING ECONOMY':
                st.subheader(filter_bowler)
                # Best Bowling Economy by bowler in a selected season  Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Best Bowling Economy Innings/Best Bowling Economy Innings - ' + year + '.csv'
                economy_bowl = pd.read_csv(path)
                best_economy = economy_bowl.groupby('Player')['Econ'].mean().sort_values(ascending=True)
                show_df(best_economy.head(10))
                most_wins_plot(data=best_economy.head(10), title=f"Best Economic Bowler of IPL {year}", x='Players', y='Economy')

            elif filter_bowler == 'BEST BOWLING ECONOMY (INNINGS)' :
                st.subheader(f'{filter_bowler}')
                #Best Bowling Economy in a inning by bowler in a selected season  Chart and Data
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Best Bowling Economy Innings/Best Bowling Economy Innings - ' + year + '.csv'
                economy_bowling_df = pd.read_csv(path)
                economy_bowling = economy_bowling_df[economy_bowling_df['Ov'] > 3]
                generate_plot(x=economy_bowling['Player'].head(10),y = economy_bowling['Econ'].head(10), title=f"Best Bowling Economy in IPL {year}",x_title='Player', y_title='Economy')
                show_df(economy_bowling[eco_feature].head(10))
            elif filter_bowler == 'MOST DOT BALLS':
                #Most dot balls by bowler in a selected season  Chart and Data
                st.subheader(f'{filter_bowler}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Dot Balls Innings/Most Dot Balls Innings - ' + year + '.csv'
                dot = pd.read_csv(path)
                most_dot = dot.groupby('Player')['Dots'].sum().sort_values(ascending=False)
                show_df(most_dot.head(10))
                most_wins_plot(data=most_dot.head(10), title=f'Most Dot Balls in IPL {year}', x='Players', y='Dot Balls')

            elif filter_bowler == 'MOST DOT BALLS (INNINGS)':
                #Most Dot Balls in a inning by bowler in a selected season Chart and Data
                st.subheader(f'{filter_bowler}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Dot Balls Innings/Most Dot Balls Innings - ' + year + '.csv'
                most_dot = pd.read_csv(path)
                show_df(most_dot.head(10))
                generate_plot(x=most_dot['Player'].head(10),y = most_dot['Dots'].head(10), title=f"Most Dot Ball in a Inning - IPL {year}",x_title='Player', y_title='Dot Balls')
            elif filter_bowler == 'BEST BOWLING STRIKE RATE':
                #best Bowling Strike rate by bowler in a selected season  Chart and Data
                st.subheader(f'{filter_bowler}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Best Bowling Strike Rate Innings/Best Bowling Strike Rate Innings - ' + year + '.csv'
                sr_bowler = pd.read_csv(path)
                best_sr = sr_bowler.groupby('Player')['SR'].mean().sort_values(ascending=True)
                show_df(best_sr.head(10))
                most_wins_plot(data=best_sr.head(10), title=f'Best Strike Rate for Bowlers in IPL {year}', x='Players', y='SR')

            elif filter_bowler == 'BEST BOWLING STRIKE RATE (INNINGS)':
                #Most Bowling Strike Rate per Innings by bowler in a selected season  Chart and Data
                st.subheader(f'{filter_bowler}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Best Bowling Strike Rate Innings/Best Bowling Strike Rate Innings - ' + year + '.csv'
                best_sr_bowler = pd.read_csv(path)
                show_df(best_sr_bowler.head(10))
                generate_plot(x=best_sr_bowler['Player'].head(10),y = best_sr_bowler['SR'].head(10), title=f"Most Dot Ball in a Inning - IPL {year}",x_title='Player', y_title='SR')
            elif filter_bowler == 'MOST RUNS CONCEDED (INNINGS)':
                #Most Runs Conceded by bowler in a selected season Chart and Data
                st.subheader(f'{filter_bowler}')
                path = './IPL - Player Performance Dataset/IPL - Player Performance Dataset/Most Runs Conceded Innings/Most Runs Conceded Innings - ' + year + '.csv'
                run_conceded = pd.read_csv(path)
                show_df(run_conceded.head(10))
                generate_plot(x=run_conceded['Player'].head(10),y = run_conceded['Runs'].head(10), title=f"Most Dot Ball in a Inning - IPL {year}",x_title='Player', y_title='Runs')
        #Display Most wins in IPL Chart and Data of selected season
        st.subheader(f"Most Team Wins In IPL {year}")
        most_wins = data['WinningTeam'].value_counts().sort_values(ascending=False)
        most_wins_df = pd.DataFrame({'Wins': most_wins})
        most_wins_df.index.name = 'Team'
        most_wins_plot(data=most_wins, title=f"Most Match Winners in IPL {year}", x='Teams', y='Wins')
        st.dataframe(most_wins_df, use_container_width=True)
else: 
    #This code will display wehn the slected year values is default -- 'SELECT YEAR"
    if team != 'All Teams':
        data = matches[(matches['Team1'] == team) | (matches['Team2'] == team)]
        st.dataframe(data[stats_feature], use_container_width=True)
    else:
        #This code will display when the selected team values is default -- 'All Teams" and the selected year values is default -- 'SELECT YEAR"
        if filter_options == 'BATTING': 
            
            if filter_batter == 'HIGHEST RUNS':
                #Most Runs in IPL Chart and Data
                st.subheader(filter_batter)
                data = batting_stats_df.sort_values(by='Runs', ascending=False).head(10)
                st.dataframe(data[batting_feature], use_container_width=True)
                generate_plot(x=data.index, y=data['Runs'], title='Highest Run Scorer of IPL', x_title='Players', y_title='Runs')

            elif filter_batter == 'MOST SIXES':
                # Most Sixes in IPL Chart and Data
                st.subheader(filter_batter)
                data = batting_stats_df.sort_values(by='6s', ascending=False).head(10)
                st.dataframe(data[global_six], use_container_width=True)
                generate_plot(x=data.index, y=data['6s'], title='Most Sixes in IPL', x_title='Players', y_title='6s')
            elif filter_batter == 'MOST FOURS':
                #Most Fours in IPL Chart and Data
                st.subheader(f'{filter_batter}')
                data = batting_stats_df.sort_values(by='4s', ascending=False).head(10)
                st.dataframe(data[global_four], use_container_width=True)
                generate_plot(x=data.index, y=data['4s'], title='Most Fours in IPL', x_title='Players', y_title='4s')
            elif filter_batter == 'MOST SIXES (INNINGS)':
                st.subheader(f'Showing HIGHEST SCORE Instead of {filter_batter}')
                #Most Sixes (INNINGS) in IPL Chart and Data
                data = batting_stats_df.sort_values(by='Runs', ascending=False).head(10)
                st.dataframe(data, use_container_width=True)
                generate_plot(x=data.index, y=data['Runs'], title='Highest Run Scorer of IPL', x_title='Players', y_title='Runs')
                
            elif filter_batter == 'MOST FOURS (INNINGS)':
                st.subheader(f'Showing HIGHEST SCORE Instead of {filter_batter}')
                #Most Sixes (INNINGS) in IPL Chart and Data
                data = batting_stats_df.sort_values(by='Runs', ascending=False).head(10)
                st.dataframe(data, use_container_width=True)
                generate_plot(x=data.index, y=data['Runs'], title='Highest Run Scorer of IPL', x_title='Players', y_title='Runs')
                
            elif filter_batter == 'MOST FIFTIES':
                st.subheader(f'{filter_batter}')
                data = batting_stats_df.sort_values(by='50', ascending=False).head(10)
                st.dataframe(data, use_container_width=True)
                generate_plot(x=data.index, y=data['50'], title='Most Fifties in IPL', x_title='Players', y_title='Fifties')
                

                
            elif filter_batter == 'MOST CENTURIES':
                st.subheader(f'{filter_batter}')
                data = batting_stats_df.sort_values(by='100', ascending=False).head(10)
                st.dataframe(data, use_container_width=True)
                generate_plot(x=data.index, y=data['100'], title='Most Centuries in IPL', x_title='Players', y_title='Centuries')
                
            elif filter_batter == 'FASTEST FIFTIES':
                st.subheader(f'{filter_batter}')
                #Fastest Fifties in IPL Chart and Data
               
            elif filter_batter == 'FASTEST CENTURIES':
                st.subheader(f'{filter_batter}')
                #Fastes Fifties in IPL Chart and Data
               
        else:
            if  filter_bowler == 'HIGHEST WICKETS':
                st.subheader(f'{filter_bowler}')
                #Most Wickets Chart and Data
                data = bowling_stats_df.sort_values(by='Wkts', ascending=False).head(10)
                st.dataframe(data[bowling_feature], use_container_width=True)
                generate_plot(x=data.index, y=data['Wkts'], title='Highest Wicket Taker of IPL', x_title='Players', y_title='Wickets')
            elif filter_bowler == 'BEST BOWLING ECONOMY':
                st.subheader(filter_bowler)
                # Best Bowling Economy Chart and Data
           

            elif filter_bowler == 'BEST BOWLING ECONOMY (INNINGS)' :
                st.subheader(f'{filter_bowler}')
                #Best Bowling Economy Chart and Data
            
            elif filter_bowler == 'MOST DOT BALLS':
                #Most Wickets Chart and Data
                st.subheader(f'{filter_bowler}')
                

            elif filter_bowler == 'MOST DOT BALLS (INNINGS)':
                #Most Wickets Chart and Data
                st.subheader(f'{filter_bowler}')
                
            elif filter_bowler == 'BEST BOWLING STRIKE RATE':
                #Most Wickets Chart and Data
                st.subheader(f'{filter_bowler}')
                

            elif filter_bowler == 'BEST BOWLING STRIKE RATE (INNINGS)':
                #Most Wickets Chart and Data
                st.subheader(f'{filter_bowler}')
      
            elif filter_bowler == 'MOST RUNS CONCEDED (INNINGS)':
                #Most Runs Conceded Chart and Data
                st.subheader(f'{filter_bowler}')
                
        # Display the top 10 teams with the most wins
        st.subheader("Most Win In IPL for a Team")
        st.table(most_wins_df.head(10))
        most_wins_plot(data=most_wins, title="Most Match Winners in IPL", x='Teams', y='Wins')

        
