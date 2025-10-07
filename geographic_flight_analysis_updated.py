# Geographic Flight Analysis and Visualization for United Airlines Flight Difficulty Score
# Author: Flight Operations Analytics Team
# Date: October 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cell 1: Airport Coordinates Database (Major US Airports)
# Since we can't access external APIs, we'll create a comprehensive airport coordinates dictionary
AIRPORT_COORDINATES = {
    'ORD': {'lat': 41.978611, 'lon': -87.904724, 'name': "Chicago O'Hare International"},
    'LAX': {'lat': 33.942791, 'lon': -118.410042, 'name': 'Los Angeles International'},
    'JFK': {'lat': 40.641766, 'lon': -73.780968, 'name': 'John F. Kennedy International'},
    'LGA': {'lat': 40.776863, 'lon': -73.874069, 'name': 'LaGuardia Airport'},
    'ATL': {'lat': 33.640444, 'lon': -84.426944, 'name': 'Hartsfield-Jackson Atlanta International'},
    'DEN': {'lat': 39.858667, 'lon': -104.667, 'name': 'Denver International'},
    'DFW': {'lat': 32.899722, 'lon': -97.040556, 'name': 'Dallas/Fort Worth International'},
    'MIA': {'lat': 25.795160, 'lon': -80.279594, 'name': 'Miami International'},
    'SEA': {'lat': 47.448889, 'lon': -122.309444, 'name': 'Seattle-Tacoma International'},
    'SFO': {'lat': 37.618889, 'lon': -122.375, 'name': 'San Francisco International'},
    'LAS': {'lat': 36.083134, 'lon': -115.148315, 'name': 'Harry Reid International'},
    'PHX': {'lat': 33.435249, 'lon': -112.010216, 'name': 'Phoenix Sky Harbor International'},
    'IAH': {'lat': 29.993067, 'lon': -95.341812, 'name': 'George Bush Intercontinental'},
    'MCO': {'lat': 28.424618, 'lon': -81.310753, 'name': 'Orlando International'},
    'EWR': {'lat': 40.689491, 'lon': -74.174538, 'name': 'Newark Liberty International'},
    'MSP': {'lat': 44.883333, 'lon': -93.216667, 'name': 'Minneapolis-Saint Paul International'},
    'DTW': {'lat': 42.213249, 'lon': -83.352859, 'name': 'Detroit Metropolitan Wayne County'},
    'BOS': {'lat': 42.365556, 'lon': -71.009444, 'name': 'Boston Logan International'},
    'PHL': {'lat': 39.872940, 'lon': -75.243988, 'name': 'Philadelphia International'},
    'CLT': {'lat': 35.213890, 'lon': -80.943054, 'name': 'Charlotte Douglas International'},
    'BWI': {'lat': 39.177540, 'lon': -76.668526, 'name': 'Baltimore/Washington International'},
    'DCA': {'lat': 38.851944, 'lon': -77.040556, 'name': 'Ronald Reagan Washington National'},
    'IAD': {'lat': 38.953333, 'lon': -77.456667, 'name': 'Washington Dulles International'},
    'FLL': {'lat': 26.074215, 'lon': -80.150726, 'name': 'Fort Lauderdale-Hollywood International'},
    'TPA': {'lat': 27.979168, 'lon': -82.539337, 'name': 'Tampa International'},
    'SAN': {'lat': 32.731944, 'lon': -117.196667, 'name': 'San Diego International'},
    'STL': {'lat': 38.747222, 'lon': -90.361389, 'name': 'St. Louis Lambert International'},
    'HNL': {'lat': 21.325556, 'lon': -157.925, 'name': 'Daniel K. Inouye International'},
    'PDX': {'lat': 45.589444, 'lon': -122.596389, 'name': 'Portland International'},
    'MDW': {'lat': 41.785833, 'lon': -87.7525, 'name': 'Chicago Midway International'},
    'BNA': {'lat': 36.126667, 'lon': -86.681944, 'name': 'Nashville International'},
    'AUS': {'lat': 30.197222, 'lon': -97.662222, 'name': 'Austin-Bergstrom International'},
    'MCI': {'lat': 39.299722, 'lon': -94.713889, 'name': 'Kansas City International'},
    'RDU': {'lat': 35.877222, 'lon': -78.787222, 'name': 'Raleigh-Durham International'},
    'IND': {'lat': 39.717222, 'lon': -86.293889, 'name': 'Indianapolis International'},
    'CMH': {'lat': 39.999444, 'lon': -82.891944, 'name': 'John Glenn Columbus International'},
    'PIT': {'lat': 40.492222, 'lon': -80.240833, 'name': 'Pittsburgh International'},
    'MSY': {'lat': 29.988889, 'lon': -90.258056, 'name': 'Louis Armstrong New Orleans International'},
    'CLE': {'lat': 41.409722, 'lon': -81.847222, 'name': 'Cleveland Hopkins International'},
    'MEM': {'lat': 35.041667, 'lon': -89.976111, 'name': 'Memphis International'},
    'ANC': {'lat': 61.1744, 'lon': -149.9960, 'name': 'Ted Stevens Anchorage International'},
    'SLC': {'lat': 40.785833, 'lon': -111.978056, 'name': 'Salt Lake City International'},
    'SMF': {'lat': 38.694444, 'lon': -121.590833, 'name': 'Sacramento International'},
    'SJC': {'lat': 37.365556, 'lon': -121.929167, 'name': 'San Jose International'},
    'OAK': {'lat': 37.712222, 'lon': -122.221111, 'name': 'Oakland International'},
    'BUR': {'lat': 34.201667, 'lon': -118.358611, 'name': 'Hollywood Burbank Airport'},
    'ONT': {'lat': 34.056, 'lon': -117.600833, 'name': 'Ontario International'},
    'SNA': {'lat': 33.676111, 'lon': -117.867222, 'name': 'John Wayne Airport'},
    'PSP': {'lat': 33.826944, 'lon': -116.506389, 'name': 'Palm Springs International'},
    'FAT': {'lat': 36.773056, 'lon': -119.717222, 'name': 'Fresno Yosemite International'},
    'RAP': {'lat': 44.045278, 'lon': -103.057222, 'name': 'Rapid City Regional'},
    'BIL': {'lat': 45.808056, 'lon': -108.537778, 'name': 'Billings Logan International'},
    'COD': {'lat': 44.520556, 'lon': -109.023889, 'name': 'Yellowstone Regional'},
    'JAC': {'lat': 43.606389, 'lon': -110.737778, 'name': 'Jackson Hole Airport'},
    'SUN': {'lat': 43.504444, 'lon': -114.296389, 'name': 'Friedman Memorial'},
    'BOI': {'lat': 43.564444, 'lon': -116.222778, 'name': 'Boise Airport'},
    'GEG': {'lat': 47.619861, 'lon': -117.533611, 'name': 'Spokane International'},
    'ABQ': {'lat': 35.040833, 'lon': -106.609167, 'name': 'Albuquerque International Sunport'},
    'ELP': {'lat': 31.807222, 'lon': -106.378056, 'name': 'El Paso International'},
    'SAT': {'lat': 29.533694, 'lon': -98.469806, 'name': 'San Antonio International'},
    'HOU': {'lat': 29.645278, 'lon': -95.278889, 'name': 'William P. Hobby Airport'},
    'DAL': {'lat': 32.848152, 'lon': -96.851349, 'name': 'Dallas Love Field'},
    'OKC': {'lat': 35.393056, 'lon': -97.600833, 'name': 'Will Rogers World Airport'},
    'TUL': {'lat': 36.198056, 'lon': -95.888111, 'name': 'Tulsa International'},
    'LIT': {'lat': 34.726944, 'lon': -92.224167, 'name': 'Bill and Hillary Clinton National'},
    'XNA': {'lat': 36.281944, 'lon': -94.306667, 'name': 'Northwest Arkansas Regional'},
    'SGF': {'lat': 37.245833, 'lon': -93.388611, 'name': 'Springfield-Branson National'},
    'DSM': {'lat': 41.533056, 'lon': -93.663056, 'name': 'Des Moines International'},
    'OMA': {'lat': 41.303056, 'lon': -95.894167, 'name': 'Eppley Airfield'},
    'ICT': {'lat': 37.649722, 'lon': -97.433056, 'name': 'Wichita Dwight D. Eisenhower National'},
    'COS': {'lat': 38.805833, 'lon': -104.7, 'name': 'Colorado Springs Airport'},
    'GJT': {'lat': 39.123611, 'lon': -108.526667, 'name': 'Grand Junction Regional'},
    'ASE': {'lat': 39.223056, 'lon': -106.868611, 'name': 'Aspen/Pitkin County'},
    'EGE': {'lat': 39.642778, 'lon': -106.917778, 'name': 'Eagle County Regional'},
    'HDN': {'lat': 40.481944, 'lon': -107.217222, 'name': 'Yampa Valley'},
    'MTJ': {'lat': 38.509444, 'lon': -107.893889, 'name': 'Montrose Regional'},
    'PUB': {'lat': 38.289167, 'lon': -104.496389, 'name': 'Pueblo Memorial'},
    'ROA': {'lat': 37.325556, 'lon': -79.975417, 'name': 'Roanoke-Blacksburg Regional'},
    'CRW': {'lat': 38.373056, 'lon': -81.593056, 'name': 'Charleston Yeager'},
    'MGW': {'lat': 39.641944, 'lon': -79.916389, 'name': 'Morgantown Municipal'},
    'BDL': {'lat': 41.938889, 'lon': -72.683056, 'name': 'Bradley International'},
    'PWM': {'lat': 43.646111, 'lon': -70.309167, 'name': 'Portland International Jetport'},
    'BGR': {'lat': 44.807222, 'lon': -68.828056, 'name': 'Bangor International'},
    'ALB': {'lat': 42.748056, 'lon': -73.801944, 'name': 'Albany International'},
    'SYR': {'lat': 43.111944, 'lon': -76.106389, 'name': 'Syracuse Hancock International'},
    'ROC': {'lat': 43.118889, 'lon': -77.672222, 'name': 'Greater Rochester International'},
    'BUF': {'lat': 42.940556, 'lon': -78.732222, 'name': 'Buffalo Niagara International'},
    'ERI': {'lat': 42.083056, 'lon': -80.173889, 'name': 'Erie International'},
    'CAK': {'lat': 40.915556, 'lon': -81.441944, 'name': 'Akron-Canton Airport'},
    'TOL': {'lat': 41.586667, 'lon': -83.807778, 'name': 'Eugene F. Kranz Toledo Express'},
    'DAY': {'lat': 39.902222, 'lon': -84.219444, 'name': 'James M. Cox Dayton International'},
    'CVG': {'lat': 39.048056, 'lon': -84.667778, 'name': 'Cincinnati/Northern Kentucky International'},
    'SDF': {'lat': 38.174444, 'lon': -85.736, 'name': 'Louisville International'},
    'LEX': {'lat': 38.036389, 'lon': -84.605833, 'name': 'Blue Grass Airport'},
    'TYS': {'lat': 35.811944, 'lon': -83.994167, 'name': 'McGhee Tyson Airport'},
    'TRI': {'lat': 36.475278, 'lon': -82.407778, 'name': 'Tri-Cities Airport'},
    'CHA': {'lat': 35.035278, 'lon': -85.203889, 'name': 'Chattanooga Metropolitan'},
    'BHM': {'lat': 33.562944, 'lon': -86.753531, 'name': 'Birmingham-Shuttlesworth International'},
    'HSV': {'lat': 34.637222, 'lon': -86.775, 'name': 'Huntsville International'},
    'MOB': {'lat': 30.691944, 'lon': -88.243056, 'name': 'Mobile Regional'},
    'PNS': {'lat': 30.473056, 'lon': -87.186944, 'name': 'Pensacola International'},
    'TLH': {'lat': 30.396389, 'lon': -84.350278, 'name': 'Tallahassee International'},
    'GNV': {'lat': 29.690556, 'lon': -82.271944, 'name': 'Gainesville Regional'},
    'DAB': {'lat': 29.179889, 'lon': -81.058056, 'name': 'Daytona Beach International'},
    'MLB': {'lat': 28.102778, 'lon': -80.645278, 'name': 'Melbourne Orlando International'},
    'PBI': {'lat': 26.683056, 'lon': -80.095556, 'name': 'Palm Beach International'},
    'RSW': {'lat': 26.536111, 'lon': -81.755556, 'name': 'Southwest Florida International'},
    'PIE': {'lat': 27.910167, 'lon': -82.687389, 'name': 'St. Pete-Clearwater International'},
    'SRQ': {'lat': 27.395833, 'lon': -82.554167, 'name': 'Sarasota-Bradenton International'},
    'JAX': {'lat': 30.494167, 'lon': -81.687778, 'name': 'Jacksonville International'},
    'SAV': {'lat': 32.127778, 'lon': -81.202222, 'name': 'Savannah/Hilton Head International'},
    'AGS': {'lat': 33.369944, 'lon': -81.964444, 'name': 'Augusta Regional'},
    'CAE': {'lat': 33.938889, 'lon': -81.119444, 'name': 'Columbia Metropolitan'},
    'CHS': {'lat': 32.898611, 'lon': -80.040556, 'name': 'Charleston International'},
    'MYR': {'lat': 33.679167, 'lon': -78.928333, 'name': 'Myrtle Beach International'},
    'ILM': {'lat': 34.270556, 'lon': -77.902222, 'name': 'Wilmington International'},
    'FAY': {'lat': 34.991389, 'lon': -78.880278, 'name': 'Fayetteville Regional'},
    'GSO': {'lat': 36.097778, 'lon': -79.937222, 'name': 'Piedmont Triad International'},
    'RIC': {'lat': 37.505167, 'lon': -77.319667, 'name': 'Richmond International'},
    'ORF': {'lat': 36.894444, 'lon': -76.201389, 'name': 'Norfolk International'},
    'LHR': {'lat': 51.4775, 'lon': -0.461389, 'name': 'London Heathrow'},
    'CDG': {'lat': 49.009722, 'lon': 2.547778, 'name': 'Charles de Gaulle'},
    'FRA': {'lat': 50.0333, 'lon': 8.5706, 'name': 'Frankfurt am Main'},
    'NRT': {'lat': 35.764722, 'lon': 140.386111, 'name': 'Narita International'},
    'ICN': {'lat': 37.4691, 'lon': 126.4510, 'name': 'Incheon International'},
    'YVR': {'lat': 49.194722, 'lon': -123.184444, 'name': 'Vancouver International'},
    'YYZ': {'lat': 43.676667, 'lon': -79.630556, 'name': 'Toronto Pearson International'},
    'YUL': {'lat': 45.457222, 'lon': -73.749167, 'name': 'Montreal-Pierre Elliott Trudeau'},
    'GRU': {'lat': -23.435556, 'lon': -46.473056, 'name': 'São Paulo-Guarulhos International'},
    'MEX': {'lat': 19.436389, 'lon': -99.072222, 'name': 'Mexico City International'},
    'CUN': {'lat': 21.036389, 'lon': -86.876944, 'name': 'Cancún International'},
    # Additional smaller airports
    'MCW': {'lat': 41.441667, 'lon': -90.504167, 'name': 'Mason City Municipal'},
    'LAF': {'lat': 40.412222, 'lon': -86.936944, 'name': 'Lafayette Regional'},
    'DEC': {'lat': 39.834167, 'lon': -88.865833, 'name': 'Decatur Airport'},
    'MSN': {'lat': 43.139833, 'lon': -89.337500, 'name': 'Dane County Regional'},
    'GRB': {'lat': 44.485000, 'lon': -88.129167, 'name': 'Green Bay-Austin Straubel International'},
    'LSE': {'lat': 43.879167, 'lon': -91.256389, 'name': 'La Crosse Regional'},
    'EAU': {'lat': 44.865556, 'lon': -91.484444, 'name': 'Chippewa Valley Regional'},
    'CWA': {'lat': 44.777778, 'lon': -89.666667, 'name': 'Central Wisconsin'},
    'ATW': {'lat': 44.258333, 'lon': -88.519167, 'name': 'Appleton International'}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r

# Cell 2: Load CSVs (as provided)
flight_df   = pd.read_csv('./data/Flight_Level_Data.csv', 
                          parse_dates=['scheduled_departure_datetime_local',
                                       'actual_departure_datetime_local'])
pnr_df      = pd.read_csv('./data/PNR_Flight_Level_Data.csv')
remarks_df  = pd.read_csv('./data/PNR_Remark_Level_Data.csv')
bag_df      = pd.read_csv('./data/Bag_Level_Data.csv')
airport_df  = pd.read_csv('./data/Airports_Data.csv')

# Display row counts
print("Dataset Row Counts:")
print(f"Flight Level Data: {len(flight_df):,} rows")
print(f"PNR Flight Level Data: {len(pnr_df):,} rows")
print(f"PNR Remarks Data: {len(remarks_df):,} rows")
print(f"Bag Level Data: {len(bag_df):,} rows")
print(f"Airport Data: {len(airport_df):,} rows")

# Cell 3: Data Preprocessing and Feature Engineering
def preprocess_flight_data(flight_df):
    """
    Preprocess flight data and calculate difficulty score components
    """
    # Calculate delays
    flight_df['delay_minutes'] = (
        pd.to_datetime(flight_df['actual_departure_datetime_local']) - 
        pd.to_datetime(flight_df['scheduled_departure_datetime_local'])
    ).dt.total_seconds() / 60
    
    # Add coordinates for destinations
    flight_df['dest_lat'] = flight_df['scheduled_arrival_station_code'].map(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get('lat', None))
    flight_df['dest_lon'] = flight_df['scheduled_arrival_station_code'].map(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get('lon', None))
    flight_df['dest_name'] = flight_df['scheduled_arrival_station_code'].map(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get('name', x))
    
    # ORD coordinates (origin)
    ord_lat, ord_lon = 41.978611, -87.904724
    
    # Calculate flight distances
    flight_df['flight_distance_miles'] = flight_df.apply(
        lambda row: haversine_distance(ord_lat, ord_lon, row['dest_lat'], row['dest_lon']) 
        if pd.notna(row['dest_lat']) and pd.notna(row['dest_lon']) else None, axis=1)
    
    # Ground time pressure (negative means tight turnaround)
    if 'ground_time_pressure' not in flight_df.columns:
        flight_df['ground_time_pressure'] = (
            flight_df.get('scheduled_ground_time_minutes', 0) - 
            flight_df.get('minimum_turn_minutes', 0)
        )
    
    # Load factor (if not present)
    if 'load_factor' not in flight_df.columns:
        flight_df['load_factor'] = (
            flight_df.get('total_passengers', 0) / 
            flight_df.get('total_seats', 1)
        )
    
    # SSR count (if not present)
    if 'ssr_count' not in flight_df.columns:
        flight_df['ssr_count'] = 0  # Will be filled from remarks data
    
    # Calculate difficulty score (simplified version)
    # Normalize components to 0-1 scale
    flight_df['ground_time_score'] = np.clip(-flight_df['ground_time_pressure'] / 60, 0, 1)
    flight_df['load_factor_score'] = np.clip(flight_df['load_factor'], 0, 1)
    flight_df['delay_score'] = np.clip(flight_df['delay_minutes'] / 120, 0, 1)
    
    # Composite difficulty score
    flight_df['difficulty_score'] = (
        0.4 * flight_df['ground_time_score'] +
        0.3 * flight_df['load_factor_score'] +
        0.3 * flight_df['delay_score']
    )
    
    # Classify difficulty levels
    flight_df['difficulty_level'] = pd.cut(
        flight_df['difficulty_score'], 
        bins=[0, 0.33, 0.66, 1.0], 
        labels=['Easy', 'Medium', 'Difficult'],
        include_lowest=True
    )
    
    return flight_df

# Preprocess the data
flight_df = preprocess_flight_data(flight_df)

# Cell 4: Create Geographic Visualizations
def create_route_network_map(flight_df):
    """
    Create an interactive route network complexity map
    """
    # Aggregate data by destination
    route_stats = flight_df.groupby('scheduled_arrival_station_code').agg({
        'difficulty_score': ['mean', 'count'],
        'delay_minutes': 'mean',
        'load_factor': 'mean',
        'flight_distance_miles': 'mean'
    }).round(3)
    
    # Flatten column names
    route_stats.columns = ['_'.join(col).strip() for col in route_stats.columns]
    route_stats = route_stats.reset_index()
    
    # Add coordinates
    route_stats['lat'] = route_stats['scheduled_arrival_station_code'].map(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get('lat', None))
    route_stats['lon'] = route_stats['scheduled_arrival_station_code'].map(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get('lon', None))
    route_stats['airport_name'] = route_stats['scheduled_arrival_station_code'].map(
        lambda x: AIRPORT_COORDINATES.get(x, {}).get('name', x))
    
    # Filter out missing coordinates
    route_stats = route_stats.dropna(subset=['lat', 'lon'])
    
    # Create the map
    fig = go.Figure()
    
    # Add ORD as the hub
    fig.add_trace(go.Scattergeo(
        lon=[-87.904724],
        lat=[41.978611],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        text=['ORD Hub'],
        textposition='top center',
        name='Chicago ORD Hub',
        textfont=dict(size=14, color='red')
    ))
    
    # Add destination airports
    fig.add_trace(go.Scattergeo(
        lon=route_stats['lon'],
        lat=route_stats['lat'],
        mode='markers+text',
        marker=dict(
            size=route_stats['difficulty_score_count'] * 2,  # Size by flight volume
            color=route_stats['difficulty_score_mean'],        # Color by difficulty
            colorscale='RdYlBu_r',
            colorbar=dict(
                title="Average Difficulty Score",
                x=0.85
            ),
            sizemode='diameter',
            sizeref=max(route_stats['difficulty_score_count']) / 50,
            sizemin=8,
            line=dict(width=1, color='black')
        ),
        text=route_stats['scheduled_arrival_station_code'],
        textposition='top center',
        name='Destination Airports',
        customdata=np.column_stack((
            route_stats['airport_name'],
            route_stats['difficulty_score_mean'],
            route_stats['difficulty_score_count'],
            route_stats['delay_minutes_mean'],
            route_stats['load_factor_mean']
        )),
        hovertemplate='<b>%{text}</b><br>' +
                      'Airport: %{customdata[0]}<br>' +
                      'Avg Difficulty Score: %{customdata[1]:.3f}<br>' +
                      'Number of Flights: %{customdata[2]}<br>' +
                      'Avg Delay: %{customdata[3]:.1f} min<br>' +
                      'Avg Load Factor: %{customdata[4]:.1f}%<extra></extra>'
    ))
    
    # Add route lines from ORD to destinations (top 20 by volume)
    top_routes = route_stats.nlargest(20, 'difficulty_score_count')
    
    for _, route in top_routes.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[-87.904724, route['lon']],
            lat=[41.978611, route['lat']],
            mode='lines',
            line=dict(
                width=max(1, route['difficulty_score_mean'] * 5),
                color=f"rgba(255, 0, 0, {route['difficulty_score_mean']})"
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'United Airlines Route Network Complexity Map<br><sub>From Chicago ORD - Bubble Size: Flight Volume, Color: Avg Difficulty Score</sub>',
            'x': 0.5,
            'font': {'size': 18}
        },
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            center=dict(lat=39, lon=-98),
            scope='world'
        ),
        height=600,
        width=1000,
        showlegend=True,
        font=dict(size=12)
    )
    
    return fig

def create_hub_operations_radial_chart(flight_df):
    """
    Create a radial chart showing hub operations with ORD at center
    """
    # Aggregate by destination
    hub_stats = flight_df.groupby('scheduled_arrival_station_code').agg({
        'difficulty_score': 'mean',
        'flight_number': 'count',
        'delay_minutes': 'mean'
    }).round(3)
    
    # Get top 20 destinations by flight count
    top_destinations = hub_stats.nlargest(20, 'flight_number')
    
    # Create angles for radial chart
    angles = np.linspace(0, 2*np.pi, len(top_destinations), endpoint=False)
    
    # Create the radial plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Plot each route as a line from center
    for i, (dest, stats) in enumerate(top_destinations.iterrows()):
        angle = angles[i]
        difficulty = stats['difficulty_score']
        flight_count = stats['flight_number']
        
        # Line thickness based on difficulty
        linewidth = max(1, difficulty * 10)
        
        # Color based on difficulty
        color = plt.cm.RdYlBu_r(difficulty)
        
        # Draw line from center to edge
        ax.plot([0, angle], [0, flight_count], 
                linewidth=linewidth, color=color, alpha=0.7)
        
        # Add airport code at the end
        ax.text(angle, flight_count + max(top_destinations['flight_number']) * 0.05, 
                dest, rotation=np.degrees(angle) - 90 if angle > np.pi/2 and angle < 3*np.pi/2 
                else np.degrees(angle) + 90 if angle > np.pi/2 and angle < 3*np.pi/2 
                else np.degrees(angle),
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Center hub label
    ax.text(0, 0, 'ORD\nHUB', ha='center', va='center', 
            fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    # Customize the plot
    ax.set_ylim(0, max(top_destinations['flight_number']) * 1.2)
    ax.set_title('Hub Operations Load - Radial View\n' + 
                 'Line Thickness: Difficulty Score, Distance: Flight Volume',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Remove radial labels for cleaner look
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                              norm=plt.Normalize(vmin=top_destinations['difficulty_score'].min(),
                                                 vmax=top_destinations['difficulty_score'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Average Difficulty Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

def create_distance_complexity_analysis(flight_df):
    """
    Create scatter plot showing relationship between flight distance and complexity
    """
    # Filter out missing distance data
    distance_data = flight_df.dropna(subset=['flight_distance_miles', 'difficulty_score'])
    
    # Create the scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Distance vs Difficulty Score
    scatter = ax1.scatter(distance_data['flight_distance_miles'], 
                          distance_data['difficulty_score'],
                          c=distance_data['delay_minutes'], 
                          s=distance_data['load_factor'] * 100,
                          alpha=0.6, cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Flight Distance (Miles)')
    ax1.set_ylabel('Difficulty Score')
    ax1.set_title('Flight Distance vs Operational Complexity\nBubble Size: Load Factor, Color: Delay Minutes')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(distance_data['flight_distance_miles'], distance_data['difficulty_score'], 1)
    p = np.poly1d(z)
    ax1.plot(distance_data['flight_distance_miles'], 
             p(distance_data['flight_distance_miles']), 
             "r--", alpha=0.8, linewidth=2, label=f'Trend Line (R²={np.corrcoef(distance_data["flight_distance_miles"], distance_data["difficulty_score"])[0,1]**2:.3f})')
    ax1.legend()
    
    # Colorbar for delays
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Average Delay (Minutes)', rotation=270, labelpad=20)
    
    # Plot 2: Distance categories vs Difficulty
    # Create distance bins
    distance_data['distance_category'] = pd.cut(distance_data['flight_distance_miles'], 
                                               bins=[0, 500, 1000, 1500, 5000], 
                                               labels=['Short (<500mi)', 'Medium (500-1000mi)', 
                                                      'Long (1000-1500mi)', 'Very Long (1500mi+)'],
                                               include_lowest=True)
    
    # Box plot by distance category
    distance_categories = ['Short (<500mi)', 'Medium (500-1000mi)', 'Long (1000-1500mi)', 'Very Long (1500mi+)']
    difficulty_by_distance = [distance_data[distance_data['distance_category'] == cat]['difficulty_score'].values 
                              for cat in distance_categories if cat in distance_data['distance_category'].values]
    
    box_plot = ax2.boxplot(difficulty_by_distance, labels=distance_categories[:len(difficulty_by_distance)], 
                           patch_artist=True, notch=True)
    
    # Color the boxes
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(difficulty_by_distance)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Difficulty Score')
    ax2.set_title('Difficulty Distribution by Flight Distance Category')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_comprehensive_dashboard(flight_df):
    """
    Create a comprehensive dashboard with multiple visualizations
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Flight Volume by Destination', 'Avg Difficulty by Destination',
                       'Distance vs Difficulty Correlation', 'Operational Metrics by Route'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Data aggregation
    route_summary = flight_df.groupby('scheduled_arrival_station_code').agg({
        'difficulty_score': ['mean', 'count'],
        'delay_minutes': 'mean',
        'load_factor': 'mean',
        'flight_distance_miles': 'mean'
    }).round(3)
    
    route_summary.columns = ['_'.join(col).strip() for col in route_summary.columns]
    route_summary = route_summary.reset_index()
    
    # Get top destinations
    top_20_volume = route_summary.nlargest(20, 'difficulty_score_count')
    top_20_difficulty = route_summary.nlargest(20, 'difficulty_score_mean')
    
    # Plot 1: Flight Volume
    fig.add_trace(
        go.Bar(x=top_20_volume['scheduled_arrival_station_code'], 
               y=top_20_volume['difficulty_score_count'],
               marker_color='steelblue',
               name='Flight Count'),
        row=1, col=1
    )
    
    # Plot 2: Average Difficulty
    fig.add_trace(
        go.Bar(x=top_20_difficulty['scheduled_arrival_station_code'],
               y=top_20_difficulty['difficulty_score_mean'],
               marker_color='red',
               name='Avg Difficulty'),
        row=1, col=2
    )
    
    # Plot 3: Distance vs Difficulty
    clean_data = route_summary.dropna()
    fig.add_trace(
        go.Scatter(x=clean_data['flight_distance_miles_mean'],
                  y=clean_data['difficulty_score_mean'],
                  mode='markers',
                  marker=dict(size=clean_data['difficulty_score_count'],
                              color=clean_data['delay_minutes_mean'],
                              colorscale='RdYlBu_r',
                              sizemode='diameter',
                              sizeref=max(clean_data['difficulty_score_count'])/50,
                              sizemin=8,
                              showscale=True,
                              colorbar=dict(title="Avg Delay (min)", x=0.45, len=0.4)),
                  text=clean_data['scheduled_arrival_station_code'],
                  name='Routes'),
        row=2, col=1
    )
    
    # Plot 4: Load Factor vs Difficulty
    fig.add_trace(
        go.Scatter(x=clean_data['load_factor_mean'],
                  y=clean_data['difficulty_score_mean'],
                  mode='markers+text',
                  marker=dict(size=10, color='orange'),
                  text=clean_data['scheduled_arrival_station_code'],
                  textposition='top center',
                  name='Load Factor vs Difficulty'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="United Airlines Flight Operations Dashboard - Chicago ORD Hub",
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Destination Airport", row=1, col=1)
    fig.update_yaxes(title_text="Number of Flights", row=1, col=1)
    fig.update_xaxes(title_text="Destination Airport", row=1, col=2)
    fig.update_yaxes(title_text="Avg Difficulty Score", row=1, col=2)
    fig.update_xaxes(title_text="Flight Distance (Miles)", row=2, col=1)
    fig.update_yaxes(title_text="Avg Difficulty Score", row=2, col=1)
    fig.update_xaxes(title_text="Avg Load Factor", row=2, col=2)
    fig.update_yaxes(title_text="Avg Difficulty Score", row=2, col=2)
    
    return fig

def save_plotly_png(fig, path, width=1200, height=700, scale=2):
    """
    Save Plotly figure as a static PNG using kaleido.
    """
    try:
        fig.write_image(path, width=width, height=height, scale=scale)
    except ValueError as e:
        print(f"[Warning] Failed to save {path} as PNG. Ensure 'kaleido' is installed. Error: {e}")

# Cell 5: Generate All Visualizations
def main():
    """
    Main function to generate all geographic visualizations
    """
    print("Generating Geographic Flight Analysis Visualizations...\n")
    
    # 1. Route Network Complexity Map
    print("1. Creating Route Network Complexity Map...")
    map_fig = create_route_network_map(flight_df)
    # was: write_html + show. Now save as PNG (and still show for continuity).
    save_plotly_png(map_fig, "route_network_complexity_map.png", width=1400, height=900, scale=2)
    map_fig.show()
    print("   ✓ Saved as 'route_network_complexity_map.png'")
    
    # 2. Hub Operations Radial Chart
    print("\n2. Creating Hub Operations Radial Chart...")
    radial_fig = create_hub_operations_radial_chart(flight_df)
    radial_fig.savefig('hub_operations_radial_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Saved as 'hub_operations_radial_chart.png'")
    
    # 3. Distance vs Complexity Analysis
    print("\n3. Creating Distance vs Complexity Analysis...")
    distance_fig = create_distance_complexity_analysis(flight_df)
    distance_fig.savefig('distance_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Saved as 'distance_complexity_analysis.png'")
    
    # 4. Comprehensive Dashboard
    print("\n4. Creating Comprehensive Operations Dashboard...")
    dashboard_fig = create_comprehensive_dashboard(flight_df)
    # was: write_html + show. Now save as PNG (and still show for continuity).
    save_plotly_png(dashboard_fig, "operations_dashboard.png", width=1400, height=1000, scale=2)
    dashboard_fig.show()
    print("   ✓ Saved as 'operations_dashboard.png'")
    
    # 5. Summary Statistics
    print("\n" + "="*50)
    print("FLIGHT DIFFICULTY ANALYSIS SUMMARY")
    print("="*50)
    
    # Route complexity summary
    route_stats = flight_df.groupby('scheduled_arrival_station_code').agg({
        'difficulty_score': ['mean', 'count'],
        'delay_minutes': 'mean',
        'flight_distance_miles': 'mean'
    }).round(3)
    
    print(f"Total Routes Analyzed: {len(route_stats)}")
    print(f"Total Flights: {len(flight_df):,}")
    print(f"Average Flight Distance: {flight_df['flight_distance_miles'].mean():.0f} miles")
    print(f"Average Difficulty Score: {flight_df['difficulty_score'].mean():.3f}")
    
    # Top 10 most difficult routes (kept prints for continuity)
    route_stats.columns = ['_'.join(col).strip() for col in route_stats.columns]
    top_difficult = route_stats.nlargest(10, 'difficulty_score_mean')
    
    print(f"\nTop 10 Most Operationally Complex Routes:")
    for i, (dest, stats) in enumerate(top_difficult.iterrows(), 1):
        airport_name = AIRPORT_COORDINATES.get(dest, {}).get('name', dest)
        flight_count = int(stats['difficulty_score_count'])
        print(f"{i:2d}. {dest} - {airport_name[:30]:<30} "
              f"(Score: {stats['difficulty_score_mean']:.3f}, "
              f"Flights: {flight_count:3d})")

    # NEW: Graph for Top 15 most difficult routes instead of just printing them
    print("\nCreating bar chart for Top 15 Most Operationally Complex Routes...")
    top15 = route_stats.nlargest(15, 'difficulty_score_mean').copy()
    # Order for plotting
    top15 = top15.sort_values('difficulty_score_mean', ascending=True)
    
    # Matplotlib horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top15.index, top15['difficulty_score_mean'], color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(top15))))
    ax.set_xlabel('Average Difficulty Score')
    ax.set_title('Top 15 Most Operationally Complex Routes (Avg Difficulty Score)')
    
    # Annotate with flight counts
    for rect, cnt in zip(bars, top15['difficulty_score_count'].astype(int)):
        width = rect.get_width()
        ax.text(width + 0.01, rect.get_y() + rect.get_height()/2, f"Flights: {cnt}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('top15_difficult_routes.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Saved as 'top15_difficult_routes.png'")
    
    print(f"\nVisualization files created:")
    print("- route_network_complexity_map.png (Static map image)")
    print("- hub_operations_radial_chart.png (Radial chart)")
    print("- distance_complexity_analysis.png (Scatter analysis)")
    print("- operations_dashboard.png (Comprehensive dashboard)")
    print("- top15_difficult_routes.png (Bar chart)")

if __name__ == "__main__":
    main()
