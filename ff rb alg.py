import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV file
data = pd.read_csv('stats.csv')

# Replace 'n/a' with NaN
data.replace('n/a', pd.NA, inplace=True)

# Identify numeric columns (excluding 'Name' and 'Team')
numeric_columns = data.columns.drop(['Name', 'Team'])

# Convert numeric columns to numeric, coercing errors to NaN
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the mean of their respective columns (only for numeric columns)
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Normalize the numeric columns
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Invert the normalized values for stats where lower is better
stats_to_invert = ['Depth_Chart', 'RB_Expected_Tier', 'Offensive_Line_Rank']
for stat in stats_to_invert:
    data[stat] = 1 - data[stat]

# Your rankings for the stats
rankings = {
    'Depth_Chart': 3.5,
    'RB_Expected_Tier': 1,
    'Rec_Yds': 4.5,
    'Games_Played': 6,
    'Targets': 5,
    'Receptions': 4,
    'Yds_Tgt': 4.5,
    'Rec_TDs': 3,
    'Yds_per_Carry': 4,
    'Rush_Yds': 4.5,
    'Carries': 4,
    'Rushing_TDs': 3,
    'Snap_Percent': 4,
    'Redzone_Yds': 6,
    'Redzone_Targets': 4,
    'Redzone_Carries': 4,
    '20_Plus_Yd_Rushes': 5,
    'Rush_Yds_Over_Expected_per_Carry': 3.5,
    'Offensive_Line_Rank': 3.5,
    'Team_Rush_Percent_2023': 3.5
}

# Invert rankings to give more importance to lower rankings
inverse_rankings = {k: 1/v for k, v in rankings.items()}

# Normalize the weights so they sum to 1
total = sum(inverse_rankings.values())
weights = {k: v/total for k, v in inverse_rankings.items()}

# Calculate a weighted score for each player
data['Score'] = data.apply(lambda row: sum(row[stat] * weights[stat] for stat in rankings.keys()), axis=1)

# Rank the players based on their scores
data['Rank'] = data['Score'].rank(ascending=False)

# Sort the DataFrame by Rank
data_sorted = data.sort_values(by='Rank')

# Select relevant columns for the output
output_columns = ['Name', 'Team', 'Score', 'Rank']
final_output = data_sorted[output_columns]

# Print the entire DataFrame as a string
print(final_output.to_string(index=False))
'''
Results
                 Name Team    Score  Rank
  Christian_McCaffrey   SF 0.854945   1.0
       Bijan_Robinson  ATL 0.666685   2.0
       Kyren_Williams  LAR 0.661367   3.0
          Breece_Hall  NYJ 0.654727   4.0
       Saquon_Barkley  PHI 0.632128   5.0
         Jahmyr_Gibbs  DET 0.624648   6.0
            Joe_Mixon  HOU 0.619841   7.0
           James_Cook  BUF 0.612481   8.0
        Rachaad_White   TB 0.608817   9.0
   Travis_Etienne_Jr.  JAC 0.590786  10.0
        Isiah_Pacheco   KC 0.588167  11.0
        Derrick_Henry  BAL 0.567244  12.0
       Raheem_Mostert  MIA 0.564209  13.0
        De_Von_Achane  MIA 0.548455  14.0
        D_Andre_Swift  CHI 0.537663  15.0
         Tony_Pollard  TEN 0.533616  16.0
      Jonathan_Taylor  IND 0.524381  17.0
         James_Conner  ARI 0.521928  18.0
          Josh_Jacobs   GB 0.519960  19.0
         Najee_Harris  PIT 0.508955  20.0
         Alvin_Kamara   NO 0.501379  21.0
     David_Montgomery  DET 0.501087  22.0
          Jerome_Ford  CLE 0.489859  23.0
        Jaylen_Warren  PIT 0.488404  24.0
   Kenneth_Walker_III  SEA 0.472868  25.0
            Zack_Moss  CIN 0.467389  26.0
          Gus_Edwards  LAC 0.465739  27.0
     Javonte_Williams  DEN 0.460373  28.0
          Aaron_Jones  MIN 0.439567  29.0
        Austin_Ekeler  WAS 0.435260  30.0
      Ezekiel_Elliott  DAL 0.433058  31.0
   Brian_Robinson_Jr.  WAS 0.429810  32.0
        Chuba_Hubbard  CAR 0.423463  33.0
     Devin_Singletary  NYG 0.422854  34.0
  Rhamondre_Stevenson   NE 0.405264  35.0
       Tyler_Allgeier  ATL 0.392513  36.0
         Tyjae_Spears  TEN 0.389582  37.0
   Alexander_Mattison   LV 0.352100  38.0
       Khalil_Herbert  CHI 0.342748  39.0
          Zamir_White   LV 0.341317  40.0
    Jaleel_McLaughlin  DEN 0.329239  41.0
         Justice_Hill  BAL 0.327307  42.0
     Kenneth_Gainwell  PHI 0.326695  43.0
       Antonio_Gibson   NE 0.326269  44.0
          Rico_Dowdle  DAL 0.316509  45.0
            AJ_Dillon   GB 0.314593  46.0
        Samaje_Perine  DEN 0.309251  47.0
      Zach_Charbonnet  SEA 0.294763  48.0
          Ty_Chandler  MIN 0.292906  49.0
        Miles_Sanders  CAR 0.276878  50.0
      Keaton_Mitchell  BAL 0.274620  51.0
Clyde_Edwards_Helaire   KC 0.264073  52.0
           Nick_Chubb  CLE 0.257305  53.0
          Chase_Brown  CIN 0.255987  54.0
          Kareem_Hunt   FA 0.244759  55.0
        Dameon_Pierce  HOU 0.238098  56.0
      Elijah_Mitchell   SF 0.203828  57.0
    Pierre_Strong_Jr.  CLE 0.192859  58.0
      Emari_Demercado  ARI 0.189206  59.0
      Jamaal_Williams   NO 0.167694  60.0
  Chris_Rodriguez_Jr.  WAS 0.109528  61.0
'''