import pybaseball
from pybaseball import statcast_pitcher, statcast, playerid_lookup, statcast_batter
import math
import pandas as pd
import numpy as np
import pitch_perfect

def retrieve_data():
  pybaseball.cache.enable()
  d = statcast('2024-04-1', '2024-10-1', verbose = False)
  d = d[['pitch_type','zone', 'events', 'description', 'balls', 'strikes']]

  d00 = d[(d['balls']== 0) & (d['strikes']==0)][['pitch_type','zone', 'events', 'description']]
  d01 = d[(d['balls']== 0) & (d['strikes']==1)][['pitch_type','zone', 'events', 'description']]
  d02 = d[(d['balls']== 0) & (d['strikes']==2)][['pitch_type','zone', 'events', 'description']]
  d10 = d[(d['balls']== 1) & (d['strikes']==0)][['pitch_type','zone', 'events', 'description']]
  d11 = d[(d['balls']== 1) & (d['strikes']==1)][['pitch_type','zone', 'events', 'description']]
  d12 = d[(d['balls']== 1) & (d['strikes']==2)][['pitch_type','zone', 'events', 'description']]
  d20 = d[(d['balls']== 2) & (d['strikes']==0)][['pitch_type','zone', 'events', 'description']]
  d21 = d[(d['balls']== 2) & (d['strikes']==1)][['pitch_type','zone', 'events', 'description']]
  d22 = d[(d['balls']== 2) & (d['strikes']==2)][['pitch_type','zone', 'events', 'description']]
  d30 = d[(d['balls']== 3) & (d['strikes']==0)][['pitch_type','zone', 'events', 'description']]
  d31 = d[(d['balls']== 3) & (d['strikes']==1)][['pitch_type','zone', 'events', 'description']]
  d32 = d[(d['balls']== 3) & (d['strikes']==2)][['pitch_type','zone', 'events', 'description']]
  d_split = [d00, d01, d02, d10, d11, d12, d20, d21, d22, d30, d31, d32]

  totals = {}
  swings = {}
  whiffs = {}
  strikes = {}
  hits = {}
  fouls = {}

  for i in range(len(d_split)):
    for idx,pitch in d_split[i].iterrows():
      # Initialize all counts with 1 to avoid divide by zero
      if (i, pitch['pitch_type'], pitch['zone']) not in totals:
        totals[(i, pitch['pitch_type'], pitch['zone'])] = 1
        swings[(i, pitch['pitch_type'], pitch['zone'])] = 1
        whiffs[(i, pitch['pitch_type'], pitch['zone'])] = 1
        strikes[(i, pitch['pitch_type'], pitch['zone'])] = 1
        hits[(i, pitch['pitch_type'], pitch['zone'])] = 1
        fouls[(i, pitch['pitch_type'], pitch['zone'])] = 1

      totals[(i, pitch['pitch_type'], pitch['zone'])] += 1
      if pitch['description'] in ['swinging_strike', 'hit_into_play', 'foul']:
        swings[(i, pitch['pitch_type'], pitch['zone'])] += 1
      if pitch['description'] == 'swinging_strike':
        whiffs[(i, pitch['pitch_type'], pitch['zone'])] += 1
      if pitch['description'] == 'called_strike':
        strikes[(i, pitch['pitch_type'], pitch['zone'])] += 1
      if pitch['events'] in ['single', 'double', 'triple', 'home_run']:
        hits[(i, pitch['pitch_type'], pitch['zone'])] += 1
      if pitch['description'] == 'foul':
        fouls[(i, pitch['pitch_type'], pitch['zone'])] += 1

  pswing = {k: float(swings[k])/totals[k] for k in swings}
  pwhiff = {k: float(whiffs[k])/swings[k] for k in whiffs}
  phit = {k: float(hits[k])/swings[k] for k in hits}
  pfoul = {k: float(fouls[k])/swings[k] for k in fouls}
  pstrike = {k: float(strikes[k])/(totals[k] - swings[k] + 1) for k in fouls}

  d_all = pd.DataFrame({
      'Count': pd.Series(totals),
      'Swing %': pd.Series(pswing), # Probability that batter swings at pitch
      'Whiff %': pd.Series(pwhiff), # Probability that batter misses given that they swung
      'Hit Prob': pd.Series(phit), # Probability of hit given that batter swings
      'Strike Prob': pd.Series(pstrike), # Probability of strike given the batter takes pitch
      'Foul %': pd.Series(pfoul) # Probability of foul ball given that batter swings
  })

  d_all = d_all[d_all.Count > 30] # Keep only pitches with more than 30 observations over the season
  d_all.sort_index() # Probabilities for each count by pitch
  return d_all

def get_pitcher_data(last, first):
    id = playerid_lookup(last, first, fuzzy=True)['key_mlbam'][0]
    data = statcast_pitcher('2024-04-1', '2024-10-1', id)[['pitch_type','zone', 'events', 'description', 'balls', 'strikes']]
    return data[data['pitch_type'].notna()]

def get_batter_data(last, first):
  id = playerid_lookup(last, first, fuzzy=True)['key_mlbam'][0]
  data = statcast_batter('2024-04-1', '2024-10-1', id)[['pitch_type','zone', 'events', 'description', 'balls', 'strikes']]
  return data[data['pitch_type'].notna()]