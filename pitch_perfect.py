import pandas as pd
import numpy as np

class PitchPerfect:
  def __init__(self, data):
    self.data = data
    self.state_lookup = {'0-0':0, '0-1':1, '0-2':2,'1-0':3, '1-1':4, '1-2':5, '2-0':6, '2-1':7, '2-2':8, '3-0':9, '3-1':10, '3-2':11}
    self.actions = data.index[0:134].droplevel(0)
    self.pitches = {'FA': 'Fastball', 'FT': 'Two-Seam Fastball', 'FC': 'Cutter', 'FS': 'Splitter', 'SI': 'Sinker', 'SL': 'Slider', 'CU': 'Curveball', 'KC': 'Knuckle Curve', 'EP': 'Eephus', 'CH': 'Changeup', 'SC': 'Screwball', 'KN': 'Knuckleball', 'ST': 'Sweeper', 'SV': 'Slurve', 'FF': 'Four-Seam Fastball'}

    # pitches where we don't have enough data need to be stored
    self.not_enough_data = set()
    for s in range(16):
      for a in range(134):
        # s represents our count
        pitch_type = self.actions[a][0]
        zone = self.actions[a][1]
        if (s, pitch_type, zone) not in self.data.index:
          self.not_enough_data.add((s, pitch_type, zone))    

  def T_hit(self, count, type, zone):
    # P(hit) = P(hit|swing)*P(swing)
    return self.data.loc[(count, type, zone)]['Swing %']*self.data.loc[(count, type, zone)]['Hit Prob']

  def T_strikeout(self, count, type, zone):
    k = (count, type, zone)
    # If two strikes
    if (count + 1) % 3 != 0:
      return 0
    else:
      swinging_out = self.data.loc[k]['Swing %']*(self.data.loc[(count, type, zone)]['Whiff %']) # P(swing and miss)
      taken_out = (1-self.data.loc[(count, type, zone)]['Swing %'])*self.data.loc[(count, type, zone)]['Strike Prob'] # P(take strike 3) = P(strike)*(1-P(swing))
      return swinging_out + taken_out

  def T_out(self, count, type, zone):
    # P(swing and not a hit)
    return self.data.loc[(count, type, zone)]['Swing %']*(1-self.data.loc[(count, type, zone)]['Hit Prob']-self.data.loc[(count, type, zone)]['Whiff %']-self.data.loc[(count, type, zone)]['Foul %'])

  def T_walk(self, count, type, zone):
    if (count < 9):
      return 0 # If there aren't any balls, P(walk)=0
    return (1-self.data.loc[(count, type, zone)]['Swing %'])*(1-self.data.loc[(count, type, zone)]['Strike Prob']) # P(take ball 4)

  def T_foul(self, count, type, zone):
    k = (count, type, zone)
    return self.data.loc[k]['Swing %']*(self.data.loc[k]['Foul %']) #P(swing)*P(foul|swing)

  def T_b(self, count, type, zone):
    if (count >= 9):
      return 0 # If ball with 3 balls, can't add to ball count
    return (1-self.data.loc[(count, type, zone)]['Swing %'])*(1-self.data.loc[(count, type, zone)]['Strike Prob']) # P(take ball)

  def T_s(self, count, type, zone):
    if (count + 1) % 3 == 0:
      return 0 # If strike with 2 strike, can't add to strike count
    # P(take strike)+P(foul or whiff)
    return (1-self.data.loc[(count, type, zone)]['Swing %'])*self.data.loc[(count, type, zone)]['Strike Prob'] + self.data.loc[(count, type, zone)]['Swing %']*(self.data.loc[(count, type, zone)]['Whiff %']+self.data.loc[(count, type, zone)]['Foul %'])

  def get_T(self):
    # Construct T(s' | s, a) table for all pitch counts
    # Possible states: 0-0, 0-1, 0-2, 1-0, 1-1, 1-2, 2-0, 2-1, 2-2, 3-0, 3-1, 3-2, HIT, OUT, WALK, STRIKEOUT = 16 total states
    # Possible actions: 134 total (per the block above)
    # Table will be 16x134x16 but very sparse
    T = np.zeros((16, 134, 16))

    for s in range(16):
      for a in range(134):
        # s represents our count
        pitch_type = self.actions[a][0]
        zone = self.actions[a][1]

        # if this pitch isn't "allowed" (i.e. not enough data probably)
        # we set the probability of a HIT to 1 to disincentivize this pitch
        # (also for the end states)
        if (s, pitch_type, zone) not in self.data.index:
          T[s, a, -4] = 1
          continue

        # hit probability
        T[s, a, -4] = self.T_hit(s, pitch_type, zone)

        # out probability
        T[s, a, -3] = self.T_out(s, pitch_type, zone)

        # walk probability (zero if state < 9)
        T[s, a, -2] = self.T_walk(s, pitch_type, zone)

        # strikeout probability (zero if state < 9)
        T[s, a, -1] = self.T_strikeout(s, pitch_type, zone)

        # strike probability (zero if we are already have 2 strikes)
        if (s + 1) % 3 != 0:
          T[s, a, s+1] = self.T_s(s, pitch_type, zone)

        # ball probability (zero if we already have 3 balls)
        if s < 9:
          T[s, a, s+3] = self.T_b(s, pitch_type, zone)

        # foul probability (only if we already have 2 strikes)
        if (s + 1) % 3 == 0:
          T[s, a, s] = self.T_foul(s, pitch_type, zone)
    return T

  def get_Rs(self):
    # using https://docs.google.com/spreadsheets/d/18iJ9rTnABwFry3Qc_rH9kkoaiC5gWJJgsdQDM6FS_54/edit?gid=0#gid=0
    # our rewards depend only on states

    R_s = np.zeros((16, 16))

    R_s[0, 3] = -0.036  # 0-0 -> 1-0
    R_s[0, 1] = 0.045  # 0-0 -> 0-1
    R_s[0, -3] = 0.24  # field out
    R_s[0, -4] = -0.79  # hit

    R_s[1, 4] = -0.025  # 0-1 -> 1-1
    R_s[1, 2] = 0.062  # 0-1 -> 0-2
    R_s[1, -3] = 0.25  # field out
    R_s[1, -4] = -0.81  # hit

    R_s[2, 5] = -0.014  # 0-2 -> 1-2
    R_s[2, -1] = 0.24  # strikeout
    R_s[2, -3] = 0.18  # field out
    R_s[2, -4] = -0.75  # hit

    R_s[3, 6] = -0.05  # 1-0 -> 2-0
    R_s[3, 4] = 0.055  # 1-0 -> 1-1
    R_s[3, -3] = 0.25  # field out
    R_s[3, -4] = -0.81  # hit

    R_s[4, 7] = -0.047  # 1-1 -> 2-1
    R_s[4, 5] = 0.073  # 1-1 -> 1-2
    R_s[4, -3] = 0.22  # field out
    R_s[4, -4] = -0.78  # hit

    R_s[5, 8] = -0.031  # 1-2 -> 2-2
    R_s[5, -1] = 0.25  # strikeout
    R_s[5, -3] = 0.19  # field out
    R_s[5, -4] = -0.76  # hit

    R_s[6, 9] = -0.120  # 2-0 -> 3-0
    R_s[6, 7] = 0.058  # 2-0 -> 2-1
    R_s[6, -3] = 0.28  # field out
    R_s[6, -4] = -0.81  # hit

    R_s[7, 10] = -0.226  # 2-1 -> 3-1
    R_s[7, 8] = 0.089  # 2-1 -> 2-2
    R_s[7, -3] = 0.25  # field out
    R_s[7, -4] = -0.79  # hit

    R_s[8, 11] = -0.112  # 2-2 -> 3-2
    R_s[8, -1] = 0.26  # strikeout
    R_s[8, -3] = 0.2  # field out
    R_s[8, -4] = -0.76  # hit

    R_s[9, 10] = 0.072  # 3-0 -> 3-1
    R_s[9, -2] = -0.53  # walk
    R_s[9, -3] = 0.34  # field out
    R_s[9, -4] = -0.89  # hit

    R_s[10, 11] = 0.083  # 3-1 -> 3-2
    R_s[10, -2] = -0.43  # walk
    R_s[10, -3] = 0.3  # field out
    R_s[10, -4] = -0.82  # hit

    R_s[11, -1] = 0.32  # strikeout
    R_s[11, -2] = -0.39  # walk
    R_s[11, -3] = 0.27  # field out
    R_s[11, -4] = -0.78  # hit

    return R_s

  def get_R(self):
    # Build rewards matrix R(s, a) which is 12x134

    '''
    Indices for pitch counts:
    0: 0-0
    1: 0-1
    2: 0-2
    3: 1-0
    4: 1-1
    5: 1-2
    6: 2-0
    7: 2-1
    8: 2-2
    9: 3-0
    10: 3-1
    11: 3-2
    12: HIT
    13: FIELD OUT
    14: WALK
    15: STRIKEOUT
    '''
    R_s = self.get_Rs()
    
    R_sas = np.tile(np.expand_dims(R_s, axis=1), (1, 134, 1))

    # To get R(s, a), we multiply by the transition probabilities and sum over the s' axis

    R = np.sum(R_sas * self.get_T(), axis=2)
    return R

  def get_obs(self, data):
    s_obs = []
    a_obs = []
    r_obs = []
    sp_obs = []

    R_s = self.get_Rs()

    for _,row in data.iterrows():
      if (row['pitch_type'], row['zone']) not in self.actions:
        continue

      if (row['pitch_type'], row['zone']) not in self.actions:
        continue

      a = list(self.actions).index((row['pitch_type'], row['zone']))
      s = self.state_lookup[str(row['balls'])+'-'+str(row['strikes'])]
      a_obs.append(a)
      s_obs.append(s)

      if row['description'] == 'hit_into_play':
        if row['events'] == 'field_out':
          sp_obs.append(13)
          r_obs.append(R_s[s, 13])
        else:
          sp_obs.append(12)
          r_obs.append(R_s[s, 12])

      elif row['events'] in ['strikeout', 'strikeout_double_play']:
        sp_obs.append(15)
        r_obs.append(R_s[s, 15])
      elif row['events'] in ['walk', 'hit_by_pitch']:
        sp_obs.append(14)
        r_obs.append(R_s[s, 14])
      elif row['description'] in ['called_strike', 'swinging_strike', 'missed_bunt']:
        sp = self.state_lookup[str(row['balls'])+'-'+str(row['strikes']+1)]
        sp_obs.append(sp)
        r_obs.append(R_s[s, sp])
      elif row['description'] == 'swinging_strike_blocked':
        if row['strikes'] == 2:
          sp_obs.append(15)
          r_obs.append(R_s[s, 15])
        else:
          sp = self.state_lookup[str(row['balls'])+'-'+str(row['strikes']+1)]
          sp_obs.append(sp)
          r_obs.append(R_s[s, sp])
      elif row['description'] in ['foul', 'foul_tip', 'foul_bunt']:
        if row['strikes'] == 2:
          sp_obs.append(s)
          r_obs.append(R_s[s, s])
        else:
          sp = self.state_lookup[str(row['balls'])+'-'+str(row['strikes']+1)]
          sp_obs.append(sp)
          r_obs.append(R_s[s, sp])
      elif row['description'] in ['ball', 'blocked_ball']:
        sp = self.state_lookup[str(row['balls']+1)+'-'+str(row['strikes'])]
        sp_obs.append(sp)
        r_obs.append(R_s[s, sp])
      else:
        print(row)

    obs = pd.DataFrame({
      's': s_obs,
      'a': s_obs,
      'r': r_obs,
      'sp': sp_obs
    })
    return obs

  def get_pitch_seq(self, Q, arsenal):
    pi = np.zeros(12)

    for a in self.actions:
      if a[0] not in arsenal:
        a = list(self.actions).index((a[0], a[1]))
        Q[:, a] = -10000

    for s in range(12):
      pi[s] = max(range(len(Q[s])), key=Q[s].__getitem__)
    return [self.actions[int(pi[i])] for i in range(12)]
  
  def generate_heat_map(self, Q, arsenal):
    min = float('inf')
    max = -float('inf')
    heat_map = np.zeros((12, len(arsenal), 16, 10))
  
    for a in self.actions:
      p = a[0]
      if p not in arsenal:
        continue
      p_ind = arsenal.index(p)
      zone = a[1]
      a = list(self.actions).index((a[0], a[1]))
      for s in range(12):
        Q_value = Q[s, a]

        if (s, p, zone) in self.not_enough_data:
          Q_value = float('nan')
        else:
          if Q_value > max:
            max = Q_value
          if Q_value < min:
            min = Q_value
        
        # Now we've extracted Q value, we can build the heat map
        if zone == 1:
          heat_map[s, p_ind, 2:6, 2:4] = Q_value  # zone 1
        elif zone == 2:
          heat_map[s, p_ind, 2:6, 4:6] = Q_value  # zone 2
        elif zone == 3:
          heat_map[s, p_ind, 2:6, 6:8] = Q_value  # zone 3
        elif zone == 4:
          heat_map[s, p_ind, 6:10, 2:4] = Q_value  # zone 4
        elif zone == 5:
          heat_map[s, p_ind, 6:10, 4:6] = Q_value  # zone 5
        elif zone == 6:
          heat_map[s, p_ind, 6:10, 6:8] = Q_value  # zone 6
        elif zone == 7:
          heat_map[s, p_ind, 10:14, 2:4] = Q_value  # zone 7
        elif zone == 8:
          heat_map[s, p_ind, 10:14, 4:6] = Q_value  # zone 8
        elif zone == 9:
          heat_map[s, p_ind, 10:14, 6:8] = Q_value  # zone 9
        elif zone == 11:
          heat_map[s, p_ind, 0:2, 0:5] = Q_value  # zone 11
          heat_map[s, p_ind, 0:8, 0:2] = Q_value  # zone 11
        elif zone == 12:
          heat_map[s, p_ind, 0:2, 5:] = Q_value  # zone 12
          heat_map[s, p_ind, 0:8, 8:] = Q_value  # zone 12
        elif zone == 13:
          heat_map[s, p_ind, 8:, 0:2] = Q_value  # zone 13
          heat_map[s, p_ind, 14:, 0:5] = Q_value  # zone 13
        elif zone == 14:
          heat_map[s, p_ind, 14:, 5:] = Q_value  # zone 14
          heat_map[s, p_ind, 8:, 8:] = Q_value  # zone 14            

    return heat_map, min, max
        