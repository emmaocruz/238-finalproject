import gradio as gr
import numpy as np
from qlearning import QLearning
import load_statcast
from pitch_perfect import PitchPerfect

def run_notebook(pitcher_name, batter_name):
    # Example: Simulate running part of the notebook
    zones = gr.Image("zones.png")
    pitcher = pitcher_name.split()
    batter = batter_name.split()

    try:
       data_pitcher = load_statcast.get_pitcher_data(pitcher[1], pitcher[0])
       obs_pitcher = p.get_obs(data_pitcher)
    except:
       return "Whoops, looks like that pitcher name was not valid."

    arsenal = list(data_pitcher['pitch_type'].drop_duplicates())

    try:
        data_batter = load_statcast.get_batter_data(batter[1], batter[0])
        obs_batter = p.get_obs(data_batter)
    except:
       return "Whoops, looks like that batter name was not valid."

    Qp = Q.copy()
    Qp = model.QLearn(Qp, obs_pitcher, 0.3)

    Qb = Qp.copy()
    Qb = model.QLearn(Qb, obs_batter, 0.3)

    seq = p.get_pitch_seq(Qb, arsenal)
    pitch_sequence = ""

    states = list(p.state_lookup)
    for i in range(len(seq)):
      pitch_sequence += (states[i] + ": "+ str(p.pitches[seq[i][0]]) + ", Zone " + str(seq[i][1]) + "\n")

    return pitch_sequence, zones

# start by initializing Q with all data
data = load_statcast.retrieve_data()
p = PitchPerfect(data)
model = QLearning(p)
Q = model.initialize_q(data)

title = "Pitch Perfect"
f = open("description.md")
desc = f.read()

interface = gr.Interface(theme=gr.themes.Soft(), fn=run_notebook, inputs=[gr.Textbox(label="Pitcher Name"), gr.Textbox(label="Batter Name")], outputs=[gr.Textbox(label="The predicted optimal pitch sequence is:"), gr.Image(label="Pitch Zones (for reference)")], title=title,
                description=desc)
interface.launch(debug=True)