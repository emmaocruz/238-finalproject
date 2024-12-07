import gradio as gr
import numpy as np
from qlearning import QLearning
import load_statcast
from pitch_perfect import PitchPerfect
import matplotlib.pyplot as plt

def run_notebook(pitcher_name, batter_name):
    # Example: Simulate running part of the notebook
    zones = gr.Image("zones.png")
    pitcher = pitcher_name.split()
    batter = batter_name.split()
    fig = plt.figure(figsize=(8, 30))

    print(f"Gathering pitcher data for pitcher {pitcher_name}")
    try:
       data_pitcher = load_statcast.get_pitcher_data(pitcher[1], pitcher[0])
       obs_pitcher = p.get_obs(data_pitcher)
    except:
       return "Whoops, looks like that pitcher name was not valid.", zones, fig
    print("Success!")

    arsenal = list(data_pitcher['pitch_type'].drop_duplicates())

    print(f"Gathering batter data for batter {batter_name}")
    try:
        data_batter = load_statcast.get_batter_data(batter[1], batter[0])
        obs_batter = p.get_obs(data_batter)
    except:
       return "Whoops, looks like that batter name was not valid.", zones, fig
    print("Success!")

    print("Optimizing Q for this pitcher and batter combo")
    Qp = Q.copy()
    Qp = model.QLearn(Qp, obs_pitcher, 0.3)

    Qb = Qp.copy()
    Qb = model.QLearn(Qb, obs_batter, 0.3)

    print("Calculating pitch sequence")
    seq = p.get_pitch_seq(Qb, arsenal)
    pitch_sequence = ""

    states = list(p.state_lookup)
    for i in range(len(seq)):
      pitch_sequence += (states[i] + ": "+ str(p.pitches[seq[i][0]]) + ", Zone " + str(seq[i][1]) + "\n")

    print("Done!")

    data, min, max = p.generate_heat_map(Qb, arsenal)
    plt.rcParams.update({'font.size': 8})
    for i in range(len(arsenal)):
        for j in range(12):
            ax = fig.add_subplot(3*len(arsenal), 4, 12 * i + j+1)
            im = ax.imshow(data[j, i, :, :], cmap="RdBu")
            ax.set_title(f'{arsenal[i]} in {states[j]}')
            ax.set_axis_off()
    return pitch_sequence, zones, fig

# start by initializing Q with all data
print("Retrieving all statcast data... (this should only happen once)")
data = load_statcast.retrieve_data()
print("Creating models")
p = PitchPerfect(data)
model = QLearning(p)
print("Initializing Q learning")
Q = model.initialize_q(data)

title = "Pitch Perfect"
f = open("description.md")
desc = f.read()

interface = gr.Interface(theme=gr.themes.Soft(), fn=run_notebook, inputs=[gr.Textbox(label="Pitcher Name"), gr.Textbox(label="Batter Name")], outputs=[gr.Textbox(label="The predicted optimal pitch sequence is:"), gr.Image(label="Pitch Zones (for reference)"), gr.Plot(label="Q values by pitch and count", format="png")], title=title,
                description=desc)
interface.launch(debug=True)