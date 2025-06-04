import json





def load_mission(mission_file):
    with open(mission_file, 'r') as file:
        d = json.load(file)

    mission = d["scenario_objective"]

    eoi = mission["entities_of_interest"]
    constaints = d["scenario_constraints"]
    
    return (mission, eoi,constaints)


