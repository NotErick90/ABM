from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from model import StandingOvationModel
import solara

from model import (StandingOvationModel, percent_standing, stick_in_the_muds, informational_efficiency, number_of_iterations)
def agent_portrayal(agent):
    return {
        "color": "tab:blue" if agent.standing else "lightgray",
        "size": 45,
        "marker": "s" if agent.standing else "o",
    }


model_params = {
    "rows": {
        "type": "SliderInt",
        "value": 20,
        "label": "Rows",
        "min": 5,
        "max": 30,
        "step": 1,
    },
    "columns": {
        "type": "SliderInt",
        "value": 20,
        "label": "Columns",
        "min": 5,
        "max": 30,
        "step": 1,
    },
    "threshold": {
        "type": "SliderFloat",
        "value": 0.5,
        "label": "Initial standing threshold",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "neighborhood": {
        "type": "Select",
        "value": "five",
        "label": "Neighborhood",
        "values": ["five", "cone"],
    },
    "update_rule": {
        "type": "Select",
        "value": "synchronous",
        "label": "Update rule",
        "values": ["synchronous", "random asynchronous","incentive-based asynchronous"],
    },
}


model = StandingOvationModel()

def stats_display(model):
    solara.Markdown(f"**Number of Iterations:** {model.get_ni()}"),
    solara.Markdown(f"**Stick in the Muds:** {model.get_sm():.2f}%"),
    solara.Markdown(f"**Informational Efficiency:** {model.get_ie():.0f}")



page = SolaraViz(
    model,
    components=[
        make_space_component(agent_portrayal),
        make_plot_component("Percent Standing"),
        stats_display
        

    ],
    model_params=model_params,
    name="Standing Ovation Model",
)

page