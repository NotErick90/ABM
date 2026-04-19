import solara
from mesa.visualization import SolaraViz, make_space_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from Model import SegregationModel


def agent_portrayal(agent):
    color = "black" if agent.type == 0 else "gray"
    return AgentPortrayalStyle(
        color=color,
        marker="s",
        size=70
    )


def propertylayer_portrayal(layer):
    if layer.name == "district":
        return PropertyLayerStyle(
            colormap="Pastel1",
            alpha=0.5,
            colorbar=False,
            vmin=0,
            vmax=7
        )


def global_happiness_rate(model):
    solara.Markdown(f"**Global happiness:** {model.global_happiness_rate():.2%}")


def group_0_happiness_rate(model):
    solara.Markdown(f"**Group 0 happiness:** {model.group_0_happiness_rate():.2%}")


def group_1_happiness_rate(model):
    solara.Markdown(f"**Group 1 happiness:** {model.group_1_happiness_rate():.2%}")


model = SegregationModel()



space_component = make_space_component(
    agent_portrayal,
    propertylayer_portrayal,
    backend="matplotlib"
)





model_params = {
    "width": 25,
    "height": 25,
    "n_agents": {
        "type": "SliderInt",
        "value": 250,
        "label": "Number of agents",
        "min": 50,
        "max": 500,
        "step": 10,
    },
    "n_districts": {
        "type": "SliderInt",
        "value": 8,
        "label": "Number of districts",
        "min": 3,
        "max": 12,
        "step": 1,
    },
    "seed": None,
    "homophily": {
        "type": "SliderFloat",
        "value": 0.30,
        "label": "Homophily",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "wealth_scale": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Wealth Scale",
        "min": 0.5,
        "max": 10.0,
        "step": 0.1,
    },
    "group_0_wealth_min": {
        "type": "SliderInt",
        "value": 40,
        "label": "Group 0 wealth min",
        "min": 10,
        "max": 1200,
        "step": 10,
    },
    "group_0_wealth_max": {
        "type": "SliderInt",
        "value": 120,
        "label": "Group 0 wealth max",
        "min": 20,
        "max": 2000,
        "step": 10,
    },
    "group_1_wealth_min": {
        "type": "SliderInt",
        "value": 40,
        "label": "Group 1 wealth min",
        "min": 10,
        "max": 1200,
        "step": 10,
    },
    "group_1_wealth_max": {
        "type": "SliderInt",
        "value": 120,
        "label": "Group 1 wealth max",
        "min": 20,
        "max": 2000,
        "step": 10,
    },
}
def type0_remaining_wealth(model):
    solara.Markdown(f"**Group 0 Relative Remaining Wealth:** {model.relative_remaining_wealth(0):.2f}%")

def type1_remaining_wealth(model):
    solara.Markdown(f"**Group 1 Relative Remaining Wealth:** {model.relative_remaining_wealth(1):.2f}%")

def no_wealth_left(model):
    empty_cells = list(model.grid.empties.cells)

    stuck = 0
    for a in model.agents:
        can_move = any(model.cell_price(cell) <= a.wealth for cell in empty_cells)
        if not can_move:
            stuck += 1

    pct = 100 * stuck / model.n_agents if model.n_agents > 0 else 0
    solara.Markdown(f"**Agents Unable To Afford Any Move:** {pct:.2f}%")



def dashboard(model):
    solara.Markdown(f"**Global happiness:** {model.global_happiness_rate():.2%}")
    solara.Markdown(f"**Group 0 happiness:** {model.group_0_happiness_rate():.2%}")
    solara.Markdown(f"**Group 1 happiness:** {model.group_1_happiness_rate():.2%}")
    solara.Markdown(f"**Group 0 Relative Remaining Wealth:** {model.relative_remaining_wealth(0):.2f}%")
    solara.Markdown(f"**Group 1 Relative Remaining Wealth:** {model.relative_remaining_wealth(1):.2f}%")

    empty_cells = list(model.grid.empties.cells)
    stuck = 0
    for a in model.agents:
        can_move = any(model.cell_price(cell) <= a.wealth for cell in empty_cells)
        if not can_move:
            stuck += 1

    pct = 100 * stuck / model.n_agents if model.n_agents > 0 else 0
    solara.Markdown(f"**Agents Unable To Afford Any Move:** {pct:.2f}%")



page = SolaraViz(
    model,
    components=[
        (space_component, 0),
        (dashboard, 1),
    ],
    model_params=model_params,
    name="Segregation Geography Prototype",
)