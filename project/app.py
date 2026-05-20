import solara
import networkx as nx
from matplotlib.figure import Figure
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter
from model import SchoolSmokingModel


# GUI controls for the main counterfactual assumptions.
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed"
    },
    "n_students": Slider("Students", 300, 50, 800, 50),
    "n_groups": Slider("Peer Groups", 8, 2, 20, 1),
    "initial_smoking_rate": Slider("Initial Smoking Rate", 0.10, 0.0, 0.5, 0.01),
    "share_naive": Slider("Naive Present-Biased Share", 0.40, 0.0, 1.0, 0.05),
    "share_sophisticated": Slider("Sophisticated Present-Biased Share", 0.30, 0.0, 1.0, 0.05),
    "p_in": Slider("Within-Group Tie Probability", 0.08, 0.01, 0.25, 0.01),
    "p_out": Slider("Between-Group Tie Probability", 0.01, 0.0, 0.10, 0.005),
    "peer_influence_weight": Slider("Peer Influence Strength", 2.0, 0.0, 8.0, 0.25),
    "addiction_reinforcement": Slider("Addiction Reinforcement", 0.5, 0.0, 5.0, 0.1),
    "abstinence_reinforcement": Slider("Abstinence Reinforcement", 0.8, 0.0, 5.0, 0.1),
    "risk_bias_strength": Slider("Risk Perception Bias", 1.0, 0.0, 1.5, 0.05),
    "smoking_threshold": Slider("Initiation Threshold", 1.5, 0.0, 5.0, 0.1),
    "quit_rate": Slider("Baseline Quit Rate", 0.10, 0.0, 0.5, 0.01)
}


@solara.component
def NetPlot(model):

    update_counter.get()

    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()

    node_colors = []

    # Smokers are red.
    # Non-smokers are colored by decision type.
    for node in model.network.nodes():
        agent = model.students[node]

        if agent.smoking:
            node_colors.append("red")
        elif agent.decision_type == "naive_present_biased":
            node_colors.append("orange")
        elif agent.decision_type == "sophisticated_present_biased":
            node_colors.append("green")
        else:
            node_colors.append("lightblue")

    # More connected students show up as larger nodes.
    node_sizes = [
        70 + model.students[node].popularity * 12
        for node in model.network.nodes()
    ]

    nx.draw(
        model.network,
        pos=model.position,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        width=0.5,
        alpha=0.85,
        with_labels=False
    )

    ax.set_title("School Friendship Network")
    ax.axis("off")

    solara.FigureMatplotlib(fig)


def post_process_lineplot(ax):

    ax.set_ylim(ymin=0)
    ax.set_ylabel("Value")
    ax.legend(
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left"
    )


# Main simulation outputs over time.
SmokingPlot = make_plot_component(
    {
        "Smoking rate": "tab:red",
        "Naive smoking rate": "tab:orange",
        "Sophisticated smoking rate": "tab:green",
        "Time-consistent smoking rate": "tab:blue",
        "Smoking homophily": "tab:purple",
        "Average addiction stock": "tab:brown",
        "Average abstinence stock": "tab:cyan",
        "Average risk perception": "tab:gray"
    },
    post_process=post_process_lineplot
)


def model_summary(model):

    return solara.Markdown(
        f"""
        **Smoking rate:** {model.smoking_rate():.2f}  
        **Naive smoking rate:** {model.smoking_rate_by_type("naive_present_biased"):.2f}  
        **Sophisticated smoking rate:** {model.smoking_rate_by_type("sophisticated_present_biased"):.2f}  
        **Time-consistent smoking rate:** {model.smoking_rate_by_type("time_consistent"):.2f}  
        **Average addiction stock:** {model.average_addiction_stock():.2f}  
        **Average abstinence stock:** {model.average_abstinence_stock():.2f}  
        **Average peer pressure:** {model.average_peer_pressure():.2f}  
        **Smoking homophily:** {model.smoking_homophily():.2f}  
        **Average risk perception:** {model.average_risk_perception():.2f}  
        **Average future discount:** {model.average_future_discount():.2f}  
        **Average forecast accuracy:** {model.average_forecast_accuracy():.2f}
        """
    )


model = SchoolSmokingModel()


page = SolaraViz(
    model,
    components=[
        NetPlot,
        SmokingPlot,
        model_summary
    ],
    model_params=model_params,
    name="School Smoking Network Model"
)

page