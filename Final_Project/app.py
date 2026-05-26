import solara
import networkx as nx
from matplotlib.figure import Figure
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter
from model import SchoolSmokingModel


# GUI controls for the assumptions you are likely to vary to ensure that things are working generally as expected but obviously won't necessarily be the case in the batch run analysis
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
    "smoking_peer_influence_weight": Slider("Smoking Peer Influence", 2.0, 0.0, 8.0, 0.25),
    "protective_peer_influence_weight": Slider("Protective Peer Influence", 0.75, 0.0, 8.0, 0.25),
    "smoking_reinforcement_weight": Slider("Smoking Reinforcement", 0.5, 0.0, 5.0, 0.1),
    "nonsmoking_reinforcement_weight": Slider("Non-Smoking Reinforcement", 0.8, 0.0, 5.0, 0.1),
    "risk_bias_strength": Slider("Risk Perception Bias", 1.0, 0.0, 1.5, 0.05),
    "smoking_threshold": Slider("Initiation Threshold", 1.5, 0.0, 5.0, 0.1),
    "quit_rate": Slider("Baseline Quit Rate", 0.10, 0.0, 0.5, 0.01),
    "information_intervention_step": Slider("Information Intervention Step", 25, 0, 100, 1),
    "information_intervention_strength": Slider("Information Intervention Strength", 0.0, 0.0, 1.0, 0.05)
}


@solara.component
def NetPlot(model):
    update_counter.get()

    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()

    node_colors = []

    # Smokers are shown separately; non-smokers are colored by decision type.
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


def post_process_rate_plot(ax):
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")


def post_process_nonnegative_plot(ax):
    ax.set_ylim(ymin=0)
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")


def post_process_centered_plot(ax):
    ax.set_ylabel("Value")
    ax.axhline(0, linewidth=0.8)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")


SmokingRatePlot = make_plot_component(
    {
        "Smoking rate": "tab:red",
        "Naive smoking rate": "tab:orange",
        "Sophisticated smoking rate": "tab:green",
        "Time-consistent smoking rate": "tab:blue"
    },
    post_process=post_process_rate_plot
)

StockPlot = make_plot_component(
    {
        "Average smoking reinforcement": "tab:brown",
        "Average non-smoking reinforcement": "tab:cyan"
    },
    post_process=post_process_nonnegative_plot
)

NetworkPlot = make_plot_component(
    {
        "Average smoking peer exposure": "tab:red",
        "Average protective peer exposure": "tab:green",
        "Smoking homophily": "tab:purple",
        "Average net peer utility": "tab:gray"
    },
    post_process=post_process_centered_plot
)

DispositionPlot = make_plot_component(
    {
        "Average true risk sensitivity": "tab:blue",
        "Average risk perception": "tab:orange",
        "Average future discount": "tab:green",
        "Average forecast accuracy": "tab:purple",
        "Average perceived health cost": "tab:red",
        "Average expected future addiction cost": "tab:brown"
    },
    post_process=post_process_nonnegative_plot
)


def model_summary(model):
    return solara.Markdown(
        f"""
        **Smoking rate:** {model.smoking_rate():.2f}  
        **Naive smoking rate:** {model.smoking_rate_by_type("naive_present_biased"):.2f}  
        **Sophisticated smoking rate:** {model.smoking_rate_by_type("sophisticated_present_biased"):.2f}  
        **Time-consistent smoking rate:** {model.smoking_rate_by_type("time_consistent"):.2f}  
        **Average smoking reinforcement:** {model.average_smoking_reinforcement():.2f}  
        **Average non-smoking reinforcement:** {model.average_nonsmoking_reinforcement():.2f}  
        **Average smoking peer exposure:** {model.average_smoking_peer_exposure():.2f}  
        **Average protective peer exposure:** {model.average_protective_peer_exposure():.2f}  
        **Average net peer utility:** {model.average_net_peer_utility():.2f}  
        **Smoking homophily:** {model.smoking_homophily():.2f}  
        **Average true risk sensitivity:** {model.average_true_risk_sensitivity():.2f}  
        **Average risk perception:** {model.average_risk_perception():.2f}  
        **Average future discount:** {model.average_future_discount():.2f}  
        **Average forecast accuracy:** {model.average_forecast_accuracy():.2f}  
        **Average perceived health cost:** {model.average_perceived_health_cost():.2f}  
        **Average expected future addiction cost:** {model.average_expected_future_addiction_cost():.2f}
        """
    )


model = SchoolSmokingModel()


page = SolaraViz(
    model,
    components=[
        NetPlot,
        SmokingRatePlot,
        StockPlot,
        NetworkPlot,
        DispositionPlot,
        model_summary
    ],
    model_params=model_params,
    name="School Smoking Network Model"
)

page
