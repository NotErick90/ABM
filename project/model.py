import networkx as nx
import numpy as np
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from agents import StudentAgent


class SchoolSmokingModel(Model):

    def __init__(
        self,
        n_students=300,
        n_groups=8,
        initial_smoking_rate=0.10,
        p_in=0.08,
        p_out=0.01,
        share_naive=0.40,
        share_sophisticated=0.30,
        peer_influence_weight=2.0,
        addiction_reinforcement=0.5,
        abstinence_reinforcement=0.8,
        risk_bias_strength=1.0,
        smoking_threshold=1.5,
        quit_rate=0.10,
        addiction_growth=0.12,
        addiction_decay=0.20,
        max_addiction_stock=5,
        initial_abstinence_stock=1.0,
        abstinence_growth=0.08,
        abstinence_decay=0.10,
        max_abstinence_stock=5,
        information_intervention_step=None,
        information_intervention_strength=0.0,
        risk_sensitivity_min=0.5,
        risk_sensitivity_max=1.5,
        naive_discount_min=0.35,
        naive_discount_max=0.75,
        naive_forecast_min=0.25,
        naive_forecast_max=0.60,
        naive_risk_perception_min=0.30,
        naive_risk_perception_max=0.70,
        sophisticated_discount_min=0.35,
        sophisticated_discount_max=0.75,
        sophisticated_forecast_min=0.80,
        sophisticated_forecast_max=1.00,
        sophisticated_risk_perception_min=0.50,
        sophisticated_risk_perception_max=0.90,
        time_consistent_risk_perception_min=0.70,
        time_consistent_risk_perception_max=1.00,
        seed=None
    ):
        super().__init__(seed=seed)

        self.n_students = n_students
        self.n_groups = n_groups
        self.initial_smoking_rate = initial_smoking_rate
        self.p_in = p_in
        self.p_out = p_out

        self.share_naive = share_naive
        self.share_sophisticated = share_sophisticated

        self.peer_influence_weight = peer_influence_weight
        self.addiction_reinforcement = addiction_reinforcement
        self.abstinence_reinforcement = abstinence_reinforcement
        self.risk_bias_strength = risk_bias_strength
        self.smoking_threshold = smoking_threshold
        self.quit_rate = quit_rate

        self.addiction_growth = addiction_growth
        self.addiction_decay = addiction_decay
        self.max_addiction_stock = max_addiction_stock

        self.initial_abstinence_stock = initial_abstinence_stock
        self.abstinence_growth = abstinence_growth
        self.abstinence_decay = abstinence_decay
        self.max_abstinence_stock = max_abstinence_stock

        self.information_intervention_step = information_intervention_step
        self.information_intervention_strength = information_intervention_strength

        self.risk_sensitivity_min = risk_sensitivity_min
        self.risk_sensitivity_max = risk_sensitivity_max

        self.naive_discount_min = naive_discount_min
        self.naive_discount_max = naive_discount_max
        self.naive_forecast_min = naive_forecast_min
        self.naive_forecast_max = naive_forecast_max
        self.naive_risk_perception_min = naive_risk_perception_min
        self.naive_risk_perception_max = naive_risk_perception_max

        self.sophisticated_discount_min = sophisticated_discount_min
        self.sophisticated_discount_max = sophisticated_discount_max
        self.sophisticated_forecast_min = sophisticated_forecast_min
        self.sophisticated_forecast_max = sophisticated_forecast_max
        self.sophisticated_risk_perception_min = sophisticated_risk_perception_min
        self.sophisticated_risk_perception_max = sophisticated_risk_perception_max

        self.time_consistent_risk_perception_min = time_consistent_risk_perception_min
        self.time_consistent_risk_perception_max = time_consistent_risk_perception_max

        self.step_count = 0
        self.rng = np.random.default_rng(seed)

        # This is a friendship network, not a spatial grid.
        self.network = nx.Graph()
        self.network.add_nodes_from(range(n_students))
        self.grid = NetworkGrid(self.network)

        self.students = {}

        self.create_agents()
        self.create_network()

        # Fixed layout for visualization.
        self.position = nx.spring_layout(
            self.network,
            seed=seed
        )

        # Main outputs tracked during the simulation.
        self.datacollector = DataCollector(
            model_reporters={
                "Smoking rate": lambda m: m.smoking_rate(),
                "Naive smoking rate": lambda m: m.smoking_rate_by_type("naive_present_biased"),
                "Sophisticated smoking rate": lambda m: m.smoking_rate_by_type("sophisticated_present_biased"),
                "Time-consistent smoking rate": lambda m: m.smoking_rate_by_type("time_consistent"),
                "Average addiction stock": lambda m: m.average_addiction_stock(),
                "Average abstinence stock": lambda m: m.average_abstinence_stock(),
                "Average peer pressure": lambda m: m.average_peer_pressure(),
                "Smoking homophily": lambda m: m.smoking_homophily(),
                "Average risk perception": lambda m: m.average_risk_perception(),
                "Average future discount": lambda m: m.average_future_discount(),
                "Average forecast accuracy": lambda m: m.average_forecast_accuracy()
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def create_agents(self):

        # Create students and give each one a behavioral type.
        for i in range(self.n_students):

            group = self.rng.integers(0, self.n_groups)
            grade = self.rng.choice([9, 10, 11, 12])
            smoking = self.rng.random() < self.initial_smoking_rate

            agent = StudentAgent(
                model=self,
                unique_id=i,
                grade=grade,
                group=group,
                smoking=smoking
            )

            self.students[i] = agent
            self.grid.place_agent(agent, i)

            self.network.nodes[i]["group"] = group
            self.network.nodes[i]["grade"] = grade
            self.network.nodes[i]["smoking"] = smoking
            self.network.nodes[i]["decision_type"] = agent.decision_type

    def create_network(self):

        # Initial friendship ties are more likely within the same group/grade.
        for i in range(self.n_students):
            for j in range(i + 1, self.n_students):

                same_group = self.students[i].group == self.students[j].group
                same_grade = self.students[i].grade == self.students[j].grade

                p = self.p_in if same_group else self.p_out

                if same_grade:
                    p *= 1.5

                if self.rng.random() < p:
                    self.network.add_edge(i, j)

    def update_network_attributes(self):

        # Keep node attributes synced with the agent states.
        for i, agent in self.students.items():
            self.network.nodes[i]["smoking"] = agent.smoking
            self.network.nodes[i]["addiction_stock"] = agent.addiction_stock
            self.network.nodes[i]["abstinence_stock"] = agent.abstinence_stock
            self.network.nodes[i]["peer_pressure"] = agent.peer_pressure
            self.network.nodes[i]["popularity"] = agent.popularity
            self.network.nodes[i]["decision_type"] = agent.decision_type
            self.network.nodes[i]["risk_perception"] = agent.risk_perception
            self.network.nodes[i]["future_discount"] = agent.future_discount
            self.network.nodes[i]["forecast_accuracy"] = agent.forecast_accuracy

    def update_friendships(self):

        # Friendships evolve over time.
        # Same smoking status makes a tie more stable.
        for i, j in list(self.network.edges):

            same_smoking = self.students[i].smoking == self.students[j].smoking
            same_group = self.students[i].group == self.students[j].group

            drop_prob = 0.02

            if same_smoking:
                drop_prob *= 0.5
            else:
                drop_prob *= 2.0

            if same_group:
                drop_prob *= 0.75

            if self.rng.random() < drop_prob:
                self.network.remove_edge(i, j)

        nodes = list(self.students.keys())

        # New ties can also form, especially among similar students.
        for i in nodes:

            possible = self.rng.choice(
                nodes,
                size=5,
                replace=False
            )

            for j in possible:

                if i == j or self.network.has_edge(i, j):
                    continue

                same_group = self.students[i].group == self.students[j].group
                same_grade = self.students[i].grade == self.students[j].grade
                same_smoking = self.students[i].smoking == self.students[j].smoking

                p = 0.005

                if same_group:
                    p *= 5

                if same_grade:
                    p *= 2

                if same_smoking:
                    p *= 2

                if self.rng.random() < p:
                    self.network.add_edge(i, j)

    def apply_information_intervention(self):

        # Optional one-time information shock.
        # This moves perceived risk closer to full risk realization.
        if self.information_intervention_step is None:
            return

        if self.step_count != self.information_intervention_step:
            return

        for agent in self.students.values():
            agent.receive_information_intervention(
                self.information_intervention_strength
            )

    def smoking_rate(self):

        return np.mean([
            agent.smoking
            for agent in self.students.values()
        ])

    def smoking_rate_by_type(self, decision_type):

        agents = [
            agent
            for agent in self.students.values()
            if agent.decision_type == decision_type
        ]

        if len(agents) == 0:
            return 0

        return np.mean([
            agent.smoking
            for agent in agents
        ])

    def average_addiction_stock(self):

        return np.mean([
            agent.addiction_stock
            for agent in self.students.values()
        ])

    def average_abstinence_stock(self):

        return np.mean([
            agent.abstinence_stock
            for agent in self.students.values()
        ])

    def average_peer_pressure(self):

        return np.mean([
            agent.peer_pressure
            for agent in self.students.values()
        ])

    def average_risk_perception(self):

        return np.mean([
            agent.risk_perception
            for agent in self.students.values()
        ])

    def average_future_discount(self):

        return np.mean([
            agent.future_discount
            for agent in self.students.values()
        ])

    def average_forecast_accuracy(self):

        return np.mean([
            agent.forecast_accuracy
            for agent in self.students.values()
        ])

    def smoking_homophily(self):

        # Share of ties connecting students with the same smoking status.
        if self.network.number_of_edges() == 0:
            return 0

        same_smoking_edges = 0

        for i, j in self.network.edges:
            if self.students[i].smoking == self.students[j].smoking:
                same_smoking_edges += 1

        return same_smoking_edges / self.network.number_of_edges()

    def step(self):

        self.apply_information_intervention()

        for agent in self.students.values():
            agent.step()

        self.update_friendships()
        self.update_network_attributes()
        self.datacollector.collect(self)

        self.step_count += 1