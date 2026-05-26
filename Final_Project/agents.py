from mesa import Agent
import numpy as np


class StudentAgent(Agent):

    def __init__(
        self,
        model,
        unique_id,
        grade,
        group,
        smoking=False,
        decision_type=None
    ):
        super().__init__(model)

        self.unique_id = unique_id
        self.grade = grade
        self.group = group
        self.smoking = smoking

        self.decision_type = (
            decision_type
            if decision_type is not None
            else self.assign_decision_type()
        )

        # These are the two internal history states. One builds when an
        # agent smokes, and the other builds when an agent keeps not smoking--but both are inherent to the same utility function
        # the power of coming back to things with a fresh mind 😭
        self.smoking_reinforcement = 0.0
        self.nonsmoking_reinforcement = (
            self.model.initial_nonsmoking_reinforcement
        )

        self.peer_pressure = 0.0
        self.smoking_peer_pressure = 0.0
        self.protective_peer_pressure = 0.0
        self.popularity = 0

        self.future_discount = 1.0
        self.forecast_accuracy = 1.0
        self.true_risk_sensitivity = 1.0
        self.risk_perception = 1.0

        self.assign_behavioral_profile()

    def assign_decision_type(self):
        r = self.random.random()

        if r < self.model.share_naive:
            return "naive_present_biased"

        if r < self.model.share_naive + self.model.share_sophisticated:
            return "sophisticated_present_biased"

        return "time_consistent"

    def assign_behavioral_profile(self):
        # This is how strongly the agent would care about smoking harm
        # if the risk was fully perceived and fully weighted.
        self.true_risk_sensitivity = self.random.uniform(
            self.model.risk_sensitivity_min,
            self.model.risk_sensitivity_max
        )

        if self.decision_type == "naive_present_biased":
            self.future_discount = self.random.uniform(
                self.model.naive_discount_min,
                self.model.naive_discount_max
            )

            self.forecast_accuracy = self.random.uniform(
                self.model.naive_forecast_min,
                self.model.naive_forecast_max
            )

            base_risk_perception = self.random.uniform(
                self.model.naive_risk_perception_min,
                self.model.naive_risk_perception_max
            )

        elif self.decision_type == "sophisticated_present_biased":
            self.future_discount = self.random.uniform(
                self.model.sophisticated_discount_min,
                self.model.sophisticated_discount_max
            )

            self.forecast_accuracy = self.random.uniform(
                self.model.sophisticated_forecast_min,
                self.model.sophisticated_forecast_max
            )

            base_risk_perception = self.random.uniform(
                self.model.sophisticated_risk_perception_min,
                self.model.sophisticated_risk_perception_max
            )

        else:
            self.future_discount = 1.0
            self.forecast_accuracy = 1.0

            base_risk_perception = self.random.uniform(
                self.model.time_consistent_risk_perception_min,
                self.model.time_consistent_risk_perception_max
            )

        # Risk perception is the amount of the health risk that actually
        # enters the agent's decision. This is where information bias lives.
        self.risk_perception = (
            self.model.risk_bias_strength
            * base_risk_perception
        )
        self.risk_perception = min(max(self.risk_perception, 0), 1)

    def update_peer_pressure(self):
        neighbors = list(self.model.network.neighbors(self.unique_id))

        if len(neighbors) == 0:
            self.peer_pressure = 0.0
            self.smoking_peer_pressure = 0.0
            self.protective_peer_pressure = 0.0
            return

        smoking_neighbors = sum(
            self.model.students[n].smoking
            for n in neighbors
        )

        # The same neighborhood can create two different social signals:
        # smoker friends push smoking upward, while non-smoker friends can
        # make not smoking easier to maintain.
        self.smoking_peer_pressure = smoking_neighbors / len(neighbors)
        self.protective_peer_pressure = 1 - self.smoking_peer_pressure
        self.peer_pressure = self.smoking_peer_pressure

    def update_popularity(self):
        self.popularity = len(list(self.model.network.neighbors(self.unique_id)))

    def current_smoking_benefit(self):
        smoking_social_utility = (
            self.model.smoking_peer_influence_weight
            * self.smoking_peer_pressure
        )

        protective_social_utility = (
            self.model.protective_peer_influence_weight
            * self.protective_peer_pressure
        )

        smoking_history_utility = (
            self.model.smoking_reinforcement_weight
            * self.smoking_reinforcement
        )

        nonsmoking_history_utility = (
            self.model.nonsmoking_reinforcement_weight
            * self.nonsmoking_reinforcement
        )

        return (
            smoking_social_utility
            - protective_social_utility
            + smoking_history_utility
            - nonsmoking_history_utility
        )

    def perceived_health_cost(self):
        return (
            self.true_risk_sensitivity
            * self.risk_perception
            * self.future_discount
        )

    def expected_future_addiction_cost(self):
        future_stock_if_smoke = (
            self.smoking_reinforcement
            + self.model.reinforcement_growth
        )
        perceived_future_stock = future_stock_if_smoke * self.forecast_accuracy
        return perceived_future_stock * self.future_discount

    def smoking_utility(self):
        return (
            self.current_smoking_benefit()
            - self.perceived_health_cost()
            - self.expected_future_addiction_cost()
            - self.model.smoking_threshold
        )

    def evaluate_smoking(self):
        utility = self.smoking_utility()
        probability = 1 / (1 + np.exp(-utility))

        if self.smoking:
            # Quitting is still possible, but it gets harder as smoking
            # becomes more reinforced by past behavior.
            quit_probability = (
                self.model.quit_rate
                * (
                    1
                    - (
                        self.smoking_reinforcement
                        / self.model.max_smoking_reinforcement
                    )
                )
            )
            quit_probability = min(max(quit_probability, 0), 1)

            if self.random.random() < quit_probability:
                self.smoking = False
        else:
            if self.random.random() < probability:
                self.smoking = True

    def update_reinforcement(self):
        # This is the individual path-dependence piece. It is separate from
        # peers: agents can become more stable non-smokers even when friends
        # do not directly cause that stability.
        if self.smoking:
            self.smoking_reinforcement += self.model.reinforcement_growth
            self.nonsmoking_reinforcement *= 1 - self.model.nonsmoking_decay
        else:
            self.smoking_reinforcement *= 1 - self.model.reinforcement_decay
            self.nonsmoking_reinforcement += self.model.nonsmoking_growth

        self.smoking_reinforcement = min(
            max(self.smoking_reinforcement, 0),
            self.model.max_smoking_reinforcement
        )

        self.nonsmoking_reinforcement = min(
            max(self.nonsmoking_reinforcement, 0),
            self.model.max_nonsmoking_reinforcement
        )

    def receive_information_intervention(self, strength):
        # Information correction only changes perceived risk. It does not
        # directly remove peer pressure or undo smoking reinforcement.
        self.risk_perception += strength * (1 - self.risk_perception)
        self.risk_perception = min(max(self.risk_perception, 0), 1)

    def step(self):
        self.update_peer_pressure()
        self.update_popularity()
        self.evaluate_smoking()
        self.update_reinforcement()
