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

        self.addiction_stock = 0.0
        self.abstinence_stock = self.model.initial_abstinence_stock

        self.peer_pressure = 0.0
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

        # This is the agent's underlying concern for smoking harms.
        # It is not the same as their perceived information.
        self.true_risk_sensitivity = self.random.uniform(
            self.model.risk_sensitivity_min,
            self.model.risk_sensitivity_max
        )

        if self.decision_type == "naive_present_biased":

            # Naive agents discount the future and underestimate future addiction.
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

            # Sophisticated agents still discount the future,
            # but they are better at anticipating future addiction.
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

            # Time-consistent agents do not have present bias or forecast bias,
            # but they can still have imperfect information about smoking risk.
            self.future_discount = 1.0
            self.forecast_accuracy = 1.0

            base_risk_perception = self.random.uniform(
                self.model.time_consistent_risk_perception_min,
                self.model.time_consistent_risk_perception_max
            )

        # Risk bias shifts how much of the true risk is actually perceived.
        self.risk_perception = (
            self.model.risk_bias_strength
            * base_risk_perception
        )

        self.risk_perception = min(
            max(self.risk_perception, 0),
            1
        )

    def update_peer_pressure(self):

        # Peer pressure is the share of friends who currently smoke.
        neighbors = list(
            self.model.network.neighbors(self.unique_id)
        )

        if len(neighbors) == 0:
            self.peer_pressure = 0
            return

        smoking_neighbors = sum(
            self.model.students[n].smoking
            for n in neighbors
        )

        self.peer_pressure = smoking_neighbors / len(neighbors)

    def update_popularity(self):

        # Popularity is just degree centrality here.
        self.popularity = len(
            list(
                self.model.network.neighbors(self.unique_id)
            )
        )

    def current_smoking_benefit(self):

        # Immediate benefit from smoking:
        # social pressure plus reinforcement from existing addiction stock.
        return (
            self.model.peer_influence_weight * self.peer_pressure
            + self.model.addiction_reinforcement * self.addiction_stock
        )

    def realized_future_cost(self):

        # This separates true concern from perceived risk.
        # Someone can care about health but still underestimate the risk.
        return (
            self.true_risk_sensitivity
            * self.risk_perception
            * self.future_discount
        )

    def expected_future_addiction_cost(self):

        # What the agent thinks addiction will look like after smoking now.
        # Naive agents underestimate this through lower forecast accuracy.
        future_stock_if_smoke = (
            self.addiction_stock
            + self.model.addiction_growth
        )

        perceived_future_stock = (
            future_stock_if_smoke
            * self.forecast_accuracy
        )

        return (
            perceived_future_stock
            * self.future_discount
        )

    def abstinence_protection(self):

        # Not smoking can also become self-reinforcing.
        # This helps avoid the unrealistic result where everyone eventually smokes.
        return (
            self.model.abstinence_reinforcement
            * self.abstinence_stock
        )

    def smoking_utility(self):

        # Latent utility of smoking.
        # Non-smoking is the outside option, normalized to zero.
        return (
            self.current_smoking_benefit()
            - self.realized_future_cost()
            - self.expected_future_addiction_cost()
            - self.abstinence_protection()
            - self.model.smoking_threshold
        )

    def evaluate_smoking(self):

        # Turn utility into a probability instead of a hard yes/no rule.
        utility = self.smoking_utility()

        probability = 1 / (1 + np.exp(-utility))

        if self.smoking:

            # Quitting gets harder as addiction stock rises.
            quit_probability = (
                self.model.quit_rate
                * (
                    1
                    - (
                        self.addiction_stock
                        / self.model.max_addiction_stock
                    )
                )
            )

            quit_probability = min(
                max(quit_probability, 0),
                1
            )

            if self.random.random() < quit_probability:
                self.smoking = False

        else:

            if self.random.random() < probability:
                self.smoking = True

    def update_stocks(self):

        # Smoking builds addictive capital.
        # Not smoking builds protective abstinence capital.
        if self.smoking:

            self.addiction_stock += self.model.addiction_growth

            self.abstinence_stock *= (
                1 - self.model.abstinence_decay
            )

        else:

            self.addiction_stock *= (
                1 - self.model.addiction_decay
            )

            self.abstinence_stock += (
                self.model.abstinence_growth
            )

        self.addiction_stock = min(
            max(self.addiction_stock, 0),
            self.model.max_addiction_stock
        )

        self.abstinence_stock = min(
            max(self.abstinence_stock, 0),
            self.model.max_abstinence_stock
        )

    def receive_information_intervention(self, strength):

        # Optional intervention: move perceived risk closer to full realization.
        self.risk_perception += (
            strength
            * (1 - self.risk_perception)
        )

        self.risk_perception = min(
            max(self.risk_perception, 0),
            1
        )

    def step(self):

        self.update_peer_pressure()
        self.update_popularity()
        self.evaluate_smoking()
        self.update_stocks()