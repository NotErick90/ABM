from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import AudienceMember

def all_agents(model):
    return list(model.agents)


def percent_standing(model):
    agents = all_agents(model)
    if len(agents) == 0:
        return 0
    return 100* sum(agent.standing for agent in agents) / len(agents)


def stick_in_the_muds(model):
    sitting = 100 - percent_standing(model)
    return sitting


def informational_efficiency(model): ##it seems like it would be based on the average number of runs since dichotomous outcome per run? including for completion
    if not model.stable:
        return None
    final_majority_standing = percent_standing(model) >= 50
    return 100 if final_majority_standing == model.initial_majority_standing else 0

def number_of_iterations(model):
    if not model.stable:
        return None
    return model.iterations_to_stability



class StandingOvationModel(Model):
    def __init__(
        self,
        rows=20,
        columns=20,
        threshold=0.5,
        neighborhood="five",
        update_rule="synchronous",
        seed=None,
    ):
        super().__init__(seed=seed)

        self.rows = rows
        self.columns = columns
        self.threshold = threshold
        self.neighborhood = neighborhood
        self.update_rule = update_rule
        self.grid = MultiGrid(rows, columns, torus=False)

        for row in range(rows):
            for col in range(columns):
                quality = self.random.random()
                agent = AudienceMember(self, quality)
                self.grid.place_agent(agent, (row, col)) ##auditorium with fixed placement, otherwise pretty similar in behavior to schelling
                ## bc they are deciding whether threshold is met based on their neighbors. although initial quality threshold then is very deterministic...
                ## maybe should add a way to decide where to place agents who stand so as to evaluate the influential locations. but again this isn't explicitly addressed. 
                ## But also in the canonical sort of thinking where those at the very front will be especially influential... there's no set limit?
                ## bc the more rows there are then the more influence those in front they have? will that effectively decrease the threshold?

        self.steps = 0
        self.stable = False
        self.iterations_to_stability = None
        self.previous_states = self.get_state()

        self.initial_majority_standing = percent_standing(self) >= 50 ##paper says share percentage ig but difficult bc it says that "horrendous" perception means no standing even if neighbors are?
    ## (top of pg.14) but still majority "heuristic" (hi psych) here (2nd column pg.14)

        self.datacollector = DataCollector(
            model_reporters={
                "Percent Standing": percent_standing,
                "Stick in the Muds": stick_in_the_muds,
                "Informational Efficiency": informational_efficiency,
                "Number of Iterations": number_of_iterations,
            }
        )

        self.datacollector.collect(self)

    def get_sm(self):
        sm = stick_in_the_muds(self)
        return sm if sm is not None else 0
    
    def get_ie(self):
        ie = informational_efficiency(self)
        return ie if ie is not None else 0
    
    def get_ni(self):
        ni = number_of_iterations(self)
        return self.iterations_to_stability if self.iterations_to_stability is not None else self.steps

    def get_state(self):
        return tuple(agent.standing for agent in list(self.agents))

    def step(self):
        agents = list(self.agents)

        old_states = self.get_state()

        if self.stable:
            self.datacollector.collect(self)
            return ## if stable then just keep collecting data but not changing state


        if self.update_rule == "synchronous":
            for agent in self.agents:
                agent.decide_next()
            for agent in self.agents:
                agent.advance()

        elif self.update_rule == "random asynchronous":
            agents = list(self.agents)
            self.random.shuffle(agents)
            for agent in agents:
                agent.decide_next()
                agent.advance()

        elif self.update_rule == "incentive-based asynchronous": ##the rule is very non-specific "an explicit ordering rule..."
            agents = sorted(
                list(self.agents),
                key=lambda agent: agent.incentive_to_switch(), 
                reverse=True,
            )
            for agent in agents:
                agent.decide_next()
                agent.advance()

        new_states = self.get_state()
        self.steps += 1
                
        if new_states  == old_states:
            self.stable = True
            self.iterations_to_stability = self.steps
            print("Stable at step", self.steps)


        self.previous_states = new_states

        self.datacollector.collect(self)