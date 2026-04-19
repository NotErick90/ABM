from mesa.discrete_space import CellAgent


class HouseholdAgent(CellAgent):
    def __init__(self, model, agent_type, wealth):
        super().__init__(model)
        self.type = agent_type
        self.wealth = wealth
        self.happy = False
        self.initial_wealth = wealth

    def share_alike(self):
        neighbors = [agent for agent in self.cell.neighborhood.agents if agent is not self]

        if len(neighbors) == 0:
            return 0

        similar_neighbors = len([agent for agent in neighbors if agent.type == self.type])
        return similar_neighbors / len(neighbors)

    def move(self):
        share = self.share_alike()

        if share < self.model.homophily:
            self.happy = False

            current_price = self.model.cell_price(self.cell)

            empty_cells = list(self.model.grid.empties.cells)
            affordable_cells = [
                cell for cell in empty_cells
                if self.model.cell_price(cell) <= self.wealth
            ]

            if affordable_cells:
                new_cell = self.random.choice(affordable_cells)
                new_price = self.model.cell_price(new_cell)
                move_cost = new_price

                if move_cost <= self.wealth:
                    self.wealth -= move_cost
                    self.cell = new_cell
        else:
            self.happy = True
            self.model.happy += 1

            if self.type == 0:
                self.model.happy_group_0 += 1
            else: 
                self.model.happy_group_1 += 1



                