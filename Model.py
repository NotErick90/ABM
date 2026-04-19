import mesa
import math

from rich_click import group
from Agents import HouseholdAgent
from mesa.discrete_space import OrthogonalMooreGrid


class SegregationModel(mesa.Model):
    def __init__(
        self,
        width=25,
        height=25,
        n_agents=250,
        n_districts=8,
        seed=None,
        homophily = 0.30,
        wealth_scale = 1,
        group_0_wealth_min = 40,
        group_0_wealth_max = 120,
        group_1_wealth_min = 40,
        group_1_wealth_max = 120,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_districts = n_districts

        self.homophily = homophily
        self.happy = 0 
        self.happy_group_0  = 0
        self.happy_group_1 = 0
        
        self.wealth_scale = wealth_scale

        self.group_0_wealth_min = group_0_wealth_min
        self.group_0_wealth_max = group_0_wealth_max    
        self.group_1_wealth_min = group_1_wealth_min
        self.group_1_wealth_max = group_1_wealth_max
        
        

        self.grid = OrthogonalMooreGrid((self.width, self.height), torus=False, random=self.random)

        self.district_seeds = self._generate_district_seeds()
        self.cell_district = self._assign_districts()

        self.district_price = {}
        for d in range(self.n_districts):
            self.district_price[d] = self.random.randint(40,120)

        self.district_layer = self.grid.create_property_layer(
            "district",
            default_value=0.0,
            dtype=float
        )

        for x in range(self.width):
            for y in range(self.height):
                self.district_layer.data[x, y] = self.cell_district[(x, y)]

        self._create_agents()

    def _generate_district_seeds(self):
        seeds = []
        used = set()

        while len(seeds) < self.n_districts:
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            if (x, y) not in used:
                seeds.append((x, y))
                used.add((x, y))

        return seeds

    def _assign_districts(self):
        cell_district = {}

        for x in range(self.width):
            for y in range(self.height):
                nearest_district = min(
                    range(len(self.district_seeds)),
                    key=lambda i: (x - self.district_seeds[i][0]) ** 2
                    + (y - self.district_seeds[i][1]) ** 2
                )
                cell_district[(x, y)] = nearest_district

        return cell_district
    
    def cell_price(self,cell):
        district_id = self.cell_district[cell.coordinate]
        return self.district_price[district_id]

    def _create_agents(self):
        all_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.random.shuffle(all_cells)

        for i in range(self.n_agents):
            group = 0 if i < self.n_agents / 2 else 1
            
            if group == 0:
                base_wealth = self.random.randint(self.group_0_wealth_min, self.group_0_wealth_max)
            else:
                base_wealth = self.random.randint(self.group_1_wealth_min, self.group_1_wealth_max)
                
            wealth = int(base_wealth * self.wealth_scale) ##scale it up so that there's more runs basically but also significance bc of opportunities to move
            agent = HouseholdAgent(self, group, wealth)
            agent.cell = self.grid.find_nearest_cell( all_cells[i]) 


    def group_counts(self):
        g0 = sum(1 for a in self.agents if a.type == 0)
        g1 = sum(1 for a in self.agents if a.type == 1)
        return g0, g1

    def global_happiness_rate(self):
        return self.happy / self.n_agents if self.n_agents > 0 else 0

    def group_0_happiness_rate(self):
        g0, _ = self.group_counts()
        return self.happy_group_0 / g0 if g0 > 0 else 0

    def group_1_happiness_rate(self):
        _, g1 = self.group_counts()
        return self.happy_group_1 / g1 if g1 > 0 else 0
    

    ##relative wealth remaining bc relevant in a way, especially if there was a different utility function to health
    ##remaining bc since we can manipulate teh distribution (either more variability or just one having less wealth)
    ## then simply showing who has less wealth would be mechanical
    def relative_remaining_wealth(self,agent_type):
        group = [a for a in self.agents if a.type == agent_type]
        
        if not group: 
            return 0
        

        current_wealth = sum(a.wealth for a in group)
        initial_wealth = sum(a.initial_wealth for a in group)

        if initial_wealth == 0:
                return 0
        return round(100 * current_wealth/initial_wealth, 2)
    

    def update_district_prices(self):
        district_wealth = {d: [] for d in range(self.n_districts)}

        for agent in self.agents:
            district_id = self.cell_district[agent.cell.coordinate]
            district_wealth[district_id].append(agent.wealth)

        for d in range(self.n_districts): ##update district price based on a percentage of avg wealth 
            #price is updated as a proportion of the average wealth calculation, could have made this slider but im tired
            #also did it this way so that updating the wealth scale doesn't become futile as district price would immediately increase.
            if district_wealth[d]:
                avg_wealth = sum(district_wealth[d]) / len(district_wealth[d])
                target_price = 0.30*avg_wealth
                old_price = self.district_price[d]
                
                if target_price > old_price:
                    new_price = old_price + 0.10 *(target_price-old_price)
                    self.district_price[d] = math.ceil(new_price)
        

    def step(self):
        self.happy = 0
        self.happy_group_0 = 0
        self.happy_group_1 = 0
        self.agents.shuffle_do("move")
        self.update_district_prices()

        empty_cells = list(self.grid.empties.cells)
        stuck = 0 
        for a in self.agents:
            can_move = any(self.cell_price(cell) <= a.wealth for cell in empty_cells)
            if not can_move:
                stuck += 1
        
        if stuck == self.n_agents:
            self.running = False ##stop run once there's no agents left that are able to move