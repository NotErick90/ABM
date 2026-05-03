from mesa import Agent


class AudienceMember(Agent):
    def __init__(self, model, quality):
        super().__init__(model)
        self.quality = quality ## own signal about performance "q_ij" that oguht to be in [0,1]
        self.standing = quality >= model.threshold ## underlying response for dichotomous behavior "to stand or not to stand"
        self.next_standing = self.standing ## everyone decide then synchronous updating
    
 
    def visible_neighbors(self):
        row, col = self.pos
        seats = []

        if self.model.neighborhood == "five": ##3 in front and 1 on each side (total of 2)
            candidates = [
                (row, col - 2),
                (row, col - 1),
                (row, col + 1),
                (row, col + 2),
                (row - 1, col - 1),
                (row - 1, col),
                (row - 1, col + 1),
            ]

        else:
            candidates = [
                (row, col - 2),  ## cone for the triangle like perception of everyone in front and same side 
                (row, col - 1),  ##which means that those in front will have more total influence as they are propagating the signal thru others + can still be seen
                (row, col + 1), ## but they will have limited information/less influenced by others.
                (row, col + 2),
            ]

            for r in range(row - 1, -1, -1):
                distance = row - r
                for c in range(col - distance, col + distance + 1):
                    candidates.append((r, c))

        for seat in candidates:
            if not self.model.grid.out_of_bounds(seat):
                cell = self.model.grid.get_cell_list_contents([seat])
                seats.extend(cell)

        return seats

    def decide_next(self):
        neighbors = self.visible_neighbors()

        if not neighbors:
            self.next_standing = self.standing ## no neighbors to influence decision--> apply decision
            return

        standing_count = sum(neighbor.standing for neighbor in neighbors) ##count based on method of choice
        share_standing = standing_count / len(neighbors)

        if share_standing > 0.5:  ##conformity aspect, only initial state of ovation will be driven by quality.
            self.next_standing = True ##but incentive base may be relevant for this bc given the immediacy of this spread, some runs are such that they will be switching between states and well...
        elif share_standing < 0.5:
            self.next_standing = False
        else:
            self.next_standing = self.random.choice([True, False]) ##random choice based on the paper although it kinda made results displayning hard.
    ##bc could revert back to self-judgment in this case? but how many are at this standstill? but then they might just be shifting back and forth--> lose coordination?
    ## so their computational model and their SOP differ crucially here bc in the computational this cannot happen...

    def incentive_to_switch(self):
        neighbors = self.visible_neighbors() 

        if not neighbors:
            return 0  
        opposite_count = sum(neighbor.standing != self.standing for neighbor in neighbors)
        return opposite_count / len(neighbors) ##switch order within a timestep such that those with strongest incentive will switch
    ## will probably propagate the response faster since earlier update means that susbequent updates will now also include those who just switched.
    ## feels kinda subtle, but within a step it might make the difference between at, or below/above threshold?

    def advance(self):
        self.standing = self.next_standing