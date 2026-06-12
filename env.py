import numpy as np
from game_constants import mrx_start_positions, detective_start_positions




class MrX():
    def __init__(self, mrx_start_pos):
        self.mrx_pos = mrx_start_pos
        self.mrx_tickets = {"black": 5,
                            "double": 2,
                            "taxi": 10,
                            "bus": 10,
                            "metro": 10}
        self.ticket_history = []
        

class Detective():
    def __init__(self, detective_start_pos):
        self.detective_pos = detective_start_pos
        self.detective_tickets = {"metro": 4,
                                  "bus": 8,
                                  "taxi": 11}

class environment():
    def __init__(self, detective_amount):
        self.detective_amount = detective_amount
        self.detectives = []

    def setup_game(self):
        #First select mrx starting pos and initiate player
        start_pos = np.random.choice(mrx_start_positions)
        self.mrx = MrX(start_pos)

        #Initiate detectives
        detective_positions = np.random.choice(
        detective_start_positions,
        size=self.detective_amount,
        replace=False)
        for i in range(self.detective_amount):
            new_detective = Detective(detective_positions[i])
            self.detectives.append(new_detective)

    def mrx_state(self):
        ticket_state = self.mrx.mrx_tickets
        pos = self.mrx.mrx_pos
        return ticket_state, pos
        
    def detective_state(self, detective_id):
        ticket_state = self.detectives[detective_id].detective_tickets
        pos = self.detectives[detective_id].detective_pos
        return ticket_state, pos

    def apply_mrx_move(self, action_id, next_pos, transport, use_black):
        self.mrx.mrx_pos = next_pos
        if use_black == False:
            self.mrx.mrx_tickets[transport] -= 1
        else:
            self.mrx.mrx_tickets['black'] -= 1
        


    def apply_detective_move(self, detective_id,action_id,next_pos,transport):
        current_detective = self.detectives[detective_id]
        current_detective.detective_pos = next_pos
        current_detective.detective_tickets[transport] -= 1
        self.mrx.mrx_tickets[transport] += 1



