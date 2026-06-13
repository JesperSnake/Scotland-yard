import numpy as np
from game_constants import mrx_start_positions, detective_start_positions, reveal_rounds




class MrX():
    def __init__(self, mrx_start_pos):
        self.mrx_pos = mrx_start_pos
        self.mrx_tickets = {"black": 5,
                            "double": 2,
                            "taxi": 4,
                            "bus": 3,
                            "metro": 2}
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
        self.last_reveal_location = None
        self.last_reveal_round = 0

    def setup_game(self):
        self.detectives = []
        #First select mrx starting pos and initiate player
        start_pos = int(np.random.choice(mrx_start_positions))
        self.mrx = MrX(start_pos)
        self.last_reveal_location = start_pos
        self.last_reveal_round = 0
        #Initiate detectives
        detective_positions = np.random.choice(
        detective_start_positions,
        size=self.detective_amount,
        replace=False)
        for i in range(self.detective_amount):
            new_detective = Detective(int(detective_positions[i]))
            self.detectives.append(new_detective)

    def mrx_state(self, round):
        return {
            "round": int(round),
            "mr_x_location": int(self.mrx.mrx_pos),
            "mr_x_tickets": self.mrx.mrx_tickets.copy(),
            "mr_x_ticket_history": self.mrx.ticket_history.copy(),
            "detective_locations": [int(d.detective_pos) for d in self.detectives],
            "detective_tickets": [d.detective_tickets.copy() for d in self.detectives],
            "last_reveal_location": int(self.last_reveal_location),
            "moves_since_reveal": int(round - self.last_reveal_round),
        }

        
    def detective_state(self, detective_id, round):
        current_detective = self.detectives[detective_id]
        return {
            "round": int(round),
            "detective_id": int(detective_id),
            "my_position": int(current_detective.detective_pos),
            "my_tickets": current_detective.detective_tickets.copy(),
            "detective_locations": [int(d.detective_pos) for d in self.detectives],
            "detective_tickets": [d.detective_tickets.copy() for d in self.detectives],
            "mr_x_ticket_history": self.mrx.ticket_history.copy(),
            "mr_x_last_revealed_position": int(self.last_reveal_location),
            "moves_since_last_reveal": int(round - self.last_reveal_round),
        }

    def apply_mrx_move(self, action_id, next_pos, transport, use_black, round):
        next_pos = int(next_pos)
        ticket_used = "black" if use_black else transport

        if use_black == False:
            self.mrx.mrx_tickets[transport] -= 1
        else:
            self.mrx.mrx_tickets['black'] -= 1

        if (round + 1) in reveal_rounds:
            self.last_reveal_location = next_pos
            self.last_reveal_round = int(round)
            self.mrx.ticket_history = []
        else:
            self.mrx.ticket_history.append(ticket_used)
        self.mrx.mrx_pos = next_pos
        


    def apply_detective_move(self, detective_id,action_id,next_pos,transport):
        current_detective = self.detectives[detective_id]
        current_detective.detective_pos = next_pos
        current_detective.detective_tickets[transport] -= 1
        self.mrx.mrx_tickets[transport] += 1


