# Reinforcement learning Task 1
# Zura Isakadze

import copy
import random


class State(object):
    def __init__(self):
        self.player_score = 0
        self.dealer_score = 0
        self.terminal = False


class Actor:
    # enum
    dealer = "dealer"
    player = "player"


class Action:
    # enum
    stick = "stick"
    hit = "hit"


class Game(object):
    def __init__(self):
        pass

    def draw_card(self):
        num = random.randint(1, 10)
        if random.randint(1, 3) == 1:
            num *= -1
        return num

    def debug_card(self, num, actor):
        col = 'b' if num > 0 else 'r'
        num = abs(num)
        print '{} received card ({}, {})'.format(actor, col, num)

    def debug_score(self, actor, score):
        print "{}'s new score is {}".format(actor, score)
    
    def debug_result(self, res):
        # -1  player lost, 0 draw,  1 player won
        if res == 0:
            print 'Game was Drawn!'
        elif res == -1:
            print 'Player Lost'
        else:
            print 'Player won'


    def step(self, state, action):
        """ draw card and reutrn a tuple of new state and reward"""

        if state.terminal:
            raise ('Action initiated while in the terminal state')
        

        nstate = copy.deepcopy(state) # nstate is a new state
        reward = 0

        if action == Action.hit:
            print 'Player hits'
            card = self.draw_card()
            self.debug_card(card, Actor.player)
            nstate.player_score += card
            self.debug_score(Actor.player, nstate.player_score)
            if not 1 <= nstate.player_score <= 21:
                # player goes bust
                print 'Player goes bust'
                reward = -1
                self.debug_result(-1)
                nstate.terminal = True

        elif action == Action.stick:
            print 'Player sticks'
            # it's now dealer's turn

            nstate.terminal = True
            # we end up in terminal stage one way or enouther
            while 1 <= nstate.dealer_score < 17:
                # dealer decides to hit
                card = self.draw_card()
                self.debug_card(card, Actor.dealer)
                nstate.dealer_score += card
                self.debug_score(Actor.dealer, nstate.dealer_score)

            if not 1 <= nstate.dealer_score <= 21:
                print 'Dealer goes bust'
                reward = 1
            else:
                print 'Dealer sticks'
                if nstate.dealer_score > nstate.player_score:
                    reward = -1
                elif nstate.dealer_score == nstate.player_score:
                    reward = 0
                else:
                    reward = 1
                # reward right now denotes who won as well
            
            self.debug_result(reward)

        return (nstate, reward)



    def run(self):
        state = State()
        state.player_score = abs(self.draw_card())
        state.dealer_score = abs(self.draw_card())
        print 'player starts with {}'.format(state.player_score)
        print 'dealer starts with {}'.format(state.dealer_score)
        while not state.terminal:
            if random.randint(1, 3) <=2: 
                action = Action.hit
            else:
                action = Action.stick
            state, rew = self.step(state, action)
            print 'reward is {}'.format(rew)


        
game = Game()
game.run()

