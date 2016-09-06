import random
from q_value_net import QValueNet

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

DRAW = 3

# define number of checkers to win
NUM_TO_WIN = 4
NAMES = [' ', 'X', 'O']
WIN_REWARD = 10.0

class BoardGame(object):
    def __init__(self, size):
        self.width = size
        self.height = size
        self.board_format = self.gen_board_format()

    def gen_board_format(self):
        board_format = ""
        delim = "\n|" + "-" * (4 * self.width - 1) + '|\n'
        board_format += delim

        for i in range(self.height):
            f_row = []
            for j in range(self.width):
                f_row.append(" {" + str(i*self.width + j) + "} ")

            board_format += "|" + "|".join(f_row) + "|"
            board_format += delim

        return board_format

    def print_board(self, state):
        cells = []
        for i in range(self.height):
            for j in range(self.width):
                cells.append(NAMES[state[i][j]])

        print self.board_format.format(*cells)


class Agent(object):
    def __init__(self, game, order, learning=True):
        self.epsilon = 0.1
        self.game = game
        self.order = order
        self.learning = learning
        self.prev_state = None
        self.prev_score = 0.0
        self.alpha = 0.2
        pass

    def random_move(self, state):
        available = []

        for i in range(self.game.size):
            for j in range(self.game.size):
                if state[i][j] == EMPTY:
                    available.append((i, j))

        return random.choice(available)

    def episode_over(self, state, last_move):
        gameover = self.game.is_game_over(state, last_move)
        if gameover != EMPTY:
            # update weights of neunet
            value = WIN_REWARD
            output = [0, 0]
            output[self.order - 1] = value
            output[self.order % 2] = -value
            self.game.train_state(state, output)
            self.prev_state = None
            self.prev_score = 0.0

    def greedy_move(self, state):
        maxval = -50000.0
        maxmove = None

        size = self.game.size

        for i in range(size):
            for j in range(size):
                if state[i][j] == EMPTY:
                    state[i][j] = self.order
                    val = self.lookup(state, (i, j))
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)


        self.backup(maxval)
        return maxmove

    def action(self):
        r = random.random()
        current_state = self.game.get_current_state()
        if r < self.epsilon:
            move = self.random_move(current_state)
        else:
            move = self.greedy_move(current_state)

        self.prev_state = current_state
        self.prev_score = self.lookup(self.prev_state, self.game.last_move)

        self.game.move(self.order, move)
        self.episode_over(self.game.get_current_state(), move)

    def look_ahead(self, state):
        size = self.game.size
        competitor = self.order%2 + 1

        for i in range(size):
            for j in range(size):
                if state[i][j] == EMPTY:
                    state[i][j] = competitor
                    if self.game.is_game_over(state, (i, j)) == competitor:
                        state[i][j] = EMPTY
                        return -WIN_REWARD
                    state[i][j] = EMPTY

        return self.game.get_state_value(state)[self.order-1]

    def lookup(self, state, move):
        gameover = self.game.is_game_over(state, move)
        if gameover == EMPTY:
            return self.look_ahead(state)
        else:
            return WIN_REWARD

    def backup(self, nextval):
        if not (self.prev_state is None) and self.learning:
            value = self.prev_score + self.alpha*(nextval - self.prev_score)
            output = [0, 0]
            output[self.order-1] = value
            output[self.order%2] = -value
            self.game.train_state(self.prev_state, output)


class Checker(object):
    def __init__(self, size):
        self.board = BoardGame(size)
        self.size = size
        self._current_state = self.empty_state()
        self.current_player = PLAYER_X
        self.last_move = None
        self.qvaluenn = QValueNet(size)

    def reset(self):
        self._current_state = self.empty_state()
        self.current_player = PLAYER_X
        self.last_move = None

    def get_current_state(self):
        return self._current_state

    def get_reward(self, gameover, current_player):
        if gameover == current_player:
            return WIN_REWARD
        elif gameover == EMPTY:
            return 0
        else:
            return -WIN_REWARD

    def move(self, player, move):
        self.last_move = move
        self._current_state[move[0]][move[1]] = player
        self.current_player = player

        pass

    def empty_state(self):
        state = [[EMPTY]*self.size for i in range(self.size)]
        return state

    def gen_random_state(self):
        state = self.empty_state()
        for i in range(self.size):
            for j in range(self.size):
                state[i][j] = random.randint(EMPTY, PLAYER_O)
        return state

    # X_player always plays first
    def get_last_move(self, state):
        count_x = 0
        count_o = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == PLAYER_O: count_o += 1
                if state[i][j] == PLAYER_X: count_x += 1

        if count_o == count_x:
            return PLAYER_O

        return PLAYER_X

    # last_move is a tuple to present the coordinate of the last move
    # the previous game state isn't always over yet.
    def is_game_over(self, state, last_move):
        # check horizontal
        def get_range(x, y, size, delta_x, delta_y):
            l_p = [(x, y)]
            t_x = x
            t_y = y
            count = 0
            while True:
                t_x += delta_x
                t_y += delta_y
                if t_x < 0 or t_x >= size:
                    break
                if t_y < 0 or t_y >= size:
                    break
                count += 1
                if count > NUM_TO_WIN:
                    break

                l_p.append((t_x, t_y))

            t_x = x
            t_y = y
            count = 0
            while True:
                t_x -= delta_x
                t_y -= delta_y
                if t_x < 0 or t_x >= size:
                    break
                if t_y < 0 or t_y >= size:
                    break
                count += 1
                if count > NUM_TO_WIN:
                    break

                l_p.append((t_x, t_y))

            if delta_x != 0:
                l_p = sorted(l_p, key=lambda x:x[0])
            else:
                l_p = sorted(l_p, key=lambda x:x[1])

            return l_p

        def is_win(state, list_points, player):
            #print list_points
            if len(list_points) < NUM_TO_WIN:
                return EMPTY
            i = 0
            p = list_points[i]
            while i < len(list_points):
                while state[p[0]][p[1]] != player:
                    i += 1
                    if i > (len(list_points) - NUM_TO_WIN): return EMPTY
                    p = list_points[i]

                j = i
                p = list_points[j]
                while state[p[0]][p[1]] == player and j < len(list_points):
                    j += 1
                    if j == len(list_points): break
                    p = list_points[j]

                if (j - i) >= NUM_TO_WIN:
                    return player

                i = j

            return EMPTY

        if not last_move: return EMPTY

        for (delta_x, delta_y) in zip([0,1,1,1], [1,0,1,-1]):
            list_points = get_range(last_move[0], last_move[1], self.size, delta_x, delta_y)
            winner = is_win(state, list_points, state[last_move[0]][last_move[1]])
            if winner != EMPTY:
                return winner

        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == EMPTY: return EMPTY

        return DRAW

    def convert_state(self, state):
        v = []
        for i in range(self.size):
            for j in range(self.size):
                v.append(state[i][j])

        return v

    # add neural network here
    # return the approximation of the state-value winning of the input state
    def get_state_value(self, state):
        return self.qvaluenn.predict(self.convert_state(state))

    # add a state to training the neural network
    def train_state(self, state, output):
        c_in = self.convert_state(state)
        self.qvaluenn.train(c_in, output)

    def play(self, agent_x, agent_o, monitor=False):
        def print_winner(winner):
            if winner == PLAYER_X:
                return 'X'
            elif winner == PLAYER_O:
                return 'O'
            else:
                return 'DRAW'
        i = 0
        winner = self.is_game_over(self._current_state, self.last_move)
        while winner == EMPTY:
            if monitor:
                print "\n"
                print "State-value:"
                print self.get_state_value(self._current_state)
                print "\n"
                self.board.print_board(self._current_state)
                raw_input()

            if i%2 == 0: agent_x.action()
            else: agent_o.action()

            winner = self.is_game_over(self._current_state, self.last_move)
            i += 1

        if monitor:
            print "Game Over, winner:%s!\n" % print_winner(winner)
            print "State-value:"
            print self.get_state_value(self._current_state)
            self.board.print_board(self._current_state)
        self.reset()
        pass

if __name__ == '__main__':
    game = Checker(5)
    bot_x = Agent(game, PLAYER_X, learning=True)
    bot_o = Agent(game, PLAYER_O, learning=True)


    print "Trainning...\n"
    bot_x.epsilon = 0.5
    bot_o.epsilon = 0.5
    for i in range(20000):
        if (i % 10) == 0:
            print 'Iteration:%d\n' % i
        if i >= 10000:
            bot_x.epsilon = 0.1
            bot_o.epsilon = 0.1
        game.play(bot_x, bot_o, monitor=False)


    print "Testing...\n"
    for i in range(100):
        game.play(bot_x, bot_o, monitor=True)

    '''
    state = [[2,1,2,1,1], [1,0,0,1,1], [2,2,1,2,1], [2,2,1,2,1], [2,2,0,2,0]]
    last_move = (0, 4)
    game = Checker(5)
    game.board.print_board(state)
    print game.is_game_over(state, last_move)
    '''
