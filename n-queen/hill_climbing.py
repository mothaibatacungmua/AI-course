import random
import heapq
import time


def hill_climbing_heuristic(self, state):
    return self.count_collisions(state)


def get_successors(self, state, heurs, old_pos, new_pos):
    h = []
    current = hill_climbing_heuristic(self, state)
    min_heur = current

    for i in range(0, self.n):
        for j in range(0, self.n):
            save = state[i]
            state[i] = j
            if old_pos == None or new_pos == None or self.check_collision((i, j), old_pos) or self.check_collision((i, j), new_pos):
                heurs[i][j] = hill_climbing_heuristic(self, state)

            state[i] = save

            if min_heur > heurs[i][j]:
                min_heur = heurs[i][j]
                heapq.heappush(h, (heurs[i][j], (i, j)))

    return (heurs, min_heur, h)


def extract_minimum_successors(self, queue, min):
    min_s = []
    i = heapq.heappop(queue)

    while i[0] == min:
        min_s.append(i[1])
        if len(queue) == 0:
            break
        i = heapq.heappop(queue)

    return min_s


def is_stuck(self, state, min_heur):
    return hill_climbing_heuristic(self, state) == min_heur


def hill_climbing(self, state):
    heurs = [[0 for i in range(0, self.n)]for j in range(0, self.n)]
    min_heur = self.n*(self.n-1)/2
    old_pos = None
    new_pos = None
    count_sideways = 0
    total = 0.0

    while count_sideways < 1000:
        if is_stuck(self, state, min_heur):
            count_sideways = count_sideways + 1

        heurs, min_heur, queue = get_successors(self, state, heurs, old_pos, new_pos)

        if min_heur == 0:
            m = heapq.heappop(queue)[1]
            state[m[0]] = m[1]
            return state, True

        if len(queue) == 0:
            return state, False

        min_s = extract_minimum_successors(self, queue, min_heur)
        r = min_s[random.randint(0, len(min_s)-1)]

        old_pos = (r[0], state[r[0]])
        new_pos = r
        state[r[0]] = r[1]

    return state, False


def random_restart_hill_climbing(self):
    self.count = 0

    def search(self):
        state = []
        goal = False

        for i in range(0, self.options['max_restart']):
            self.count = self.count + 1
            state = self.gen_rand_problem()
            state, goal = hill_climbing(self, state)

            if goal:
                return state, True

        return state, False

    return search(self)