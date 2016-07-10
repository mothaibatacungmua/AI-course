import random


def min_conflicts(self):
    if not self.options.has_key('max_step'):
        self.options['max_step'] = 1000

    self.count = 0

    def count_conflicts(self, state, pos, value):
        s = state[:]
        s[pos] = value
        return self.count_collision_at(s, pos)

    def find_min_conflict_value(self, state, pos):
        t = [x for x in range(0,self.n)]
        t = sorted(t, key=lambda x:count_conflicts(self, state, pos, x))
        tt = [x for x in t if count_conflicts(self, state, pos, x)==count_conflicts(self, state, pos, t[0])]

        return tt[random.randint(0, len(tt)-1)]

    def search(self):
        state = self.problem[:]

        for i in range(0, self.options['max_step']):
            self.count = self.count + 1

            conflict_vars = self.get_conflict_vars(state)
            if len(conflict_vars) == 0:
                return state, True

            var = conflict_vars[random.randint(0, len(conflict_vars)-1)]
            min_v = find_min_conflict_value(self, state, var)

            state[var] = min_v

        return state, False

    return search(self)
    pass