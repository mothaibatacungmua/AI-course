
# [0, 0, 0, 0, 0, 0, 0, 0]
def normal_backtracking(self):
    self.__count = 0

    def select_unassigned(self, state, assignment):
        if len(assignment) == self.n:
            return None

        assigned_vars, unassigned_vars = self.get_unassigned_vars(assignment)

        return unassigned_vars[0], assigned_vars
        pass

    def search(self, state, assignment, domain_value):
        self.count = self.count + 1

        if len(assignment) == self.n:
            return assignment, True

        var, assigned_vars = select_unassigned(self, state, assignment)
        clone_domain = [k[:] for k in domain_value]

        for i in domain_value[var]:
            s = state[:]
            s[var] = i
            if self.contraints(s, assigned_vars, var):
                clone_domain[var].remove(i)
                assignment.append((var, i))
                assignment, goal = search(self, s, assignment, clone_domain)

                if goal:
                    return assignment, True
                assignment.pop()

        return assignment, False

    assignment, goal = search(self, self.problem, [], self.init_domain_value())
    ret = self.problem[:]
    for i in range(0, len(assignment)):
        ret[assignment[i][0]] = assignment[i][1]

    return ret, goal


def mrv_backtracking(self):
    self.count = 0

    def count_remain_values(self, state, domain_value, pos):
        count = 0
        s = state[:]
        for i in domain_value[pos]:
            s[pos] = i
            if self.check_collision_at(s, pos):
                count = count + 1

        return len(domain_value[pos]) - count

    def forward_checking(self, state, assignment, domain_value):
        s = state[:]
        assigned_vars, unassigned_vars = self.get_unassigned_vars(assignment)
        last_assigned = assignment[len(assignment)-1][0]

        for i in unassigned_vars:
            t = domain_value[i][:]
            for j in t:
                if self.check_collision((i, j), (last_assigned, state[last_assigned])):
                    domain_value[i].remove(j)

        return domain_value

    def remove_inconsistent_values(self, pos_a, pos_b, domain_value):
        removed = False
        t = domain_value[pos_a][:]
        for i in t:
            inconsistent = True
            for j in domain_value[pos_b]:
                if not self.check_collision((pos_a, i), (pos_b, j)):
                    inconsistent = False
                    break
            if inconsistent:
                domain_value[pos_a].remove(i)
                removed = True

        return removed

    def AC3(self, assignment, domain_value):
        assigned_vars, unassigned_vars = self.get_unassigned_vars(assignment)
        queue = []
        for i in range(0, len(unassigned_vars)):
            for j in range(i+1, len(unassigned_vars)):
                queue.insert(0, (unassigned_vars[i], unassigned_vars[j]))

        while len(queue) != 0:
            e = queue.pop()
            if remove_inconsistent_values(self, e[0], e[1], domain_value):
                for k in unassigned_vars:
                    if k != e[0] and k != e[1]:
                        queue.insert(0,(k, e[0]))

        return domain_value

    def select_mrv(self, state, assignment, domain_value):
        if len(assignment) == self.n:
            return None

        assigned_vars, unassigned_vars = self.get_unassigned_vars(assignment)

        unassigned_vars = sorted(unassigned_vars, key=lambda x: count_remain_values(self, state, domain_value, x))
        return unassigned_vars[0], assigned_vars

    def search(self, state, assignment, domain_value):
        self.count = self.count + 1
        if len(assignment) == self.n:
            return assignment, True

        var, assigned_vars = select_mrv(self, state, assignment, domain_value)
        clone_domain = [k[:] for k in domain_value]

        for i in domain_value[var]:
            s = state[:]
            s[var] = i

            if self.contraints(s, assigned_vars, var):
                clone_domain[var].remove(i)
                forward_domain = [k[:] for k in clone_domain]
                assignment.append((var, s[var]))

                #do forward checking if any
                if self.options.has_key('with_forward_checking') and self.options['with_forward_checking']:
                    forward_checking(self, s, assignment, forward_domain)

                #do AC3 to remove inconsistents if any
                if self.options.has_key('AC3') and self.options['AC3']:
                    AC3(self, assignment, forward_domain)

                assignment, goal = search(self, s, assignment, forward_domain)
                if goal:
                    return assignment, True
                assignment.pop()

        return assignment, False
        pass

    assignment, goal = search(self, self.problem, [], self.init_domain_value())
    ret = self.problem[:]
    for i in range(0, len(assignment)):
        ret[assignment[i][0]] = assignment[i][1]

    return ret, goal
    pass