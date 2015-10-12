import casadi as ca
import casadi.tools as cat

__author__ = 'belousov'


class Planner:

    # ========================================================================
    #                            Simple planning
    # ========================================================================
    @classmethod
    def create_plan(cls, model, initial_plan=0):
        # Degrees of freedom for the optimizer
        V = cat.struct_symSX([
            (
                cat.entry('X', repeat=model.n+1, struct=model.x),
                cat.entry('U', repeat=model.n, struct=model.u)
            )
        ])

        # Box constraints
        [lbx, ubx] = cls._create_box_constraints(model, V)

        # Non-linear constraints
        [g, lbg, ubg] = cls._create_nonlinear_constraints(model, V)

        # Objective function
        J = cls._create_objective_function(model, V)

        # Formulate non-linear problem
        nlp = ca.SXFunction('nlp', ca.nlpIn(x=V), ca.nlpOut(f=J, g=g))
        op = {'linear_solver': 'ma97'}
        solver = ca.NlpSolver('solver', 'ipopt', nlp, op)

        # Solve
        sol = solver(x0=initial_plan, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        return V(sol['x'])

    @staticmethod
    def _create_nonlinear_constraints(model, V):
        g, lbg, ubg = [], [], []
        for k in range(model.n):
            # Multiple shooting
            [xk_next] = model.F([V['X', k], V['U', k]])
            g.append(xk_next - V['X', k+1])
            lbg.append(ca.DMatrix.zeros(model.nx))
            ubg.append(ca.DMatrix.zeros(model.nx))

            # Control constraints
            g.append(V['U', k, 'v'] -\
                     model.v1 - model.v2 * ca.cos(V['U', k, 'theta']))
            lbg.append(-ca.inf)
            ubg.append(0)
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg]

    @staticmethod
    def _create_objective_function(model, V):
        [final_cost] = model.cl([V['X', model.n]])
        running_cost = 0
        for k in range(model.n):
            [stage_cost] = model.c([V['X', k], V['U', k]])
            running_cost += stage_cost
        return final_cost + running_cost

    # ========================================================================
    #                          Belief space planning
    # ========================================================================
    @classmethod
    def create_belief_plan(cls, model, initial_plan=0):
        # Degrees of freedom for the optimizer
        V = cat.struct_symSX([
            (
                cat.entry('X', repeat=model.n+1, struct=model.x),
                cat.entry('U', repeat=model.n, struct=model.u)
            )
        ])

        # Box constraints
        [lbx, ubx] = cls._create_box_constraints(model, V)

        # Non-linear constraints
        [g, lbg, ubg] = cls._create_belief_nonlinear_constraints(model, V)

        # Objective function
        J = cls._create_belief_objective_function(model, V)

        # Formulate non-linear problem
        nlp = ca.SXFunction('nlp', ca.nlpIn(x=V), ca.nlpOut(f=J, g=g))
        op = {'linear_solver': 'ma97'}
        solver = ca.NlpSolver('solver', 'ipopt', nlp, op)

        # Solve
        sol = solver(x0=initial_plan, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        return V(sol['x'])

    @staticmethod
    def _create_belief_nonlinear_constraints(model, V):
        """Non-linear constraints for planning"""
        bk = cat.struct_SX(model.b)
        bk['S'] = model.b0['S']
        g, lbg, ubg = [], [], []
        for k in range(model.n):
            # Belief propagation
            bk['m'] = V['X', k]
            [bk_next] = model.BF([bk, V['U', k]])
            bk_next = model.b(bk_next)

            # Multiple shooting
            g.append(bk_next['m'] - V['X', k+1])
            lbg.append(ca.DMatrix.zeros(model.nx))
            ubg.append(ca.DMatrix.zeros(model.nx))

            # Control constraints
            g.append(V['U', k, 'v'] -\
                     model.v1 - model.v2 * ca.cos(V['U', k, 'theta']))
            lbg.append(-ca.inf)
            ubg.append(0)

            # Advance time
            bk = bk_next
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg]

    @staticmethod
    def _create_belief_objective_function(model, V):
        # Simple cost
        running_cost = 0
        for k in range(model.n):
            [stage_cost] = model.c([V['X', k], V['U', k]])
            running_cost += stage_cost
        [final_cost] = model.cl([V['X', model.n]])

        # Uncertainty cost
        running_uncertainty_cost = 0
        bk = cat.struct_SX(model.b)
        bk['S'] = model.b0['S']
        for k in range(model.n):
            # Belief propagation
            bk['m'] = V['X', k]
            [bk_next] = model.BF([bk, V['U', k]])
            bk_next = model.b(bk_next)
            # Accumulate cost
            [stage_uncertainty_cost] = model.cS([bk_next])
            running_uncertainty_cost += stage_uncertainty_cost
            # Advance time
            bk = bk_next
        [final_uncertainty_cost] = model.cSl([bk_next])

        return running_cost + final_cost +\
               running_uncertainty_cost + final_uncertainty_cost

    # ========================================================================
    #                            Common functions
    # ========================================================================
    @staticmethod
    def _create_box_constraints(model, V):
        lbx = V(-ca.inf)
        ubx = V(ca.inf)

        # Control limits
        model._set_control_limits(lbx, ubx)

        # State limits
        model._set_state_limits(lbx, ubx)

        # Initial state
        lbx['X', 0] = ubx['X', 0] = model.m0

        return [lbx, ubx]




















