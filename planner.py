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
            [xk_next] = model.F([V['X', k], V['U', k]])
            g.append(xk_next - V['X', k+1])
            lbg.append(ca.DMatrix.zeros(model.nx))
            ubg.append(ca.DMatrix.zeros(model.nx))
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg]

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
        [g, lbg, ubg, final_belief] =\
            cls._create_belief_nonlinear_constraints(model, V)

        # Objective function
        J = cls._create_objective_function(model, V)
        J += model.w_S * ca.trace(final_belief['S'])

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

            # Advance time
            bk = bk_next
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg, bk]

    # ========================================================================
    #                            Common functions
    # ========================================================================
    @staticmethod
    def _create_box_constraints(model, V):
        lbx = V(-ca.inf)
        ubx = V(ca.inf)

        # Initial state
        lbx['X', 0] = ubx['X', 0] = model.m0

        # Control limits
        model._set_control_limits(lbx, ubx)

        return [lbx, ubx]

    @staticmethod
    def _create_objective_function(model, V):
        [final_cost] = model.cl([V['X', model.n]])
        running_cost = 0
        for k in range(model.n):
            [stage_cost] = model.c([V['X', k], V['U', k]])
            running_cost += stage_cost
        return final_cost + running_cost




















