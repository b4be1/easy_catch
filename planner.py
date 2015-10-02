import casadi as ca
import casadi.tools as cat

__author__ = 'belousov'


class Planner:

    @classmethod
    def create_plan(cls, model, n):
        # Degrees of freedom for the optimizer
        V = cat.struct_symSX([
            (
                cat.entry('X', repeat=n+1, struct=model.x),
                cat.entry('U', repeat=n, struct=model.u)
            )
        ])

        # Objective function
        J = cls._create_objective_function(model, V)

        # Box constraints
        [lbx, ubx] = cls._create_box_constraints(model, V)

        # Non-linear constraints
        [g, lbg, ubg] = cls._create_nonlinear_constraints(model, V)

        # Formulate non-linear problem
        nlp = ca.SXFunction('nlp', ca.nlpIn(x=V), ca.nlpOut(f=J, g=g))
        op = {'linear_solver': 'ma97'}
        solver = ca.NlpSolver('solver', 'ipopt', nlp, op)

        # Solve
        sol = solver(x0=0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        return V(sol['x'])

    @staticmethod
    def _create_objective_function(model, V):
        n = len(V['U'])
        [final_cost] = model.cl([V['X', n]])
        running_cost = 0
        for k in range(n):
            [stage_cost] = model.c([V['X', k], V['U', k]])
            running_cost += stage_cost
        return final_cost + running_cost

    @staticmethod
    def _create_box_constraints(model, V):
        lbx = V(-ca.inf)
        ubx = V(ca.inf)

        # Initial state
        lbx['X', 0] = ubx['X', 0] = model.x0

        # Control limits
        model._set_control_limits(lbx, ubx)

        return [lbx, ubx]

    # Should be used with simple planning only (no belief)
    @staticmethod
    def _create_nonlinear_constraints(model, V):
        """Non-linear constraints for planning"""
        n = len(V['U'])
        g, lbg, ubg = [], [], []
        for k in range(n):
            [xk_next] = model.F([V['X', k], V['U', k]])
            g.append(xk_next - V['X', k+1])
            lbg.append(ca.DMatrix.zeros(model.nx))
            ubg.append(ca.DMatrix.zeros(model.nx))
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg]

























