import unittest

import torch
import torchdiffeq

from problems import construct_problem, CosineODE
from odeint_tests import rel_error

class TestPIController(unittest.TestCase):
    def test_odeint(self):
        for no_reject in (False, True):
            eps = 3e-4
            with self.subTest(no_reject=no_reject):
                f, y0, t_points, sol = construct_problem(dtype=torch.float64,
                                                        device='cpu', ode='sine',
                                                        reverse=False)
                y = torchdiffeq.odeint(f, y0, t_points,
                                method='bosh3',
                                options={'is_pi_control': True, 'no_reject': no_reject})
                self.assertLess(rel_error(sol, y), eps)

    def test_derivative(self):
        f = CosineODE()
        t_points = torch.tensor([1.0, 8.0], dtype=torch.float64)
        y0 = f.y_exact(t_points)[0].clone().detach()
        #f, y0, t_points, _ = construct_problem(dtype=torch.float64,
        #                                                device='cpu', ode='sine',
        #                                                reverse=False)
        y, solver = torchdiffeq.odeint(f, y0, t_points,
                                method='bosh3',
                                options={'is_pi_control': True}, return_solver=True)
        loss = torch.sum(y)
        loss.backward()
        print(solver.beta_1.grad)
        print(solver.beta_2.grad)

if __name__ == '__main__':
    unittest.main()