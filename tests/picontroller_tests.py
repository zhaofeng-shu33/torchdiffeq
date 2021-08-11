import unittest

import torch
import torchdiffeq

from problems import construct_problem
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
        f, y0, t_points, _ = construct_problem(dtype=torch.float64,
                                                        device='cpu', ode='sine',
                                                        reverse=False)
        y = torchdiffeq.odeint(f, y0, t_points,
                                method='bosh3',
                                options={'is_pi_control': True})
        loss = torch.sum(y)
        loss.backward()

if __name__ == '__main__':
    unittest.main()