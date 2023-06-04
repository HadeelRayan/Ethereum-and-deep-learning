import sys
import time

import numpy as np
import EthereumFeesFeeModel as EthereumFeesFeeModel
from blockchain_mdps import *
import blockchain_mdps.base.solver.pto_solver as PTOSolver



def main() -> None:
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    t = time.time()
    print("time: ",t)

    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

    # for alpha in alphas:
    #     print(alpha)
    #     alpha = 0.25
    #     gamma = 0.5
    #     max_fork = 2
    #     nl = 150
    #     horizon = int(1e4)
    #     epsilon = 1e-5
    #     max_iter = 100000
    #
    #     fee = 0.25
    #     transaction_chance = 0.1
    #
    #     model = EthereumFeesFeeModel.EthereumFeesFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee,
    #                                 transaction_chance=transaction_chance, max_pool=max_fork)
    #
    #     # model = BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)
    #
    #     print("model:",model )
    #     # print(f'Honest: {alpha * (1 + fee * transaction_chance)}')
    #     print('Number of states:', model.state_space.size)
    #     solver = PTOSolver.PTOSolver(model, expected_horizon=horizon)
    #     p, r, iterations, _ = solver.calc_opt_policy(epsilon=epsilon, max_iter=max_iter, skip_check=True)
    #     print('Revenue in PT-MDP:', r)
    #
    #     # model.print_policy(p, solver.mdp.find_reachable_states(p), x_axis=1, y_axis=0, z_axis=2)
    #     rev = solver.mdp.calc_policy_revenue(p)
    #     print('Revenue in ARR-MDP:', rev)
    #
    #     print('Iterations:', iterations)
    #     print('Time elapsed:', time.time() - t)
    #     t = time.time()
    #     print("time: ", t)



    alpha = 0.25
    print(alpha)
    gamma = 0.5
    max_fork = 1
    nl = 150
    horizon = int(1e4)
    epsilon = 1e-5
    max_iter = 100000
    fee = 0.25
    transaction_chance = 0.1
    model = EthereumFeesFeeModel.EthereumFeesFeeModel(alpha=alpha, gamma=gamma, max_fork=max_fork, fee=fee,
                                    transaction_chance=transaction_chance, max_pool=max_fork)
    print("model:",model )
    # print(f'Honest: {alpha * (1 + fee * transaction_chance)}')
    print('Number of states:', model.state_space.size)
    solver = PTOSolver.PTOSolver(model, expected_horizon=horizon)
    p, r, iterations, _ = solver.calc_opt_policy(epsilon=epsilon, max_iter=max_iter, skip_check=True)
    print('Revenue in PT-MDP:', r)
    # model.print_policy(p, solver.mdp.find_reachable_states(p), x_axis=1, y_axis=0, z_axis=2)
    rev = solver.mdp.calc_policy_revenue(p)
    print('Revenue in ARR-MDP:', rev)

    print('Iterations:', iterations)
    print('Time elapsed:', time.time() - t)
    t = time.time()
    print("time: ", t)


if __name__ == '__main__':
    # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
    main()
