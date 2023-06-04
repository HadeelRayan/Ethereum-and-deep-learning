import sys
from enum import Enum
from typing import Tuple
import numpy as np
from blockchain_mdps.base.blockchain_model import BlockchainModel
from blockchain_mdps.base.base_space.space import Space
from blockchain_mdps.base.state_transitions import StateTransitions
from blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from blockchain_mdps.base.state_transitions import StateTransitions

class EthereumFeesFeeModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int, fee: float, transaction_chance: float, max_pool: int):
        self.alpha = alpha
        self.gamma = 0.5
        self.max_fork = max_fork
        self.fee = fee
        self.transaction_chance = transaction_chance  # what is this?
        self.max_pool = max(max_pool, max_fork)
        self.block_reward = 1

        self.uncle_rewards = [0, 7.0 / 8, 6.0 / 8, 5.0 / 8, 4.0 / 8, 3.0 / 8, 2.0 / 8]
        self.nephew_reward = 1.0 / 32
        self._uncle_dist_b = len(self.uncle_rewards)  # always equal to 7
        self._honest_uncles_b = 2 ** (self._uncle_dist_b - 1)  # equal to 2^6

        self.Fork = self.create_int_enum('Fork', ['Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Adopt', 'Reveal', 'Wait', 'Match', 'Override'])
        #self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])  # indicates if we have a block or not
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])  # indicates if we have a block or not
        self.Transaction = self.create_int_enum('Transaction',['NoTransaction', 'With'])  # indicates if we have a Transactions or not

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha},{self.max_fork}, {self.fee}, {self.transaction_chance}, {self.max_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork, self.fee, self.transaction_chance, self.max_pool)

    """
    get_state_space:
    the state space in our model is (a,h,fork,r,ua,uh,pool)
    the simplified model is 10-tuple of the form
    (a,h,La,Lh,Ta,Th,fork,r,uh,ua,pool)
    a: list of flags representing whether each block in the miner’s chain holds a transaction.
    h: list of flags representing whether each block in the public’s chain holds a transaction.
    La and Lh represent the length of the miner chain and the public chain
    Ta: is the number of transaction in the miner’s chain
    Th: is the number of transaction in the public chain
    fork:the current case of the system : active \ relevant 
    r: 0 if the first block of the rational miner is still a secret , else the length of the public chain since the last fork
    uh: binary vector of 6 bits, denotes whether there are blocks by honest miner since the last fork that can be included as uncles
    ua: denotes whether there is a revealed block by the rational miner since the last fork
    pool: the number of available trans since the last fork
    """

    def get_state_space(self) -> Space:
        #   (a,h,La,Lh,Ta,Th,fork,r,uh,ua,pool)
        element = [self.Block, self.Transaction] * (self.max_fork) +[self.Block, self.Transaction] * (self.max_fork)\
                                                                    +[(0, self.max_fork), (0, self.max_fork)\
                                                                     ,(0, self.max_pool), (0, self.max_pool)\
                                                                     ,self.Fork, self._uncle_dist_b,
                                                                           self._honest_uncles_b,
                                                                           2, (0, self.max_pool)]



        # elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork, (0, self.max_pool),
        #                                                                    (0, self.max_fork), (0, self.max_fork),
        #                                                                    (0, self.max_pool), (0, self.max_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*element)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_fork))

    def get_initial_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (0,) * 4 + (self.Fork.Relevant,) + (0,) * 4

    def get_final_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2* self.max_fork + (-1,) * 4 + (self.Fork.Relevant,) + (-1,) * 4

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, int, int, int, int, Enum,
                                                                   int, int, int, int]:
        a = state[:2 * self.max_fork]
        h = state[2 * self.max_fork:4 * self.max_fork]
        length_a = state[-9]
        length_h = state[-8]
        transactions_a = state[-7]
        transactions_h = state[-6]
        fork = state[-5]
        r = state[-4]
        u_h = state[-3]
        u_a = state[-2]
        pool = state[-1]
        return a, h, length_a, length_h, transactions_a, transactions_h, fork, r, u_h, u_a, pool

    def create_empty_chain(self) -> tuple:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_fork

    def is_chain_valid(self, chain: tuple) -> bool:  # unchanged
        # Check chain length
        if len(chain) != self.max_fork * 2:
            return False

        # Check chain types
        valid_parts = sum(isinstance(block, self.Block) and isinstance(transaction, self.Transaction)
                          for block, transaction in zip(chain[::2], chain[1::2]))
        if valid_parts < self.max_fork:
            return False

        # Check the chain starts with blocks
        last_block = max([0] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.Exists])
        first_no_block = min([self.max_fork - 1]
                             + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.NoBlock])
        if last_block > first_no_block:
            return False

        # Check no transactions in non-blocks
        invalid_transactions = sum(block is self.Block.NoBlock and transaction is self.Transaction.With
                                   for block, transaction in zip(chain[::2], chain[1::2]))
        if invalid_transactions > 0:
            return False

        return True

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        a, h, length_a, length_h, transactions_a, transactions_h, fork, r, u_h, u_a, pool = self.dissect_state(state)
        if transactions_a < 0:
            return False
        #print("self.chain_length(a) < self.chain_transactions(a)", self.chain_length(a) , self.chain_transactions(a))
        if self.chain_transactions(h) !=transactions_h or self.chain_length(h)!=length_h\
            or self.chain_length(h)<=self.chain_transactions(h)\
            or self.chain_transactions(a) != transactions_a or self.chain_length(a) != length_a\
            or self.chain_length(a) < self.chain_transactions(a):
            return False

        return self.is_chain_valid(a) and self.is_chain_valid(h) \
               and length_a == self.chain_length(a) \
               and length_h == self.chain_length(h) \
               and 0 <= transactions_a == self.chain_transactions(a) <= pool \
               and 0 <= transactions_h == self.chain_transactions(h) <= pool \
               and (fork == self.Fork.Relevant or fork == self.Fork.Active) \
               and 0 <= u_a <= 1 and 0 <= r <= self.max_fork and 0 <= u_h <= self._honest_uncles_b \
               and 0 <= pool <= self.max_pool

    @staticmethod
    def truncate_chain(chain: tuple, truncate_to: int) -> tuple:
        return chain[:2 * truncate_to]

    def shift_back(self, chain: tuple, shift_by: int) -> tuple:
        return chain[2 * shift_by:] + (self.Block.NoBlock, self.Transaction.NoTransaction) * shift_by

    def chain_length(self, chain: tuple) -> int:
        return len([block for block in chain[::2] if block is self.Block.Exists])

    def chain_transactions(self, chain: tuple) -> int:
        return len([block for block, transaction in zip(chain[::2], chain[1::2])
                    if block is self.Block.Exists and transaction is self.Transaction.With])

    def add_block(self, chain: tuple, add_transaction: bool) -> tuple:
        transaction = self.Transaction.With if add_transaction else self.Transaction.NoTransaction
        index = self.chain_length(chain)
        chain = list(chain)
        chain[2 * index] = self.Block.Exists
        chain[2 * index + 1] = transaction
        return tuple(chain)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        # if  check_valid and not self.is_state_valid(state):
        #     transitions.add(self.final_state, probability=1, reward=self.error_penalty,allow_merging=True)
        #     return transitions

        if not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, length_a, length_h, transactions_a, transactions_h, fork, r, uh, ua, pool = self.dissect_state(state)
        action_type, action_param = action

        if action_type is self.Action.Illegal:
            #print("Illegaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaal",str(action_type))
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            return transitions

        if action_type is self.Action.Adopt:
            print("adopt :",state)
            print("action_param , length_h , self.max_fork", action_param ,length_h , self.max_fork)
            if 0 < action_param <= length_h <= self.max_fork:
                #print("heeeeeeeeeeello")
                accepted_transactions = self.chain_transactions(self.truncate_chain(h, action_param))

                num_of_uncles_included = min(2 * action_param, bin(uh).count('1')) + (int(length_a > 0 and r > 0))

                new_uh = ((int(bin(uh).replace('1', '0', 2 * action_param), 2)) << action_param) % self._honest_uncles_b

                # that's how we know if it has been referenced or not
                r_included = bool(2 * action_param - int(ua > 0) - bin(uh).count('1') > 0) and length_a > 0 and r > 0
                new_ua = int(length_a > 0 and not r_included)
                r_include_distance = max(r, 1 + int((int(ua > 0) + bin(uh).count('1')) / 2))

                # adding the penalty in case if we added the reward before referencing
                if ua > 0 and bin(uh).count('1') >= 2:
                    ua_not_included_penalty = -1.0 / 8
                else:
                    ua_not_included_penalty = 0

                #a = self.create_empty_chain()
                # ASALA a was a=a ,changed to self.shift_back(a, action_param)
                # DONE: TODO Why? a should be the empty chain

                print("self.create_empty_chain():",self.create_empty_chain())
                next_state = self.create_empty_chain() + self.shift_back(h, action_param) \
                             + (0, length_h - action_param, 0, transactions_h - accepted_transactions,
                                self.Fork.Relevant, 0, new_uh, new_ua, pool - accepted_transactions)
                transitions.add(next_state, probability=1,
                                reward=ua_not_included_penalty + self.uncle_rewards[r],
                                difficulty_contribution=(action_param + num_of_uncles_included))
            else:
                print("error_penalty:",self.final_state)
                next_state=self.final_state
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # reveal 1- like ethereum
        if action_type is self.Action.Reveal:
            if action_param == 1 and fork is self.Fork.Relevant and length_a <= self.max_fork and length_h <= self.max_fork \
                    and length_h < self._uncle_dist_b and r == 0 and length_a > 0 and length_h > 1:
                #ASALA : change a to self.shift_back(a, 1) , is right?
                # TODO No, it's wrong, it should stay the same
                print("reveal1 :", state)
                next_state = a + h + (length_a, length_h, transactions_a, transactions_h,
                                      self.Fork.Relevant, length_h, uh, ua, pool)

                transitions.add(next_state, probability=1, reward=0, difficulty_contribution=0)

            # reveal h - like match
            # TODO The addition is correct, but action_param <= length_a is redundant
            elif action_param == length_h and action_param <= length_a < self.max_fork and fork is self.Fork.Relevant:
                print("revealh :", state)
                if r > 0:
                    new_r = r
                elif length_h < self._uncle_dist_b:
                    new_r = length_h
                else:
                    new_r = 0
                #a, h, length_a, length_h, transactions_a, transactions_h, fork, r, u_h, u_a, pool = self.dissect_state(state)

                next_a_state1 = a + h + (length_a, length_h, transactions_a, transactions_h,
                                         self.Fork.Active, new_r, uh, ua, pool)

                transitions.add(next_a_state1, probability=1, reward=0, difficulty_contribution=int(new_r > 0))

            # reveal more than h - like override
            elif length_h < action_param <= length_a:
                print("reveal :", state)
                accepted_transactions = self.chain_transactions(self.truncate_chain(a, action_param))

                if ua == 1:
                    nephew_reward = self.nephew_reward
                else:
                    nephew_reward = 0

                if 0 < length_h < self._uncle_dist_b:
                    new_uh = ((uh << (action_param)) + 2 ** (action_param - 1)) % self._honest_uncles_b
                else:
                    new_uh = (uh << (length_h + 1)) % self._honest_uncles_b

                h = self.create_empty_chain()
                print("a befor",a)
                a = self.shift_back(a, action_param)
                print("a after", a)

                # TODO Why did you put transactions_h? it shouldb be 0
                next_state = a + h + (length_a - action_param, 0,
                                      transactions_a - accepted_transactions, 0,
                                      self.Fork.Relevant, 0, new_uh, 0, pool - accepted_transactions)

                transitions.add(next_state, probability=1,
                                reward=action_param + (accepted_transactions * self.fee) + nephew_reward,
                                difficulty_contribution=action_param)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)
        if action is self.Action.Wait:
            print("wait :", state)
            if fork is self.Fork.Relevant and length_a < self.max_fork and length_h < self.max_fork and \
                    action_param in [self.Transaction.With, self.Transaction.NoTransaction]:
               #asala removed: 0 <= action_param <= 1: TODO Good
                #new_trans_a = action_param == self.Transaction.With and transactions_a < pool
                new_trans_a = 0
                if transactions_a < pool:
                    new_trans_a=1
                new_trans_h = transactions_h < pool
                new_transaction_chance = self.transaction_chance if length_a >= length_h else 0
                next_a_state1 = self.add_block(a, new_trans_a) + h \
                                + (length_a + 1, length_h, transactions_a + new_trans_a, transactions_h,
                                   self.Fork.Relevant, r, uh, ua, min(pool + 1, self.max_pool))
                transitions.add(next_a_state1, probability=self.alpha * new_transaction_chance,
                                difficulty_contribution=1)

                next_a_state2 = self.add_block(a, new_trans_a) + h \
                                + (length_a + 1, length_h, transactions_a + new_trans_a, transactions_h,
                                   self.Fork.Relevant, r, uh, ua, pool)
                transitions.add(next_a_state2, probability=self.alpha * (1 - new_transaction_chance),
                                difficulty_contribution=1, allow_merging=True)

                new_transaction_chance = self.transaction_chance if length_a <= length_h else 0

                next_h_state1 = a + self.add_block(h, new_trans_h) \
                                + (length_a, length_h + 1, transactions_a, transactions_h + new_trans_h,
                                   self.Fork.Relevant, r, uh, ua, min(pool + 1, self.max_pool))
                transitions.add(next_h_state1, probability=(1 - self.alpha) * new_transaction_chance,
                                difficulty_contribution=1)

                next_h_state2 = a + self.add_block(h, new_trans_h) \
                                + (length_a, length_h, transactions_a, transactions_h + new_trans_h,
                                   self.Fork.Relevant, r, uh, ua, pool)
                transitions.add(next_h_state2, probability=(1 - self.alpha) * (1 - new_transaction_chance),
                                difficulty_contribution=1, allow_merging=True)

            elif fork is self.Fork.Active and 0 < length_h <= length_a < self.max_fork:
                if length_h < self._uncle_dist_b:
                    new_uh = ((uh << length_h) + 2 ** (length_h - 1)) % self._honest_uncles_b
                else:
                    new_uh = (uh << length_h) % self._honest_uncles_b

                if ua == 1:
                    nephew_reward = self.nephew_reward
                else:
                    nephew_reward = 0

                new_trans_a = transactions_a < pool
                new_transaction_chance = self.transaction_chance if length_a >= length_h else 0
                next_a_state = self.add_block(a, new_trans_a) + h \
                               + (length_a + 1, length_h, transactions_a + new_trans_a, transactions_h,
                                  self.Fork.Active, r, uh, ua, min(pool + 1, self.max_pool))
                transitions.add(next_a_state, probability=self.alpha * new_transaction_chance)

                next_a_state2 = self.add_block(a, new_trans_a) + h \
                                + (length_a + 1, length_h, transactions_a + new_trans_a, transactions_h,
                                   self.Fork.Active, r, uh, ua, pool)
                transitions.add(next_a_state2, probability=self.alpha * (1 - new_transaction_chance),
                                allow_merging=True)

                # in case the honest miner add to his chain
                new_trans_h = transactions_h < pool
                new_transaction_chance_h = self.transaction_chance if length_a <= length_h else 0
                next_h_state1 = a + self.add_block(h, new_trans_h) \
                                + (length_a, length_h + 1, transactions_a, transactions_h + new_trans_h,
                                   self.Fork.Relevant, r, uh, ua, min(pool + 1, self.max_pool))
                transitions.add(next_h_state1,probability=(1 - self.alpha) * (1 - self.gamma) * new_transaction_chance_h)

                next_h_state2 = a + self.add_block(h, new_trans_h) \
                                + (length_a, length_h + 1, transactions_a, transactions_h + new_trans_h,
                                   self.Fork.Relevant, r, uh, ua, pool)
                transitions.add(next_h_state2, probability=(1 - self.alpha) * (1 - self.gamma) * (1 - new_transaction_chance_h),
                                allow_merging=True)

                h = self.create_empty_chain()
                a = self.shift_back(a, length_h)
                accepted_transactions = self.chain_transactions(self.truncate_chain(a, length_h))

                #ASALA : is new_trans_h_a true ? when honest add a block to a , it was new_trans_h ?
                # TODO No, instead of transactions_a, it should be the number of transactions in
                #  a in the first length_h blocks only
                # TODO, also it is a boolean value, and you put it in the element which is a number of transactions,
                #  and that should be an integer
                #ASALA: length_a=0 or length_a-length_h+1??
                # TODO it should be updated to be length_a-length_h
                new_trans_h_a=transactions_a < pool
                #(a, h, La, Lh, Ta, Th, fork, r, uh, ua, pool)
                next_h_state3 = a + self.add_block(h, new_trans_h_a)  \
                                + (length_a-length_h, 1, self.chain_transactions(transactions_a,length_h), new_trans_h_a, self.Fork.Relevant, r, new_uh, ua,
                                   min(pool - accepted_transactions + 1, self.max_pool))
                #ASALA :TODO : is difficulty_contribution is true ?or must be difficulty_contribution=length_h??
                # TODO it should be length_h

                # TODO it should be length_a <= length_h not >=
                new_transaction_chance = self.transaction_chance if length_a <= length_h else 0
                transitions.add(next_h_state3, probability=(1 - self.alpha) * self.gamma * new_transaction_chance,
                                reward=accepted_transactions * self.fee + nephew_reward + length_a,
                                difficulty_contribution=length_h)


                next_h_state4 = a+ self.add_block(h, new_trans_h_a) \
                                + (length_a-length_h, 1, transactions_a, new_trans_h_a, self.Fork.Relevant, r,\
                                   new_uh, ua, pool - accepted_transactions)
                transitions.add(next_h_state4,
                                probability=(1 - self.alpha) * self.gamma * (1 - new_transaction_chance),
                                reward=accepted_transactions * self.fee + nephew_reward + length_a,
                                difficulty_contribution=length_a)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha * self.block_reward * (1 + self.fee * self.transaction_chance)


if __name__ == '__main__':
    print('ethereum_fee_mdp module test')
    # from blockchain_mdps.base.blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP
    # np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # mdp = BitcoinFeeModel(0.35, 0.5, 2, fee=2, transaction_chance=0.1, max_pool=2)
    # print(mdp.state_space.size)
