import random


def random_task():
    nodes = random.sample(range(1, 10), 4)
    return dict(zip(ABCD.GOAL_SEQUENCE, nodes))


class ABCD:
    GOAL_SEQUENCE = ['A', 'B', 'C', 'D']
    NODE_TO_RC = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2),
    }
    RC_TO_NODE = {v: k for k, v in NODE_TO_RC.items()}
    ACTION_DELTAS = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}

    def __init__(self, task: dict):
        # task maps goal label -> node number, e.g. {'A':3,'B':7,'C':1,'D':5}
        self.task = task
        self.current_goal_idx = 0
        self.agent_node = self._sample_start()

    def _sample_start(self):
        goal_nodes = set(self.task.values())
        options = [n for n in range(1, 10) if n not in goal_nodes]
        return random.choice(options)

    def scramble(self):
        nodes = random.sample(range(1, 10), 4)
        self.task = dict(zip(self.GOAL_SEQUENCE, nodes))
        self.current_goal_idx = 0
        self.agent_node = self._sample_start()

    def reset(self):
        self.current_goal_idx = 0
        self.agent_node = self._sample_start()

    def step(self, action: str) -> dict:
        r, c = self.NODE_TO_RC[self.agent_node]
        dr, dc = self.ACTION_DELTAS[action]
        nr = max(0, min(2, r + dr))
        nc = max(0, min(2, c + dc))
        self.agent_node = self.RC_TO_NODE[(nr, nc)]

        current_goal_label = self.GOAL_SEQUENCE[self.current_goal_idx]
        goal_node = self.task[current_goal_label]
        reward = 0
        if self.agent_node == goal_node:
            reward = 1
            self.current_goal_idx = (self.current_goal_idx + 1) % 4

        return {
            'node': self.agent_node,
            'rc': self.NODE_TO_RC[self.agent_node],
            'reward': reward,
            'current_goal': current_goal_label,
            'goal_node': goal_node,
            'goal_rc': self.NODE_TO_RC[goal_node],
        }

    def render_grid(self) -> str:
        rows = []
        for r in range(3):
            row = ['O' if self.RC_TO_NODE[(r, c)] == self.agent_node else 'X'
                   for c in range(3)]
            rows.append(' '.join(row))
        return '\n'.join(rows)
