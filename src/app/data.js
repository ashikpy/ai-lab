export const CodeData = {
  1: {
    question: `N-Queen`,

    code: `
def solve(board=None, row=0, n=8):
    if board is None:
        board = [-1] * n  # Initialize the board with -1 for empty positions
    if row == n: 
        print([x + 1 for x in board])
        return True  # Stop recursion after finding the first solution
    for col in range(n):
        if all(board[r] != col and abs(r - row) != abs(board[r] - col) for r in range(row)):
            board[row] = col
            if solve(board, row + 1, n): return True  # Found a solution, return True
solve(n=4)  # Example for n=4
`,

    description: ``,
  },

  2: {
    question: "Constrain Satisfaction Problem",
    code: `
!pip install python-constraint
from constraint import Problem
problem = Problem()
problem.addVariable('A', range(1, 6))
problem.addVariable('B', range(1, 6))
problem.addVariable('C', range(1, 11))
problem.addConstraint(lambda a, c: a * 2 == c, ('A', 'C'))
problem.addConstraint(lambda a, b, c: a + b == c, ('A', 'B', 'C'))
for solution in problem.getSolutions():
    print("A: {}, B: {}, C: {}".format(solution['A'], solution['B'], solution['C']))
`,

    description: ``,
  },
  3: {
    question: `AO*`,

    description: ``,
    code: `
class Node:
    def __init__(self, name, node_type, children=None):
        self.name = name
        self.node_type = node_type
        self.children = children if children else []
def ao_star(node):
    if node.node_type == "OR":
        for child in node.children:
            if ao_star(child) == "Goal Reached!":
                return "Goal Reached!"
    elif node.node_type == "AND":
        return "Goal Reached!"
    return "Goal not reached"
step1 = Node("Step 1", "AND")
step2 = Node("Step 2", "AND")
route_a = Node("Route A", "OR", [step1, step2])
goal = Node("Arrive at destination", "OR", [route_a])
print(ao_star(goal))
`,
  },

  4: {
    question: `8-Queen`,
    description: ``,

    code: `
def solve(board=[-1]*8, row=0):
    if row == 8: 
        print([x + 1 for x in board])
        return True  # Stop recursion after finding the first solution
    for col in range(8):
        if all(board[r] != col and abs(r - row) != abs(board[r] - col) for r in range(row)):
            board[row] = col
            if solve(board, row + 1): return True  # Found a solution, return True
solve()
`,
  },
  5: {
    question: `A* Greedy First`,
    description: ``,
    code: `
import heapq
def greedy_best_first_search(graph, start, goal, heuristic):
    pq, visited = [(heuristic[start], start, [start])], set()
    while pq:
        _, node, path = heapq.heappop(pq)
        if node == goal: return path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic[neighbor], neighbor, path + [neighbor]))
    return None
graph = {'A': [('B', 1), ('C', 3)], 'B': [('D', 4), ('E', 2)], 'C': [('F', 5)], 'D': [('G', 6)], 'E': [('G', 1)], 'F': [('G', 2)], 'G': []}
heuristic = {'A': 10, 'B': 6, 'C': 7, 'D': 3, 'E': 2, 'F': 6, 'G': 0}
start, goal = 'A', 'G'
path = greedy_best_first_search(graph, start, goal, heuristic)
print("Path found:", " -> ".join(path) if path else "No path found.")
`,
  },
  6: {
    question: `TSP`,
    description: ``,

    code: `
def nearest_neighbor_tsp(distances):
    n = len(distances)
    visited, route, total_distance = [False] * n, [0], 0
    visited[0] = True
    for _ in range(1, n):
        last = route[-1]
        nearest = min((i for i in range(n) if not visited[i]), key=lambda i: distances[last][i])
        visited[nearest], route = True, route + [nearest]
        total_distance += distances[last][nearest]
    return route + [0], total_distance + distances[route[-1]][0]
distances = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]
route_nn, total_distance_nn = nearest_neighbor_tsp(distances)
print("Route:", route_nn)
print("Total distance:", total_distance_nn)
`,
  },
  7: {
    question: `α β pruning`,
    description: ``,
    code: `
def minimax(depth, node_index, maximizing_player, values, alpha, beta):
    if depth == 3: return values[node_index]
    eval_func = max if maximizing_player else min
    best_value = float('-inf') if maximizing_player else float('inf')
    for i in range(2):
        eval = minimax(depth + 1, node_index * 2 + i, not maximizing_player, values, alpha, beta)
        best_value = eval_func(best_value, eval)
        alpha, beta = (max(alpha, eval), beta) if maximizing_player else (alpha, min(beta, eval))
        if beta <= alpha: break
    return best_value
values = [3, 5, 6, 9, 1, 2, 0, -1] 
print("The optimal value is:", minimax(0, 0, True, values, float('-inf'), float('inf')))
`,
  },
  8: {
    question: `Propositional Logic checking`,
    description: ``,
    code: `
def evaluate_formula(formula, model):
    formula = formula.replace('∧', ' and ').replace('∨', ' or ').replace('¬', ' not ').replace('→', '<=')
    for var, value in model.items():
        formula = formula.replace(var, str(value))  # Replace variables with their values
    try:
        return eval(formula)  # Evaluate the formula after all replacements
    except Exception:
        return False  # If there is an error in evaluation, return False
if __name__ == "__main__":
    formula = "(p ∨ q) → (r ∧ ¬p)"
    model = {'p': True, 'q': False, 'r': True}
    print(f"The formula {formula} is {'satisfied' if evaluate_formula(formula, model) else 'NOT satisfied'} by the model {model}")
`,
  },
};
