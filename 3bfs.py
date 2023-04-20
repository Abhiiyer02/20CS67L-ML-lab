import heapq

def best_first_search(graph, start, goal):
    # Initialize the priority queue with the start node
    frontier = [(0, start)]
    # Initialize the explored set
    explored = set()
   
    # Loop until the frontier is empty
    while frontier:
        # Pop the node with the highest priority
        (cost, current_node) = heapq.heappop(frontier)
       
        # Check if the current node is the goal
        if current_node == goal:
            return cost
       
        # Add the current node to the explored set
        explored.add(current_node)
        print(f"Explored node: {current_node}")
       
        # Explore the neighbors of the current node
        for neighbor, neighbor_cost in graph[current_node]:
            # Check if the neighbor is not in the explored set and not in the frontier
            if neighbor not in explored and neighbor not in [node[1] for node in frontier]:
                # Add the neighbor to the frontier with its priority being its heuristic cost
                heapq.heappush(frontier, (neighbor_cost+cost, neighbor))
                print(f"Added node {neighbor} to frontier with cost {neighbor_cost}")
   
    # If the goal cannot be reached, return None
    return None

# Example graph
graph = {
    'A': [('B', 5), ('C', 6)],
    'B': [('D', 4), ('E', 7)],
    'C': [('F', 9), ('G', 8)],
    'D': [('H', 3)],
    'E': [('I', 6)],
    'F': [('J', 5)],
    'G': [('K', 7)],
    'H': [('L', 1)],
    'I': [('M', 2)],
    'J': [('N', 3)],
    'K': [('O', 4)],
    'L': [],
    'M': [],
    'N': [],
    'O': [('P', 1)],
    'P': []
}

# Get start and goal nodes from the user
start = input("Enter the start node: ")
goal = input("Enter the goal node: ")

# Run the Best First Search algorithm
result = best_first_search(graph, start, goal)

# Print the result
if result is not None:
    print(f"The minimum cost from {start} to {goal} is {result}.")
else:
    print(f"There is no path from {start} to {goal}.")
