class Node:

    def __init__(self, team: str, losses, total_diff):
        self.team = team
        self.losses = float(losses)
        self.total_diff = float(total_diff)
        self.connections = []

    def __str__(self) -> str:
        connections = ', '.join(opp.team for opp, _ in self.connections)
        return self.team + ' connects to ' + connections
    
    def add_connection(self, destination: 'Node', diff):
        value = (0.85 / self.losses) * (float(diff) / self.total_diff)
    
        # Check if the destination already exists in the connections list
        for i, (existing_dest, existing_value) in enumerate(self.connections):
            if existing_dest == destination:
                # If the destination exists, update the value by adding the new one
                self.connections[i] = (existing_dest, existing_value + value)
                return
        
        # If no connection with the destination exists, add a new one
        self.connections.append((destination, value))

    def add_random(self, teams):
        for team in teams:
            value = 0.15/float(len(teams))
            self.connections.append((team, value))

