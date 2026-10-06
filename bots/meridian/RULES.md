# Meridian rules

Two players, Blue and Red, draw straight edges on a grid to enclose area.

1. **Board.** A 19 × 19 grid of points (x, y), with x and y from −9 to 9. Blue starts with the edge (−9,0)–(−6,0) and Red with (6,0)–(9,0). The ends of a player's edges are that player's nodes.
2. **Turns.** Blue makes one move, then Red and Blue take turns of two moves each. The game ends after the 120th move.
3. **Moves.** A move draws an edge from one of your nodes to a point at most 3 away in x and in y. An empty point becomes your node, your own node is joined, and an opponent's node is captured.
4. **Own edges.** A new edge may not overlap one of your edges, pass through one of your nodes, or put a new node on one of your edges. Your own edges may cross.
5. **Cuts.** A new edge may touch at most one opponent edge, and cuts it: that edge is removed, and so are its nodes that have no edges left, unless you captured them. Crossing an edge, running along it or ending on it touches it. Passing through or capturing a node touches every edge at that node.
6. **Protection.** You may not touch an edge placed in your opponent's last turn.
7. **Area.** Your area is everything your own edges enclose: every part of the board that cannot be reached from outside without crossing one of them. Opponent edges do not reduce it.
8. **Scoring.** At the end of every turn, and after the 120th move, both players add their area to their score. Nothing is rounded.
9. **Result.** The higher score wins, and equal scores draw. If the player to move has no legal move, the game ends at once in a draw.
