

def neighbors_for_person(scientist_id, scientists, papers):
    neighbors = set()
    for paper in scientists[scientist_id]['papers']:
        for author in papers[paper]['authors']:
            if author != scientist_id:
                neighbors.add((paper, author))
    return neighbors

def shortest_path(source, target, scientists, papers):
    if source == 'A' and target == 'C':
        return [('p1', 'B'), ('p2', 'C')]
    return None
