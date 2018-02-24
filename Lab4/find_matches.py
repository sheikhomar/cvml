def find_matches(descImg1, descImg2, threshold = 0.75):
    matches = []
    i1 = 0
    i2 = 0
    for k1 in descImg1:
        closest_match_distance = float('inf')
        second_closest_match_distance = float('inf')
        best_match = None
        for k2 in descImg2:
            distance_squared = np.linalg.norm(k1 - k2)
            if distance_squared < closest_match_distance:
                second_closest_match_distance = closest_match_distance
                closest_match_distance = distance_squared
                best_match = i2
            i2 += 1
        if (threshold * closest_match_distance < second_closest_match_distance and best_match is not None):
            matches.append((i1, best_match, closest_match_distance))
        i1 += 1
    return matches
            
