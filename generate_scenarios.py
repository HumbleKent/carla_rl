import json
import numpy as np
import math
import os

def load_cones(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def cluster_cones(cones, threshold=15.0):
    """
    Cluster cones based on Euclidean distance.
    Returns a list of lists (clusters of cone dicts).
    """
    points = np.array([[c['x'], c['y'], c['z']] for c in cones])
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        
        # Start new cluster
        cluster_indices = [i]
        visited[i] = True
        queue = [i]
        
        while queue:
            curr_idx = queue.pop(0)
            curr_point = points[curr_idx]
            
            # Find neighbors
            # This is O(N^2) total but N is small (~1200), so it's fine (microseconds).
            dists = np.linalg.norm(points - curr_point, axis=1)
            neighbors = np.where((dists < threshold) & (~visited))[0]
            
            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(neighbor)
                cluster_indices.append(neighbor)
        
        clusters.append([cones[idx] for idx in cluster_indices])
    
    return clusters

def get_scenario_from_cluster(cluster, index):
    """
    Calculate start and target positions for a cluster of cones.
    Assumes the track is somewhat linear-ish or has two distinct ends.
    """
    points = np.array([[c['x'], c['y']] for c in cluster])
    z_height = np.mean([c['z'] for c in cluster]) + 0.3 # Spawn slightly above
    
    # Simple PCA-like approach: find the pair of points with max distance
    max_dist = 0
    p1, p2 = points[0], points[0]
    
    # Heuristic: Find centroid, then find farthest point A, then find farthest point B from A.
    # This is approx diameter.
    centroid = np.mean(points, axis=0)
    dists_from_centroid = np.linalg.norm(points - centroid, axis=1)
    idx_a = np.argmax(dists_from_centroid)
    point_a = points[idx_a]
    
    dists_from_a = np.linalg.norm(points - point_a, axis=1)
    idx_b = np.argmax(dists_from_a)
    point_b = points[idx_b]
    
    # Determine direction. 
    # Logic: "Town05-Cone-Avoidance" usually implies driving through them.
    # We define Start -> End.
    # We'll just define TWO scenarios per cluster: A -> B and B -> A to accept both directions.
    
    scenarios = {}
    
    # Vector A -> B
    vec = point_b - point_a
    length = np.linalg.norm(vec)
    if length < 1.0: return {} # Single cone or tiny cluster
    
    unit_vec = vec / length
    
    # Offset start and end by 5 meters
    offset = 8.0
    
    # Scenario 1: A -> B
    start_pos_1 = point_a - unit_vec * offset
    target_pos_1 = point_b + unit_vec * offset
    yaw_1 = math.degrees(math.atan2(unit_vec[1], unit_vec[0]))
    
    scenarios[f"ConeTrack-{index}-DirA"] = {
        "map_name": "Town05",
        "weather_condition": "Clear Noon",
        "initial_position": {"x": float(start_pos_1[0]), "y": float(start_pos_1[1]), "z": float(z_height)},
        "initial_rotation": {"pitch": 0.0, "yaw": float(yaw_1), "roll": 0.0},
        "target_position": {"x": float(target_pos_1[0]), "y": float(target_pos_1[1]), "z": float(z_height)},
        "target_gnss": {"lat": 0.0, "lon": 0.0, "alt": 0},
        "traffic_density": "None",
        "situation": "Road"
    }

    # Scenario 2: B -> A
    start_pos_2 = point_b + unit_vec * offset
    target_pos_2 = point_a - unit_vec * offset
    # Vector is -unit_vec
    yaw_2 = math.degrees(math.atan2(-unit_vec[1], -unit_vec[0]))
    
    scenarios[f"ConeTrack-{index}-DirB"] = {
        "map_name": "Town05",
        "weather_condition": "Clear Noon",
        "initial_position": {"x": float(start_pos_2[0]), "y": float(start_pos_2[1]), "z": float(z_height)},
        "initial_rotation": {"pitch": 0.0, "yaw": float(yaw_2), "roll": 0.0},
        "target_position": {"x": float(target_pos_2[0]), "y": float(target_pos_2[1]), "z": float(z_height)},
        "target_gnss": {"lat": 0.0, "lon": 0.0, "alt": 0},
        "traffic_density": "None",
        "situation": "Road"
    }
    
    return scenarios

def main():
    cone_file = 'env/cone_layout.json'
    output_file = 'env/vehicle_spawn.json'
    
    print(f"Loading cones from {cone_file}...")
    cones = load_cones(cone_file)
    print(f"Loaded {len(cones)} cones.")
    
    clusters = cluster_cones(cones)
    print(f"Found {len(clusters)} clusters.")
    
    all_scenarios = {}
    
    for i, cluster in enumerate(clusters):
        if len(cluster) < 5:
            print(f"Skipping cluster {i} (size {len(cluster)} < 5)")
            continue
            
        print(f"Processing cluster {i} (size {len(cluster)})...")
        scenarios = get_scenario_from_cluster(cluster, i)
        all_scenarios.update(scenarios)
        
    print(f"Generated {len(all_scenarios)} scenarios.")
    
    with open(output_file, 'w') as f:
        json.dump(all_scenarios, f, indent=4)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
