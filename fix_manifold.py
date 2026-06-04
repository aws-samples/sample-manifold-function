import open3d as o3d
import numpy as np
from pathlib import Path


def convert_to_manifold(input_glb_path, output_glb_path=None):
    """
    Convert a non-manifold GLB mesh to a manifold GLB mesh.
    
    Args:
        input_glb_path: Path to input GLB file
        output_glb_path: Path to output GLB file (optional, defaults to input_manifold.glb)
    
    Returns:
        Path to the output manifold GLB file
    """
    if output_glb_path is None:
        input_path = Path(input_glb_path)
        output_glb_path = input_path.parent / f"{input_path.stem}_manifold{input_path.suffix}"
    
    # Load the mesh
    print(f"Loading mesh from {input_glb_path}...")
    mesh = o3d.io.read_triangle_mesh(str(input_glb_path))
    
    if not mesh.has_vertices():
        raise ValueError("Mesh has no vertices")
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Check if mesh is manifold
    is_edge_manifold = mesh.is_edge_manifold()
    is_vertex_manifold = mesh.is_vertex_manifold()
    print(f"Edge manifold: {is_edge_manifold}, Vertex manifold: {is_vertex_manifold}")
    
    # Remove duplicated vertices
    mesh.remove_duplicated_vertices()
    
    # Remove duplicated triangles
    mesh.remove_duplicated_triangles()
    
    # Remove degenerate triangles
    mesh.remove_degenerate_triangles()
    
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    
    # Remove non-manifold edges
    mesh.remove_non_manifold_edges()
    
    print(f"After cleanup: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Check manifold status again
    is_edge_manifold = mesh.is_edge_manifold()
    is_vertex_manifold = mesh.is_vertex_manifold()
    print(f"After cleanup - Edge manifold: {is_edge_manifold}, Vertex manifold: {is_vertex_manifold}")
    
    # If still not manifold, try more aggressive fixes
    if not is_edge_manifold or not is_vertex_manifold:
        print("Applying additional fixes...")
        
        # Convert to legacy mesh for additional operations
        mesh_legacy = o3d.geometry.TriangleMesh(mesh)
        
        # Cluster connected triangles
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_legacy.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Keep only the largest cluster
        if len(cluster_n_triangles) > 0:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh_legacy.remove_triangles_by_mask(triangles_to_remove)
            mesh_legacy.remove_unreferenced_vertices()
            
            print(f"Kept largest cluster: {len(mesh_legacy.vertices)} vertices, {len(mesh_legacy.triangles)} triangles")
            mesh = mesh_legacy
    
    # Final check
    is_edge_manifold = mesh.is_edge_manifold()
    is_vertex_manifold = mesh.is_vertex_manifold()
    print(f"Final - Edge manifold: {is_edge_manifold}, Vertex manifold: {is_vertex_manifold}")
    
    if not is_edge_manifold or not is_vertex_manifold:
        print("Warning: Mesh is still not fully manifold. UV unwrapping may still fail.")
    
    # Save the manifold mesh
    print(f"Saving manifold mesh to {output_glb_path}...")
    o3d.io.write_triangle_mesh(str(output_glb_path), mesh)
    
    return str(output_glb_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_manifold.py <input.glb> [output.glb]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = convert_to_manifold(input_file, output_file)
        print(f"Success! Manifold mesh saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)