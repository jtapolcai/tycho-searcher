use super::graph_types::{NodeData, EdgeData};

use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use petgraph::Directed;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::cmp::min;
use std::cell::RefCell;

#[derive(Debug)]
pub struct GraphComponent {
    pub nodes: HashSet<NodeIndex>,
    pub edges: HashSet<EdgeIndex>,
}

#[derive(Debug, Clone)]
pub struct GraphComponents {
    pub graph_comps: Vec<Graph<NodeData, RefCell<EdgeData>, Directed>>,
    pub original_node_to_component_map: HashMap<NodeIndex, HashSet<(usize, NodeIndex)>>,
    pub original_edge_to_component_map: HashMap<EdgeIndex, (usize, EdgeIndex)>,   
}

impl Default for GraphComponents {
    fn default() -> Self {
        Self {
            graph_comps: Vec::new(),
            original_node_to_component_map: HashMap::new(),
            original_edge_to_component_map: HashMap::new(),
        }
    }
}

impl GraphComponents {
    pub fn build(
        original_graph: Graph<NodeData, RefCell<EdgeData>, Directed>,
        start_node_index: Option<NodeIndex>,
    ) -> Self {
        let mut result = GraphComponents::default();

        if original_graph.node_count() == 0 {
            return result;
        }

        let start = match start_node_index {
            Some(idx) if original_graph.node_weight(idx).is_some() => idx,
            _ => return result,
        };

        let mut dfs_num = HashMap::new();
        let mut low_link = HashMap::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();
        let mut edge_stack = vec![];
        let mut graph_components = vec![];
        let mut time = 0;

        dfs_traverse_from_node(
            &original_graph,
            start,
            None,
            &mut dfs_num,
            &mut low_link,
            &mut visited,
            &mut parent,
            &mut time,
            &mut edge_stack,
            &mut graph_components
        );

        // convert GraphComponent to actual Graph<NodeData, EdgeData>
        for (idx, comp) in graph_components.iter().enumerate() {
            let mut g = Graph::<NodeData, RefCell<EdgeData>, Directed>::new();
            let mut node_map = HashMap::new();

            for &node in &comp.nodes {
                let new_node = g.add_node(original_graph[node].clone());
                node_map.insert(node, new_node);
                result.original_node_to_component_map
                    .entry(node)
                    .or_insert_with(HashSet::new)
                    .insert((idx, new_node));
            }

            // --- MODIFIED: Add all edges between nodes in the component, not just those in comp.edges ---
            let comp_node_set: HashSet<_> = comp.nodes.iter().copied().collect();
            for u in &comp.nodes {
                for edge_ref in original_graph.edges(*u) {
                    let v = edge_ref.target();
                    let edge_idx = edge_ref.id();
                    if comp_node_set.contains(&v) {
                        if let (Some(&na), Some(&nb)) = (node_map.get(u), node_map.get(&v)) {
                            let new_edge = g.add_edge(na, nb, original_graph[edge_idx].clone());
                            result.original_edge_to_component_map.insert(edge_idx, (idx, new_edge));
                        }
                    }
                }
            }
            // --- END MODIFIED ---

            result.graph_comps.push(g);
        }

        result
    }

     /// Prints a detailed summary of the component graphs.
    pub fn print_summary(&self, original_graph: &Graph<NodeData, RefCell<EdgeData>, Directed>) {
        println!("GraphComponents Summary:");
        println!("Total components: {}", self.graph_comps.len());
        println!();

        for (i, comp) in self.graph_comps.iter().enumerate() {
            println!("Component #{}:", i);
            println!("  Nodes ({}):", comp.node_count());
            for node in comp.node_indices() {
                let original = self.original_node_to_component_map
                    .iter()
                    .find_map(|(orig_idx, set)| {
                        set.iter()
                            .find(|(comp_idx, new_idx)| *comp_idx == i && *new_idx == node)
                            .map(|_| *orig_idx)
                    });

                if let Some(orig_idx) = original {
                    let name = &original_graph[orig_idx].token.symbol;
                    println!("    - [{}] {}", node.index(), name);
                } else {
                    println!("    - [{}] (unknown origin)", node.index());
                }
            }

            println!("  Edges ({}):", comp.edge_count());
            for edge in comp.edge_references() {
                let orig = self.original_edge_to_component_map
                    .iter()
                    .find_map(|(orig_idx, &(comp_idx, local_idx))| {
                        if comp_idx == i && local_idx == edge.id() {
                            Some(orig_idx)
                        } else {
                            None
                        }
                    });

                if let Some(orig_edge_idx) = orig {
                    let (u, v) = original_graph.edge_endpoints(*orig_edge_idx).unwrap();
                    let u_name = &original_graph[u].token.symbol;
                    let v_name = &original_graph[v].token.symbol;
                    println!("    - [{}] {} -> {}", edge.id().index(), u_name, v_name);
                } else {
                    println!("    - [{}] (unknown origin)", edge.id().index());
                }
            }

            println!();
        }
    }
}

fn dfs_traverse_from_node(
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    u: NodeIndex,
    start_node: Option<NodeIndex>,
    dfs_num: &mut HashMap<NodeIndex, usize>,
    low_link: &mut HashMap<NodeIndex, usize>,
    visited: &mut HashSet<NodeIndex>,
    parent: &mut HashMap<NodeIndex, NodeIndex>,
    time: &mut usize,
    edge_stack: &mut Vec<(NodeIndex, NodeIndex, EdgeIndex)>,
    components: &mut Vec<GraphComponent>
) {
    visited.insert(u);
    *time += 1;
    dfs_num.insert(u, *time);
    low_link.insert(u, *time);

    for edge_ref in graph.edges(u) {
        let v = edge_ref.target();
        let edge_idx = edge_ref.id();

        if Some(v) == start_node {
            continue;
        }

        if visited.contains(&v) {
            if dfs_num[&v] < dfs_num[&u] {
                edge_stack.push((u, v, edge_idx));
            }
            *low_link.get_mut(&u).unwrap() = min(low_link[&u], dfs_num[&v]);
        } else {
            parent.insert(v, u);
            edge_stack.push((u, v, edge_idx));
            dfs_traverse_from_node(
                graph,
                v,
                Some(u),
                dfs_num,
                low_link,
                visited,
                parent,
                time,
                edge_stack,
                components,
            );

            *low_link.get_mut(&u).unwrap() = min(low_link[&u], low_link[&v]);

            if low_link[&v] >= dfs_num[&u] && start_node.is_none() {
                let mut component_nodes = HashSet::new();
                let mut component_edges = HashSet::new();
                loop {
                    if let Some((x, y, e)) = edge_stack.pop() {
                        component_nodes.insert(x);
                        component_nodes.insert(y);
                        component_edges.insert(e);
                        if (x == u && y == v) || (x == v && y == u) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if !component_edges.is_empty() {
                    components.push(GraphComponent {
                        nodes: component_nodes,
                        edges: component_edges,
                    });
                }
            }
        }
    }
}