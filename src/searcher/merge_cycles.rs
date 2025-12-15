// merge_cycles.rs
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use num_bigint::ToBigUint;
use petgraph::{
    graph::{EdgeIndex, Graph, NodeIndex},
    Directed,
};
use std::cell::RefCell;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex as PetgraphNodeIndex};
use petgraph::visit::EdgeRef;

use crate::searcher::{EdgeData, NodeData, Statistics};
use crate::searcher::bellman_ford::describe_path;
use crate::log_arb_info;

/// Check if two cycles can be merged
/// They can be merged if all their common edges are in both shareable edge lists
fn can_merge_cycles(
    cycle1: &(f64, f64, f64, Rc<Vec<EdgeIndex>>, Vec<EdgeIndex>),
    cycle2: &(f64, f64, f64, Rc<Vec<EdgeIndex>>, Vec<EdgeIndex>),
) -> bool {
    let (_, _, _, path1, shareable1) = cycle1;
    let (_, _, _, path2, shareable2) = cycle2;
    
    // Find common edges between the two cycles
    let edges1_set: HashSet<_> = path1.iter().copied().collect();
    let edges2_set: HashSet<_> = path2.iter().copied().collect();
    let common_edges: HashSet<_> = edges1_set.intersection(&edges2_set).copied().collect();
    
    // If no common edges, they can be merged (they don't interfere)
    if common_edges.is_empty() {
        return true;
    }
    
    // All common edges must be in both shareable lists
    let shareable1_set: HashSet<_> = shareable1.iter().copied().collect();
    let shareable2_set: HashSet<_> = shareable2.iter().copied().collect();
    
    common_edges.iter().all(|edge| {
        shareable1_set.contains(edge) && shareable2_set.contains(edge)
    })
}

/// Represents a merged cycle result with optimized input amounts
#[derive(Debug, Clone)]
pub struct MergedCycleResult {
    pub cycles: Vec<Rc<Vec<EdgeIndex>>>,
    pub amounts_in: Vec<f64>,
    pub amounts_out: Vec<f64>,
    pub net_profits: Vec<f64>,
    pub total_gas: f64,
    pub combined_net_profit: f64,
    /// List of (EdgeIndex, optional_ratio) pairs in topological order
    /// ratio is None when edge is not split, Some(r) where r in [0, 1] when split
    /// When split, one branch can have ratio=None meaning "remainder"
    pub edge_splits: Vec<(EdgeIndex, Option<f64>)>,
}


/// We treat splitting as multiple cycles. 
/// This function evaluate multiple cycles together as a DAG, sharing gas costs on overlapping edges
fn evaluate_merged_cycles(
    cycles: &[Rc<Vec<EdgeIndex>>],
    amounts_in: &[f64],
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    gas_price: f64,
    quoter_time: &mut Duration,
) -> Option<MergedCycleResult> {
    let log_details = true;

    if cycles.is_empty() || cycles.len() != amounts_in.len() {
        return None;
    }

    let n = cycles.len();
    let from_node_data = graph.node_weight(source)?;
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);

    // Log cycle paths for visibility - only once per subset
    // Use a thread-local static to track if we've logged this subset
    thread_local! {
        static LOGGED_CYCLES: std::cell::RefCell<std::collections::HashSet<String>> = std::cell::RefCell::new(std::collections::HashSet::new());
    }
    
    let cycles_key = format!("{:?}", cycles.iter().map(|c| c.as_ptr()).collect::<Vec<_>>());
    let should_log_cycles = LOGGED_CYCLES.with(|logged| {
        let mut set = logged.borrow_mut();
        set.insert(cycles_key)
    });
    
    if should_log_cycles {
        for (idx, cycle_rc) in cycles.iter().enumerate() {
            let path_str = describe_path(graph, cycle_rc.as_ref());
            log_arb_info!("[MERGE] Cycle {} path: {}", idx, path_str);
        }
    }

    // Build a DAG from the cycles
    // Nodes: represent actual tokens (NodeIndex), shared across cycles
    // We track amounts per (cycle_id, token) pair separately
    let mut dag: DiGraph<NodeIndex, EdgeIndex> = DiGraph::new();
    
    // Map from token NodeIndex to DAG node
    let mut token_to_dag_node: HashMap<NodeIndex, PetgraphNodeIndex> = HashMap::new();
    
    // Create distinct source and sink DAG nodes (sink represents closing at source)
    let _source_dag = *token_to_dag_node.entry(source).or_insert_with(|| dag.add_node(source));
    let sink_dag = dag.add_node(source); // separate DAG node, same underlying token

    // Build DAG edges from cycle edges and track edge usage
    let mut edge_to_cycles: HashMap<EdgeIndex, Vec<(usize, usize)>> = HashMap::new(); // edge -> [(cycle_id, step)]

    for (cycle_id, cycle) in cycles.iter().enumerate() {
        let mut prev_token = source;

        for (step, &edge_idx) in cycle.iter().enumerate() {
            edge_to_cycles.entry(edge_idx).or_insert_with(Vec::new).push((cycle_id, step));

            let (_from, to) = graph.edge_endpoints(edge_idx)?;
            
            // Get or create DAG nodes for tokens
            let prev_dag = *token_to_dag_node.entry(prev_token).or_insert_with(|| dag.add_node(prev_token));
            // if edge closes the cycle back to source, route to sink_dag instead of shared source node
            let next_dag = if to == source { sink_dag } else { *token_to_dag_node.entry(to).or_insert_with(|| dag.add_node(to)) };

            // Add edge if not already present
            if !dag.edges(prev_dag).any(|e| *e.weight() == edge_idx) {
                dag.add_edge(prev_dag, next_dag, edge_idx);
            }

            prev_token = to;
        }
    }
    
    // Log DAG structure before toposort
    if log_details {
        log_arb_info!("[MERGE] DAG structure:");
        for node_idx in dag.node_indices() {
            let token_idx = dag.node_weight(node_idx)?;
            let token_node = graph.node_weight(*token_idx)?;
            let out_count = dag.edges(node_idx).count();
            log_arb_info!("[MERGE]   Node {:?} = token {}, {} outgoing edges", 
                node_idx, token_node.token.symbol, out_count);
        }
    } 
    
    // Topological sort
    let topo_order = toposort(&dag, None).ok()?;
    
    // Log topo order with token symbols
    let topo_symbols: Vec<String> = topo_order.iter().map(|node_idx| {
        if let Some(token_idx) = dag.node_weight(*node_idx) {
            if let Some(token_node) = graph.node_weight(*token_idx) {
                format!("({})", token_node.token.symbol)
            } else {
                format!("({:?})", node_idx)
            }
        } else {
            format!("({:?})", node_idx)
        }
    }).collect();
    log_arb_info!("[MERGE] DAG has {} nodes in topo order: [{}]", topo_order.len(), topo_symbols.join(", "));
    // Print DAG edges after topo order for full visibility
    if log_details {
        log_arb_info!("[MERGE] DAG edges:");
        for edge in dag.edge_indices() {
            if let Some((from_dag, to_dag)) = dag.edge_endpoints(edge) {
                let from_token_idx = dag.node_weight(from_dag).copied();
                let to_token_idx = dag.node_weight(to_dag).copied();
                let (from_sym, to_sym) = match (from_token_idx, to_token_idx) {
                    (Some(fi), Some(ti)) => {
                        let fs = graph.node_weight(fi).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
                        let ts = graph.node_weight(ti).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
                        (fs, ts)
                    },
                    _ => ("?".to_string(), "?".to_string()),
                };
                let eidx = dag.edge_weight(edge).copied();
                if let Some(ei) = eidx {
                    let (gf, gt) = graph.edge_endpoints(ei).unwrap_or((source, source));
                    let gf_sym = graph.node_weight(gf).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
                    let gt_sym = graph.node_weight(gt).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
                    log_arb_info!("[MERGE]   DAG edge {:?}: dag {} -> {}, graph {} -> {}", edge, from_sym, to_sym, gf_sym, gt_sym);
                } else {
                    log_arb_info!("[MERGE]   DAG edge {:?}: dag {} -> {} (no graph edge index)", edge, from_sym, to_sym);
                }
            }
        }
    }
    
    if log_details {
        log_arb_info!("[MERGE] Edge usage map:");
        for (edge_idx, usages) in edge_to_cycles.iter() {
            let (from, to) = graph.edge_endpoints(*edge_idx).unwrap_or((source, source));
            let from_token = graph.node_weight(from).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
            let to_token = graph.node_weight(to).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
            log_arb_info!("[MERGE]   Edge {:?} ({} → {}) used by: {:?}", edge_idx, from_token, to_token, usages);
        }
    }

    // Initialize per-cycle amounts at shared source node
    // amounts_at[(cycle_id, token_node_index)] = amount of that token for that cycle
    let mut amounts_at: std::collections::HashMap<(usize, NodeIndex), BigUint> = std::collections::HashMap::new();
    for (cycle_id, &amt) in amounts_in.iter().enumerate() {
        let scaled = ((amt * multiplier).to_biguint()).unwrap_or(BigUint::from(0u32));
        amounts_at.insert((cycle_id, source), scaled);
    }
    
    log_arb_info!("[MERGE] Starting DAG evaluation with {} cycles", n);
    for (i, amt) in amounts_in.iter().enumerate() {
        log_arb_info!("[MERGE] Cycle {}: input = {:.6}", i, amt);
    }
    
    let mut total_gas = BigUint::from(0u32);
    let mut gas_paid_edges: HashSet<EdgeIndex> = HashSet::new();
    let mut edge_splits: Vec<(EdgeIndex, Option<f64>)> = Vec::new();  // Track edge splits in topo order
    
    // Process edges in topological order
    for node_idx in topo_order {
        // Get outgoing edges from this node
        let node_token = dag.node_weight(node_idx)?;
        let outgoing: Vec<_> = dag.edges(node_idx).map(|e| (e.id(), *e.weight())).collect();
        
        let token_node = graph.node_weight(*node_token)?;
        log_arb_info!("[MERGE] Processing DAG node (token {}) with {} outgoing edges", token_node.token.symbol, outgoing.len());
        
        for (dag_edge_idx, edge_idx) in outgoing {
            let (_from, target_dag) = dag.edge_endpoints(dag_edge_idx)?;

            let target_token = dag.node_weight(target_dag)?;
            let (from, to) = graph.edge_endpoints(edge_idx)?;
            let token_in = &graph.node_weight(from)?.token;
            let token_out = &graph.node_weight(to)?.token;
            let edge = graph.edge_weight(edge_idx)?;
            
            // Use token_in decimals for input amounts, token_out decimals for output amounts
            let multiplier_in = 10f64.powi(token_in.decimals as i32);
            let multiplier_out = 10f64.powi(token_out.decimals as i32);

            // Find all cycles sharing this edge
            static EMPTY_VEC: Vec<(usize, usize)> = Vec::new();
            let sharing_cycles = edge_to_cycles.get(&edge_idx).unwrap_or(&EMPTY_VEC);

            if sharing_cycles.len() > 1 {
                // Shared edge: sum amounts from all cycles using this edge
                let mut total_amount_in = BigUint::from(0u32);
                let mut cycle_shares = Vec::new();

                log_arb_info!("[MERGE] SHARED EDGE {} → {} (used by {} cycles)",
                    token_in.symbol, token_out.symbol, sharing_cycles.len());

                for &(other_cycle_id, step) in sharing_cycles {
                    let amt = amounts_at.get(&(other_cycle_id, *node_token)).cloned().unwrap_or_else(|| BigUint::from(0u32));
                    if amt == BigUint::from(0u32) {
                        continue;
                    }

                    total_amount_in += &amt;
                    cycle_shares.push((other_cycle_id, step));
                    let amt_f64 = amt.to_f64().unwrap_or(0.0) / multiplier_in;
                    log_arb_info!("[MERGE]   Cycle {} contributes: {:.6}", other_cycle_id, amt_f64);
                }

                if total_amount_in == BigUint::from(0u32) {
                    log_arb_info!("[MERGE]   Total summed amount is zero, skipping edge");
                    continue;
                }

                let total_in_f64 = total_amount_in.to_f64()?;
                log_arb_info!("[MERGE]   Total summed amount: {:.6}", total_in_f64 / multiplier_in);

                // Query quoter once with summed amount
                let t0 = std::time::Instant::now();
                let result = match edge.borrow().quoter_amount_out(token_in, token_out, &total_amount_in, stats) {
                    Some(r) => r,
                    None => {
                        log_arb_info!("[MERGE]   Quoter returned None, aborting evaluation");
                        return None;
                    }
                };
                *quoter_time += t0.elapsed();

                let result_out_f64 = result.amount_out.to_f64()?;
                let result_gas_f64 = result.gas.to_f64()?;
                log_arb_info!("[MERGE]   Quoter result: amount_out={:.6}, gas={:.0}",
                    result_out_f64 / multiplier_out, result_gas_f64);

                // Pay gas only once
                if !gas_paid_edges.contains(&edge_idx) {
                    total_gas += &result.gas;
                    gas_paid_edges.insert(edge_idx);
                    log_arb_info!("[MERGE]   Gas paid (first time): {:.0}", result_gas_f64);
                } else {
                    log_arb_info!("[MERGE]   Gas already paid for this edge");
                }

                // Distribute amount_out proportionally
                let total_out_f64 = result.amount_out.to_f64()?;

                for (idx, (other_cycle_id, _step)) in cycle_shares.iter().enumerate() {
                    let amt_in = amounts_at.get(&(*other_cycle_id, *node_token)).cloned().unwrap_or_else(|| BigUint::from(0u32));
                    let share = amt_in.to_f64()? / total_in_f64;
                    let proportional_out = (total_out_f64 * share).to_biguint()?;
                    let prop_out_f64 = proportional_out.to_f64()? / multiplier_out;
                    log_arb_info!("[MERGE]   Cycle {} gets {:.6} ({:.2}% share)",
                        other_cycle_id, prop_out_f64, share * 100.0);
                    amounts_at.insert((*other_cycle_id, *target_token), proportional_out);
                    
                    // Record split: last cycle gets None (remainder), others get their share
                    if idx < cycle_shares.len() - 1 {
                        edge_splits.push((edge_idx, Some(share)));
                    } else {
                        edge_splits.push((edge_idx, None));  // Last gets remainder
                    }
                }
            } else {
                // Exclusive edge: one cycle uses this edge alone
                let (only_cycle_id, _step) = sharing_cycles.get(0).cloned().unwrap_or((0, 0));
                let amt_in = amounts_at.get(&(only_cycle_id, *node_token)).cloned().unwrap_or_else(|| BigUint::from(0u32));

                if amt_in == BigUint::from(0u32) {
                    log_arb_info!("[MERGE] EXCLUSIVE EDGE {} → {} (Cycle {}) has zero input, skipping",
                        token_in.symbol, token_out.symbol, only_cycle_id);
                    continue;
                }

                let current_amt_f64 = amt_in.to_f64()?;
                log_arb_info!("[MERGE] EXCLUSIVE EDGE {} → {} (Cycle {})",
                    token_in.symbol, token_out.symbol, only_cycle_id);
                log_arb_info!("[MERGE]   Input amount: {:.6}", current_amt_f64 / multiplier_in);

                let t0 = std::time::Instant::now();
                let result = match edge.borrow().quoter_amount_out(token_in, token_out, &amt_in, stats) {
                    Some(r) => r,
                    None => {
                        log_arb_info!("[MERGE]   Quoter returned None, aborting evaluation");
                        return None;
                    }
                };
                *quoter_time += t0.elapsed();

                let result_out_f64 = result.amount_out.to_f64()?;
                let result_gas_f64 = result.gas.to_f64()?;
                log_arb_info!("[MERGE]   Quoter result: amount_out={:.6}, gas={:.0}",
                    result_out_f64 / multiplier_out, result_gas_f64);

                amounts_at.insert((only_cycle_id, *target_token), result.amount_out);
                total_gas += &result.gas;
                
                // No split for exclusive edge
                edge_splits.push((edge_idx, None));
            }
        }
    }
    
    // Calculate final results
    let mut amounts_out = vec![0.0; n];
    for (i, _cycle) in cycles.iter().enumerate() {
        if let Some(amount) = amounts_at.get(&(i, source)) {
            amounts_out[i] = amount.to_f64()? / multiplier;
        }
    }
    
    let total_gas_f64 = total_gas.to_f64()?;
    let gas_cost = total_gas_f64 * gas_price * 1e-9;
    
    log_arb_info!("[MERGE] === Final Results ===");
    log_arb_info!("[MERGE] Total gas units: {:.0}, gas cost: {:.9} ETH", total_gas_f64, gas_cost);
    
    // Calculate net profit for each cycle
    let mut net_profits = Vec::new();
    let mut combined_profit = 0.0;
    
    for i in 0..n {
        let profit = amounts_out[i] - amounts_in[i];
        net_profits.push(profit);
        combined_profit += profit;
        log_arb_info!("[MERGE] Cycle {}: in={:.6}, out={:.6}, profit={:.6}", 
            i, amounts_in[i], amounts_out[i], profit);
    }
    
    combined_profit -= gas_cost;
    log_arb_info!("[MERGE] Combined profit (after gas): {:.6} ETH", combined_profit);
    
    // Log edge splits with token symbols
    log_arb_info!("[MERGE] Edge splits (in topological order):");
    for (edge_idx, ratio_opt) in &edge_splits {
        let (from, to) = graph.edge_endpoints(*edge_idx).unwrap_or((source, source));
        let from_token = graph.node_weight(from).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
        let to_token = graph.node_weight(to).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "?".to_string());
        match ratio_opt {
            Some(ratio) => log_arb_info!("[MERGE]   Edge {:?} ({} → {}): ratio = {:.4}", edge_idx, from_token, to_token, ratio),
            None => log_arb_info!("[MERGE]   Edge {:?} ({} → {}): ratio = None (remainder/exclusive)", edge_idx, from_token, to_token),
        }
    }
    
    Some(MergedCycleResult {
        cycles: cycles.to_vec(),
        amounts_in: amounts_in.to_vec(),
        amounts_out,
        net_profits,
        total_gas: total_gas_f64,
        combined_net_profit: combined_profit,
        edge_splits,
    })
}

/// Group cycles into mergeable sets using greedy clustering
/// Returns a Vec of groups where all cycles in each group can merge with all others
fn group_mergeable_cycles(
    cycles: &[(f64, f64, f64, Rc<Vec<EdgeIndex>>, Vec<EdgeIndex>)],
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
) -> Vec<Vec<usize>> {
    let n = cycles.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; n];
    
    // Greedy clustering: for each unassigned cycle, try to build a group
    for i in 0..n {
        if assigned[i] {
            continue;
        }
        
        // Start a new group with cycle i
        let mut group = vec![i];
        assigned[i] = true;
        
        log_arb_info!("[MERGE-GROUP] Starting group with cycle {}", i);
        
        // Try to add other unassigned cycles to this group
        for j in (i + 1)..n {
            if assigned[j] {
                continue;
            }
            
            // Check if cycle j can merge with ALL cycles in the current group
            let can_merge_with_all = group.iter().all(|&k| {
                can_merge_cycles(&cycles[j], &cycles[k])
            });
            
            if can_merge_with_all {
                log_arb_info!("[MERGE-GROUP]   Cycle {} CAN merge with all in group", j);
                group.push(j);
                assigned[j] = true;
            } else {
                // Log why it cannot merge
                log_arb_info!("[MERGE-GROUP]   Cycle {} CANNOT merge with group {}", j, i);
                
                let (_, _, _, path_j, shareable_j) = &cycles[j];
                for &k in group.iter() {
                    let (_, _, _, path_k, shareable_k) = &cycles[k];
                    
                    // Find common edges
                    let edges_j_set: HashSet<_> = path_j.iter().copied().collect();
                    let edges_k_set: HashSet<_> = path_k.iter().copied().collect();
                    let common_edges: HashSet<_> = edges_j_set.intersection(&edges_k_set).copied().collect();
                    
                    if common_edges.is_empty() {
                        log_arb_info!("[MERGE-GROUP]     (with cycle {}): no common edges", k);
                    } else {
                        let shareable_j_set: HashSet<_> = shareable_j.iter().copied().collect();
                        let shareable_k_set: HashSet<_> = shareable_k.iter().copied().collect();
                        
                        // Find non-shareable common edges
                        let non_shareable_j: Vec<_> = common_edges.iter()
                            .filter(|e| !shareable_j_set.contains(e))
                            .copied()
                            .collect();
                        let non_shareable_k: Vec<_> = common_edges.iter()
                            .filter(|e| !shareable_k_set.contains(e))
                            .copied()
                            .collect();
                        
                        if !non_shareable_j.is_empty() {
                            let edge_info: Vec<String> = non_shareable_j.iter().map(|edge_idx| {
                                match graph.edge_endpoints(*edge_idx) {
                                    Some((from, to)) => {
                                        let from_sym = graph.node_weight(from)
                                            .map(|n| n.token.symbol.clone())
                                            .unwrap_or_else(|| "?".to_string());
                                        let to_sym = graph.node_weight(to)
                                            .map(|n| n.token.symbol.clone())
                                            .unwrap_or_else(|| "?".to_string());
                                        format!("{:?}({} -> {})", edge_idx, from_sym, to_sym)
                                    },
                                    None => format!("{:?}(?)", edge_idx),
                                }
                            }).collect();
                            log_arb_info!("[MERGE-GROUP]     (with cycle {}): cycle {} has {} non-shareable edges: {}", 
                                k, j, non_shareable_j.len(), edge_info.join(", "));
                        }
                        if !non_shareable_k.is_empty() {
                            let edge_info: Vec<String> = non_shareable_k.iter().map(|edge_idx| {
                                match graph.edge_endpoints(*edge_idx) {
                                    Some((from, to)) => {
                                        let from_sym = graph.node_weight(from)
                                            .map(|n| n.token.symbol.clone())
                                            .unwrap_or_else(|| "?".to_string());
                                        let to_sym = graph.node_weight(to)
                                            .map(|n| n.token.symbol.clone())
                                            .unwrap_or_else(|| "?".to_string());
                                        format!("{:?}({} -> {})", edge_idx, from_sym, to_sym)
                                    },
                                    None => format!("{:?}(?)", edge_idx),
                                }
                            }).collect();
                            log_arb_info!("[MERGE-GROUP]     (with cycle {}): cycle {} has {} non-shareable edges: {}", 
                                k, k, non_shareable_k.len(), edge_info.join(", "));
                        }
                        
                        if non_shareable_j.is_empty() && non_shareable_k.is_empty() {
                            // This shouldn't happen if can_merge_cycles returned false
                            log_arb_info!("[MERGE-GROUP]     (with cycle {}): unexpected: all common edges are shareable!", k);
                        }
                    }
                }
            }
        }
        
        log_arb_info!("[MERGE-GROUP] Group completed with {} cycles", group.len());
        // If the group contains entirely disjoint paths (no common edges among any pair),
        // then grouping is unnecessary. Split into singleton groups.
        if group.len() > 1 {
            let mut any_shared_edge = false;
            for a_idx in 0..group.len() {
                for b_idx in (a_idx + 1)..group.len() {
                    let a = group[a_idx];
                    let b = group[b_idx];
                    let (_, _, _, path_a, _) = &cycles[a];
                    let (_, _, _, path_b, _) = &cycles[b];
                    let set_a: HashSet<_> = path_a.iter().copied().collect();
                    let set_b: HashSet<_> = path_b.iter().copied().collect();
                    let common: HashSet<_> = set_a.intersection(&set_b).copied().collect();
                    if !common.is_empty() {
                        any_shared_edge = true;
                        break;
                    }
                }
                if any_shared_edge { break; }
            }

            if !any_shared_edge {
                log_arb_info!("[MERGE-GROUP] Group {} has fully disjoint paths; splitting into singletons", i);
                for &idx in &group {
                    groups.push(vec![idx]);
                }
            } else {
                groups.push(group);
            }
        } else {
            groups.push(group);
        }
    }
    
    log_arb_info!("[MERGE] Grouped {} cycles into {} mergeable clusters", n, groups.len());
    for (idx, group) in groups.iter().enumerate() {
        log_arb_info!("[MERGE]   Group {}: cycles {:?}", idx, group);
    }
    
    groups
}

/// N-dimensional optimization for multiple cycles
/// Selects k-sized subsets of cycles and optimizes each with GSS
pub fn optimize_merged_cycles(
    cycles: Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>, Vec<EdgeIndex>)>,
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in_min: f64,
    tolerance: f64,
    max_iter: usize,
    gas_price: f64,
    quoter_time: &mut Duration,
    _max_split: usize,
) -> Option<MergedCycleResult> {
    let n = cycles.len();
    if n == 0 {
        return None;
    }


    // First, group cycles into mergeable clusters
    let groups = group_mergeable_cycles(&cycles, graph);
    
    let mut best_result: Option<MergedCycleResult> = None;
    let mut best_profit = f64::NEG_INFINITY;
    
    // Optimize each mergeable group
    for (group_idx, group_indices) in groups.iter().enumerate() {
        // Skip single-cycle groups (no merging benefit)
        if group_indices.len() < 2 {
            log_arb_info!("[MERGE] Skipping group {} (only 1 cycle)", group_idx);
            continue;
        }
        
        log_arb_info!("[MERGE] Optimizing group {}: cycles {:?} (total {})", 
            group_idx, group_indices, group_indices.len());
        
        // Extract the cycles for this group
        let subset_cycles: Vec<_> = group_indices.iter()
            .map(|&i| {
                let (a, b, c, d, _e) = cycles[i].clone();
                (a, b, c, d)
            })
            .collect();
        
        // Run GSS on this group
        if let Some(result) = gss_optimize_subset(
            &subset_cycles,
            graph,
            stats,
            source,
            amount_in_min,
            tolerance,
            max_iter,
            gas_price,
            quoter_time,
        ) {
            if result.combined_net_profit > best_profit {
                best_profit = result.combined_net_profit;
                log_arb_info!("[MERGE] New best: group {} with profit {:.6}", 
                    group_idx, result.combined_net_profit);
                best_result = Some(result);
            }
        }
    }
    
    if let Some(ref result) = best_result {
        log_arb_info!(
            "Optimized cycles: best combined profit = {:.6} ETH",
            result.combined_net_profit
        );
    }
    
    best_result
}

/// GSS optimization for a subset of cycles
fn gss_optimize_subset(
    cycle_paths: &[(f64, f64, f64, Rc<Vec<EdgeIndex>>)],
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in_min: f64,
    tolerance: f64,
    max_iter: usize,
    gas_price: f64,
    quoter_time: &mut Duration,
) -> Option<MergedCycleResult> {
    let n = cycle_paths.len();
    if n == 0 {
        return None;
    }

    log_arb_info!("[MERGE-GSS] Starting N-dim GSS for {} cycles", n);
    
    // Extract individual optimized amounts and cycle paths
    let initial_amounts: Vec<f64> = cycle_paths.iter().map(|(amt, _, _, _)| *amt).collect();
    let paths: Vec<Rc<Vec<EdgeIndex>>> = cycle_paths.iter().map(|(_, _, _, p)| p.clone()).collect();
    
    for (i, &amt) in initial_amounts.iter().enumerate() {
        log_arb_info!("[MERGE-GSS] Cycle {} individual optimum: amount={:.6}", i, amt);
    }
    
    // Initialize bounds for each dimension
    let mut lower_bounds = vec![amount_in_min; n];
    let mut upper_bounds = initial_amounts.clone();  // Start from individual optima
    
    let golden_ratio = (5.0f64.sqrt() - 1.0) / 2.0;
    
    let mut best_result: Option<MergedCycleResult> = None;
    let mut best_profit = f64::NEG_INFINITY;
    
    // Evaluate with initial individual optima as starting point
    log_arb_info!("[MERGE-GSS] Evaluating initial combined point");
    if let Some(result) = evaluate_merged_cycles(
        &paths,
        &initial_amounts,
        graph,
        stats,
        source,
        gas_price,
        quoter_time,
    ) {
        best_profit = result.combined_net_profit;
        best_result = Some(result);
        log_arb_info!("[MERGE-GSS] Initial combined profit: {:.6}", best_profit);
    }
    
    for iteration in 0..max_iter {
        // For N dimensions, we sample at the golden sections
        // Simplified: for 2D we sample 4 points, for 3D we sample 8 points, etc.
        let mut sample_points = Vec::new();
        
        // Generate sample points
        fn generate_samples(
            bounds: &[(f64, f64)],
            golden: f64,
            samples: &mut Vec<Vec<f64>>,
        ) {
            if bounds.is_empty() {
                return;
            }
            
            if bounds.len() == 1 {
                let (a, b) = bounds[0];
                samples.push(vec![a + (1.0 - golden) * (b - a)]);
                samples.push(vec![a + golden * (b - a)]);
            } else {
                let mut sub_bounds = Vec::new();
                for i in 0..bounds.len() - 1 {
                    sub_bounds.push(bounds[i]);
                }
                
                let mut sub_samples = Vec::new();
                generate_samples(&sub_bounds, golden, &mut sub_samples);
                
                let (a, b) = bounds[bounds.len() - 1];
                let y1 = a + (1.0 - golden) * (b - a);
                let y2 = a + golden * (b - a);
                
                for sample in sub_samples {
                    let mut s1 = sample.clone();
                    s1.push(y1);
                    samples.push(s1);
                    
                    let mut s2 = sample.clone();
                    s2.push(y2);
                    samples.push(s2);
                }
            }
        }
        
        let bounds: Vec<_> = lower_bounds.iter().zip(upper_bounds.iter())
            .map(|(&a, &b)| (a, b))
            .collect();
        
        generate_samples(&bounds, golden_ratio, &mut sample_points);
        
        // Evaluate samples
        for amounts in sample_points {
            if let Some(result) = evaluate_merged_cycles(
                &paths,
                &amounts,
                graph,
                stats,
                source,
                gas_price,
                quoter_time,
            ) {
                if result.combined_net_profit > best_profit {
                    best_profit = result.combined_net_profit;
                    best_result = Some(result);
                }
            }
            break; // !!!!!!! Evaluate only one sample per iteration for debugging
        }
        
        // Narrow bounds (simplified: move towards best point)
        if let Some(ref result) = best_result {
            for i in 0..n {
                let best_amt = result.amounts_in[i];
                let midpoint = (lower_bounds[i] + upper_bounds[i]) / 2.0;
                
                if best_amt < midpoint {
                    upper_bounds[i] = midpoint;
                } else {
                    lower_bounds[i] = midpoint;
                }
            }
        }
        
        // Check convergence
        let converged = lower_bounds.iter().zip(upper_bounds.iter())
            .all(|(&a, &b)| (b - a).abs() < tolerance);
        
        if converged {
            log_arb_info!("[MERGE-GSS] Converged after {} iterations", iteration + 1);
            break;
        }
    }
    
    best_result
}

