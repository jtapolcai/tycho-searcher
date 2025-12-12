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

/// Represents a merged cycle result with optimized input amounts
#[derive(Debug, Clone)]
pub struct MergedCycleResult {
    pub cycles: Vec<Rc<Vec<EdgeIndex>>>,
    pub amounts_in: Vec<f64>,
    pub amounts_out: Vec<f64>,
    pub net_profits: Vec<f64>,
    pub total_gas: f64,
    pub combined_net_profit: f64,
}

/// Build a map of which cycles use which edges
fn build_edge_usage_map(
    cycles: &[(f64, f64, f64, Rc<Vec<EdgeIndex>>)]
) -> HashMap<EdgeIndex, Vec<usize>> {
    let mut edge_to_cycles: HashMap<EdgeIndex, Vec<usize>> = HashMap::new();
    
    for (cycle_idx, (_, _, _, cycle)) in cycles.iter().enumerate() {
        for &edge in cycle.as_ref() {
            edge_to_cycles.entry(edge).or_insert_with(Vec::new).push(cycle_idx);
        }
    }
    
    edge_to_cycles
}

/// Evaluate multiple cycles together as a DAG, sharing gas costs on overlapping edges
fn evaluate_merged_cycles(
    cycles: &[Rc<Vec<EdgeIndex>>],
    amounts_in: &[f64],
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    gas_price: f64,
    quoter_time: &mut Duration,
) -> Option<MergedCycleResult> {
    if cycles.is_empty() || cycles.len() != amounts_in.len() {
        return None;
    }

    let n = cycles.len();
    let from_node_data = graph.node_weight(source)?;
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);

    // Log cycle paths for visibility
    for (idx, cycle_rc) in cycles.iter().enumerate() {
        let path_str = describe_path(graph, cycle_rc.as_ref());
        log_arb_info!("[MERGE] Cycle {} path: {}", idx, path_str);
    }

    
    
    // Build a DAG from the cycles
    // Nodes: per-cycle source (0 step), sink, and cycle edges as transitions
    let mut dag: DiGraph<(usize, NodeIndex), (usize, EdgeIndex)> = DiGraph::new();
    
    // Create sink node in DAG (shared for all cycles)
    let sink_dag = dag.add_node((n, source)); // All cycles end at sink (same as source after cycle)

    // Map from (cycle_id, step) to DAG node and reverse lookup
    let mut dag_nodes: std::collections::HashMap<(usize, usize), PetgraphNodeIndex> = std::collections::HashMap::new();

    // Build DAG edges from cycle edges and track edge usage
    let mut edge_to_cycles: HashMap<EdgeIndex, Vec<(usize, usize)>> = HashMap::new(); // edge -> [(cycle_id, step)]

    for (cycle_id, cycle) in cycles.iter().enumerate() {
        // Per-cycle source node with step 0
        let source_dag = dag.add_node((cycle_id, source));
        dag_nodes.insert((cycle_id, 0), source_dag);

        let mut prev_node = source_dag;

        for (step, &edge_idx) in cycle.iter().enumerate() {
            edge_to_cycles.entry(edge_idx).or_insert_with(Vec::new).push((cycle_id, step));

            let (_from, to) = graph.edge_endpoints(edge_idx)?;
            let next_node = dag_nodes
                .entry((cycle_id, step + 1))
                .or_insert_with(|| {
                    let node = dag.add_node((cycle_id, to));
                    node
                });

            dag.add_edge(prev_node, *next_node, (cycle_id, edge_idx));
            prev_node = *next_node;
        }

        // Connect last node of cycle to sink
        dag.add_edge(prev_node, sink_dag, (cycle_id, cycle[0])); // dummy edge
    }
    
    // Log DAG structure before toposort
    log_arb_info!("[MERGE] DAG structure:");
    for node_idx in dag.node_indices() {
        let node_data = dag.node_weight(node_idx)?;
        let out_count = dag.edges(node_idx).count();
        log_arb_info!("[MERGE]   Node {:?} = (cycle {}, token_idx {:?}), {} outgoing edges", 
            node_idx, node_data.0, node_data.1, out_count);
    }
    
    // Topological sort
    let topo_order = toposort(&dag, None).ok()?;
    
    log_arb_info!("[MERGE] DAG has {} nodes in topo order: {:?}", topo_order.len(), topo_order);
    log_arb_info!("[MERGE] Edge usage map:");
    for (edge_idx, usages) in edge_to_cycles.iter() {
        log_arb_info!("[MERGE]   Edge {:?} used by: {:?}", edge_idx, usages);
    }

    // Initialize per-node amounts: start at each cycle's source
    let mut amounts_at: std::collections::HashMap<(usize, usize), BigUint> = std::collections::HashMap::new();
    for (cycle_id, &amt) in amounts_in.iter().enumerate() {
        let scaled = ((amt * multiplier).to_biguint()).unwrap_or(BigUint::from(0u32));
        amounts_at.insert((cycle_id, 0), scaled);
    }
    
    log_arb_info!("[MERGE] Starting DAG evaluation with {} cycles", n);
    for (i, amt) in amounts_in.iter().enumerate() {
        log_arb_info!("[MERGE] Cycle {}: input = {:.6}", i, amt);
    }
    
    let mut total_gas = BigUint::from(0u32);
    let mut gas_paid_edges: HashSet<EdgeIndex> = HashSet::new();
    let mut edge_step_counter = 0;
    
    // Process edges in topological order
    for node_idx in topo_order {
        // Get outgoing edges from this node
        let outgoing: Vec<_> = dag.edges(node_idx).map(|e| (e.id(), *e.weight())).collect();
        
        log_arb_info!("[MERGE] Processing DAG node {:?} with {} outgoing edges", node_idx, outgoing.len());
        
        for (dag_edge_idx, (cycle_id, edge_idx)) in outgoing {
            let (_from, target) = dag.edge_endpoints(dag_edge_idx)?;
            
            // Skip dummy edge to sink
            if target == sink_dag {
                continue;
            }

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

                log_arb_info!("[MERGE] Step {}: SHARED EDGE {} → {} (used by {} cycles)",
                    edge_step_counter, token_in.symbol, token_out.symbol, sharing_cycles.len());

                for &(other_cycle_id, step) in sharing_cycles {
                    let amt = amounts_at.get(&(other_cycle_id, step)).cloned().unwrap_or_else(|| BigUint::from(0u32));
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
                    edge_step_counter += 1;
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

                for (other_cycle_id, step) in cycle_shares {
                    let amt_in = amounts_at.get(&(other_cycle_id, step)).cloned().unwrap_or_else(|| BigUint::from(0u32));
                    let share = amt_in.to_f64()? / total_in_f64;
                    let proportional_out = (total_out_f64 * share).to_biguint()?;
                    let prop_out_f64 = proportional_out.to_f64()? / multiplier_out;
                    log_arb_info!("[MERGE]   Cycle {} gets {:.6} ({:.2}% share)",
                        other_cycle_id, prop_out_f64, share * 100.0);
                    amounts_at.insert((other_cycle_id, step + 1), proportional_out);
                }
            } else {
                // Exclusive edge: query normally
                let step = sharing_cycles.get(0).map(|&(_, s)| s).unwrap_or(0);
                let amt_in = amounts_at.get(&(cycle_id, step)).cloned().unwrap_or_else(|| BigUint::from(0u32));

                if amt_in == BigUint::from(0u32) {
                    log_arb_info!("[MERGE] Step {}: EXCLUSIVE EDGE {} → {} (Cycle {}) has zero input, skipping",
                        edge_step_counter, token_in.symbol, token_out.symbol, cycle_id);
                    edge_step_counter += 1;
                    continue;
                }

                let current_amt_f64 = amt_in.to_f64()?;
                log_arb_info!("[MERGE] Step {}: EXCLUSIVE EDGE {} → {} (Cycle {})",
                    edge_step_counter, token_in.symbol, token_out.symbol, cycle_id);
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

                amounts_at.insert((cycle_id, step + 1), result.amount_out);
                total_gas += &result.gas;
            }

            edge_step_counter += 1;
        }
    }
    
    // Calculate final results
    let mut amounts_out = vec![0.0; n];
    for (i, cycle) in cycles.iter().enumerate() {
        if let Some(amount) = amounts_at.get(&(i, cycle.len())) {
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
    
    Some(MergedCycleResult {
        cycles: cycles.to_vec(),
        amounts_in: amounts_in.to_vec(),
        amounts_out,
        net_profits,
        total_gas: total_gas_f64,
        combined_net_profit: combined_profit,
    })
}

/// N-dimensional optimization for multiple cycles
/// Selects k-sized subsets of cycles and optimizes each with GSS
pub fn optimize_merged_cycles(
    cycles: Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>)>,
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in_min: f64,
    amount_in_max: f64,
    tolerance: f64,
    max_iter: usize,
    gas_price: f64,
    quoter_time: &mut Duration,
) -> Option<MergedCycleResult> {
    let n = cycles.len();
    if n == 0 {
        return None;
    }

    log_arb_info!("Optimizing {} cycles...", n);

    // Try different subset sizes (k=2, k=3, ..., up to n)
    let mut best_result: Option<MergedCycleResult> = None;
    let mut best_profit = f64::NEG_INFINITY;
    
    for k in 2..=n.min(2) { // Try k=2 and k=3 (configurable)
        log_arb_info!("[MERGE] Testing subsets of size k={}", k);
        
        // Generate all C(n, k) subsets
        let subsets = generate_subsets(n, k);
        log_arb_info!("[MERGE] Found {} subsets of size {}", subsets.len(), k);
        
        for subset_indices in subsets {
            // Extract the cycles for this subset
            let subset_cycles: Vec<_> = subset_indices.iter()
                .map(|&i| cycles[i].clone())
                .collect();
            
            log_arb_info!("[MERGE] Optimizing subset: {:?}", subset_indices);
            
            // Run GSS on this subset
            if let Some(result) = gss_optimize_subset(
                &subset_cycles,
                graph,
                stats,
                source,
                amount_in_min,
                amount_in_max,
                tolerance,
                max_iter,
                gas_price,
                quoter_time,
            ) {
                if result.combined_net_profit > best_profit {
                    best_profit = result.combined_net_profit;
                    log_arb_info!("[MERGE] New best: subset {:?} with profit {:.6}", 
                        subset_indices, result.combined_net_profit);
                    best_result = Some(result);
                }
            }
            break; // !!!!!!! Only process one subset for debugging
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

/// Generate all C(n, k) subsets of indices
fn generate_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    
    fn backtrack(n: usize, k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        
        for i in start..n {
            current.push(i);
            backtrack(n, k, i + 1, current, result);
            current.pop();
        }
    }
    
    let mut current = Vec::new();
    backtrack(n, k, 0, &mut current, &mut result);
    result
}

/// GSS optimization for a subset of cycles
fn gss_optimize_subset(
    cycle_paths: &[(f64, f64, f64, Rc<Vec<EdgeIndex>>)],
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in_min: f64,
    amount_in_max: f64,
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
