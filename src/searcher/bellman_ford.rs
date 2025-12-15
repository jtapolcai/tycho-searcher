// bellman_ford.rs
use std::rc::Rc;
use crate::log_arb_info;
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use num_bigint::ToBigUint;
use petgraph::{
    graph::{EdgeIndex, Graph, NodeIndex},
    Directed,
    visit::{EdgeRef, NodeIndexable},
};
use std::cell::RefCell;
use std::sync::Arc;

use crate::searcher::{EdgeData,NodeData,Statistics};
use crate::searcher::price_quoter::PriceDataRaw;
use crate::searcher::one_dimensional_optimisation::{golden_section_search_with_gas,find_bottleneck_links};
use std::collections::{HashSet, HashMap};
use tycho_common::{
    simulation::protocol_sim::ProtocolSim,
};

pub fn find_all_negative_cycles(
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    target: NodeIndex,
    max_iterations: usize,
    amount_in_min: f64,
    amount_in_max: f64,
    _max_outer_iterations: usize,
    gss_tolerance: f64,
    gss_max_iter: usize,
    gas_price: f64, 
    max_split: usize,
) -> Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>)> {
    let mut tabu_cycles: HashSet<Rc<Vec<EdgeIndex>>> = HashSet::new();
    let mut best_cycles = Vec::new();
    let mut best_fractional_cycles = Vec::new();
    let mut _amount_in = amount_in_min;
    
    let mut gss_calls: usize = 0;
    let mut gss_profitable: usize = 0;
    let mut total_quoter_time: Duration = Duration::from_secs(0);
    let mut candidate_cycles: Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>)> = Vec::new();
    
    // Use reduced gas price for initial cycle search
    let gas_price_search = gas_price / (max_split as f64);

    loop {
        let search_gas_price = if _amount_in < 0.01 { 0.0 } else { gas_price_search };
        log_arb_info!("Searching cycles with amount_in = {:.6} WETH, gas_price = {:.6} gwei (search mode) src:{} target:{} graph has {} nodes", 
            _amount_in, search_gas_price, source.index(), target.index(), graph.node_count());
        
        let (_profit_wo_gas, _profit_w_gas, cycles) = find_all_negative_cycles_amount(
            graph,
            stats,
            source,
            target,
            max_iterations,
            _amount_in,
            search_gas_price,
            &mut total_quoter_time,
        );

        // Collect cycles to avoid borrow checker issues
        let new_cycles: Vec<_> = cycles
            .iter()
            .filter(|(_, _, _, c)| !tabu_cycles.iter().any(|rc| rc.as_ref() == &**c))
            .map(|(a, b, c, cycle_rc)| (*a, *b, *c, Rc::clone(cycle_rc)))
            .collect();

        // PHASE 1: Collect profitable cycles with reduced gas price
        for (cycle_amount_in, cycle_amount_out, cycle_total_gas, cycle_rc) in new_cycles {
            let cycle: &Vec<EdgeIndex> = &cycle_rc;
            
            // Check if the cycle starts with WETH
            if let Some(&first_edge) = cycle.first() {
                if let Some((start_node, _)) = graph.edge_endpoints(first_edge) {
                    let start_token = graph.node_weight(start_node).unwrap();
                    if start_token.token.symbol != "WETH" {
                        tabu_cycles.insert(Rc::clone(&cycle_rc));
                        continue;
                    }
                } else {
                    tabu_cycles.insert(Rc::clone(&cycle_rc));
                    continue;
                }
            } else {
                tabu_cycles.insert(Rc::clone(&cycle_rc));
                continue;
            }
            
            let profit = cycle_amount_out - cycle_amount_in;
            if profit > 0.0 {
                if crate::searcher::logging::is_arb_enabled() {
                    let path_str = describe_path(graph, &cycle);
                    log_arb_info!(
                        "Found cycle (reduced gas search): in={:.6}, out={:.6}, gas={:.0} | {}",
                        cycle_amount_in, cycle_amount_out, cycle_total_gas, path_str
                    );
                }
                candidate_cycles.push((cycle_amount_in, cycle_amount_out, cycle_total_gas, Rc::clone(&cycle_rc)));
            }
            
            tabu_cycles.insert(Rc::clone(&cycle_rc));
        }
        
        if cycles.iter().all(|(_, _, _, c)| tabu_cycles.iter().any(|rc| rc.as_ref() == &**c)) {
            break;
        }
        break;
    }
        
    if !candidate_cycles.is_empty() {
        log_arb_info!("Found {} candidate cycles with reduced gas_price={:.6} gwei", candidate_cycles.len(), gas_price_search);
        // PHASE 2: Validate all candidates (individual + merged) with full gas_price
        log_arb_info!("=== VALIDATION PHASE: Validating {} individual cycles with full gas_price={:.6} gwei ===", 
            candidate_cycles.len(), gas_price);
        
        for (_cycle_amount_in, _cycle_amount_out, _cycle_total_gas, cycle_rc) in candidate_cycles.iter() {
            let cycle: &Vec<EdgeIndex> = &cycle_rc;
            
            gss_calls += 1;
            
            let validation_result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                golden_section_search_with_gas(
                    &cycle,
                    graph,
                    stats,
                    source,
                    amount_in_min,
                    amount_in_max, 
                    gss_tolerance,
                    gss_max_iter,
                    gas_price, // Use REAL gas price for validation
                    &mut total_quoter_time,
                )
            })) {
                Ok(result) => result,
                Err(_) => None,
            };
            
            if let Some((best_x, amount_out, total_gas_units, net_profit)) = validation_result {
                if net_profit >= 0.0 {
                    gss_profitable += 1;
                    if crate::searcher::logging::is_arb_enabled() {
                        let path_str = describe_path(graph, &cycle);
                        log_arb_info!(
                            "Cycle validated: in={:.6}, out={:.6}, gas_units={:.0}, net_profit={:.6} WETH | {}",
                            best_x, amount_out, total_gas_units, net_profit, path_str
                        );
                    }
                    best_cycles.push((best_x, amount_out, total_gas_units, Rc::clone(&cycle_rc)));
                } else {
                    if crate::searcher::logging::is_arb_enabled() {
                        let path_str = describe_path(graph, &cycle);
                        log_arb_info!(
                            "Cycle not profitable with full gas_price: in={:.6}, net_profit={:.6} WETH | {}",
                            best_x, net_profit, path_str
                        );
                    }
                }
                let shareable_links = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    find_bottleneck_links(
                        &cycle,
                        graph,
                        stats,
                        source,
                        best_x, 
                        gas_price,
                        &mut total_quoter_time,
                    )
                })) {
                    Ok(result) => result.unwrap_or_else(|| Vec::new()),
                    Err(_) => Vec::new(),
                };
                // Only add to fractional cycles if bottleneck links were found
                if !shareable_links.is_empty() {
                    best_fractional_cycles.push((best_x, amount_out, total_gas_units, Rc::clone(&cycle_rc), shareable_links.clone()));
                }
            } 
        }
        
        if gss_calls > 0 {
            log_arb_info!("Validation complete: {}/{} cycles profitable with full gas_price", gss_profitable, gss_calls);
            log_arb_info!("Total quoter time: {:?}", total_quoter_time);
        }
        // PHASE 3: Try merging fractional candidate cycles (up to max_split) with reduced gas price
        if best_fractional_cycles.len() >= 2 && max_split > 1 {
            log_arb_info!("=== MERGE PHASE: Attempting to merge up to {} cycles ===", max_split);
            
            if let Some(merged_result) = crate::searcher::merge_cycles::optimize_merged_cycles(
                best_fractional_cycles.clone(),
                graph,
                stats,
                source,
                amount_in_min,
                gss_tolerance,
                gss_max_iter,
                gas_price, 
                &mut total_quoter_time,
                max_split,
            ) {
                log_arb_info!("✓ Merged cycles found with combined profit = {:.6} ETH (at reduced gas)", merged_result.combined_net_profit);
                // TODO: Convert merged_result to cycle format for validation
                // For now, we'll validate individual cycles only
            } else {
                log_arb_info!("✗ No profitable merged cycles found");
            }
        }
    }
    
    best_cycles
}

pub fn evaluate_cycle(
    cycle: Vec<EdgeIndex>,
    amount_start: BigUint,
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    start_node_idx: NodeIndex,
) -> Result<(BigUint, BigUint), String> {
    // the function handles that a pool is traversed multiple times
    let mut pool_ids: HashMap<String, Arc<dyn ProtocolSim>> = HashMap::new();

    let mut amount = amount_start.clone();
    let mut total_gas = BigUint::zero();

    if cycle.is_empty() {
        return Err("Cycle is empty".to_string());
    }
    let (start_node, _) = graph.edge_endpoints(cycle[0]).ok_or("Invalid edge in cycle")?;
    if start_node != start_node_idx {
        return Err("Cycle does not start at start_node_idx".to_string());
    }
    let mut end_node_idx = start_node_idx;
    for &edge_index in &cycle {
        let (from, to) = graph.edge_endpoints(edge_index).ok_or("Invalid edge in cycle")?;
        if from != end_node_idx {
            return Err(format!("Edge {:?} does not start from the last node {:?}", edge_index, end_node_idx));
        }
        end_node_idx = to;
        let token_in = graph.node_weight(from).unwrap();
        let token_out = graph.node_weight(to).unwrap();

        let edge_data = graph.edge_weight(edge_index).unwrap();
        let pool = edge_data.borrow().pool.clone();
        let pool_address=format!("{:#042x}", pool.id);
        
        let pool_state = if pool_ids.contains_key(&pool_address) {
                    pool_ids.get(&pool_address).unwrap().clone()
                } else {
                    edge_data.borrow().state.clone()
                };

        let amount_out = pool_state
            .get_amount_out(amount.clone(), &*token_in.token, &*token_out.token)
            .ok();

        if let Some(new_amount) = amount_out {
            pool_ids.insert(pool_address.clone(), Arc::from(new_amount.new_state.clone()));
            amount = new_amount.amount.clone();
            total_gas += new_amount.gas.clone();
        } else {
            return Err("Pool simulation failed".to_string());
        }
    }
    Ok((amount, total_gas))
}

pub fn find_all_negative_cycles_amount(
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    target: NodeIndex,
    max_iterations: usize,
    start_token_amount: f64,
    gas_price: f64,
    quoter_time: &mut Duration,
) -> (f64,f64,Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>)> ) {
    let from_node_data = graph.node_weight(source).unwrap();
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);
    let amount_start = (start_token_amount * &multiplier).to_biguint()
        .unwrap_or(BigUint::zero()) ;
    let gas_price_bui=(gas_price* 1e9).to_biguint().unwrap_or(BigUint::zero());
    let (distance, predecessor,distance_with_loop,predecessor_with_loop) = bellman_ford_initialize_relax(
        graph,
        stats,
        source,
        max_iterations,
        amount_start.clone(),
        gas_price_bui,
        quoter_time,
    );

    if distance.is_empty() || predecessor.is_empty() {
        return (0.0, 0.0, Vec::new());
    }
    if source != target {
        println!("Not implemented yet! Source and target are not the same.");
        return (0.0, 0.0, Vec::new());
    }
    let mut cycles: Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>)> = Vec::new();
    let mut max_profit_without_gas= 0.0; 
    let mut max_profit_with_gas= 0.0; 

    // Track how many final-edge quoter calls are performed during cycle materialization
    let mut final_quoter_calls: usize = 0;
    for edge_ref in graph.edge_references() {
        let edge_idx = edge_ref.id();
        let from = edge_ref.source();
        let to = edge_ref.target();
        let from_idx = graph.to_index(from);
        let to_idx = graph.to_index(to);
        if to_idx != source.index() {
            continue; // we handle the edges that point back to the source node in the end
        }
        let from_node_data = graph.node_weight(from).unwrap();
        let to_node_data = graph.node_weight(to).unwrap();
        if distance[from_idx].is_zero() {
            continue;
        }
        if let Some(edge) = graph.edge_weight(edge_idx) {
            final_quoter_calls += 1;
            let t0 = Instant::now();
            let result = edge.borrow().quoter_amount_out(
                &from_node_data.token,
                &to_node_data.token,
                &distance[from_idx].amount_out,
                stats,
            );
            *quoter_time += t0.elapsed();
            if let Some(price_data) = result {
                if price_data.amount_out > amount_start {
                    let profit = price_data.amount_out - amount_start.clone();
                    let profit_f64= profit.to_f64().unwrap_or(0.0)/multiplier;
                    let total_gas=price_data.gas + distance[from_idx].gas.clone();
                    let total_gas_f64= total_gas.to_f64().unwrap_or(0.0); // Keep as gas units
                    let total_gas_cost_eth = total_gas_f64 * gas_price * 1e-9; // Convert to eth for profit calculation
                    // Gas cost calculation: gas_units * gas_price (in gwei)
                    if profit_f64 > max_profit_without_gas{
                        max_profit_without_gas=profit_f64;
                    }
                    if profit_f64 - total_gas_cost_eth > max_profit_with_gas {
                        max_profit_with_gas = profit_f64 - total_gas_cost_eth;
                    }
                    // Profitable only if the profit is greater than the gas cost
                    if profit_f64 > total_gas_cost_eth {
                        let path = get_path(edge_idx, source, graph, &predecessor);
                        
                        // Compute the amount_out at the end of the cycle
                        let amount_out_f64 = profit_f64 + start_token_amount;
                        
                        cycles.push((start_token_amount, amount_out_f64, total_gas_f64, Rc::new(path)));
                    }
                } 
            }
        }
        if distance_with_loop[from_idx].is_zero() {
            continue;
        }
        if let Some(edge) = graph.edge_weight(edge_idx) {
            final_quoter_calls += 1;
            let t0 = Instant::now();
            let result = edge.borrow().quoter_amount_out(
                &from_node_data.token,
                &to_node_data.token,
                &distance_with_loop[from_idx].amount_out,
                stats,
            );
            *quoter_time += t0.elapsed();
            if let Some(price_data) = result {
                if price_data.amount_out > amount_start {
                    let profit = price_data.amount_out - amount_start.clone();
                    let profit_f64= profit.to_f64().unwrap_or(0.0)/multiplier;
                    let total_gas=price_data.gas + distance_with_loop[from_idx].gas.clone(); // Fix: use distance_with_loop
                    let total_gas_f64= total_gas.to_f64().unwrap_or(0.0); // Keep as gas units
                    let total_gas_cost_eth = total_gas_f64 * gas_price * 1e-9; // Convert to eth for profit calculation
                    // Gas cost calculation: gas_units * gas_price (in gwei)
                    if profit_f64 > max_profit_without_gas{
                        max_profit_without_gas=profit_f64;
                    }
                    if profit_f64 - total_gas_cost_eth > max_profit_with_gas {
                        max_profit_with_gas = profit_f64 - total_gas_cost_eth;
                    }
                    // Profitable only if the profit is greater than the gas cost
                    if profit_f64 > total_gas_cost_eth {
                        let path = get_path_with_loop(edge_idx, source, graph, &predecessor, &predecessor_with_loop);
                        
                        // Compute the amount_out at the end of the cycle
                        let amount_out_f64 = profit_f64 + start_token_amount;
                        
                        cycles.push((start_token_amount, amount_out_f64, total_gas_f64, Rc::new(path)));
                    }
                } 
            }
        }
    }
    // Report how many quoter calls were needed to materialize cycles from BF distances
    log_arb_info!("BF cycle finalize quoter calls: {}", final_quoter_calls);
    (max_profit_without_gas,max_profit_with_gas,cycles)
}


pub fn bellman_ford_initialize_relax(
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    max_iterations: usize,
    amount_start: BigUint,
    gas_price_bui: BigUint,
    quoter_time: &mut Duration,
) -> (Vec<PriceDataRaw>, Vec<Option<EdgeIndex>>,Vec<PriceDataRaw>, Vec<Option<EdgeIndex>>) {
    let max_index = graph.node_bound();
    let mut distance = vec![PriceDataRaw { amount_out: BigUint::zero(), gas: BigUint::zero() }; max_index];
    let mut predecessor = vec![None; max_index];
    let mut distance_with_loop = vec![PriceDataRaw { amount_out: BigUint::zero(), gas: BigUint::zero() }; max_index];
    let mut predecessor_with_loop = vec![None; max_index];

    let source_idx = graph.to_index(source);
    distance[source_idx].amount_out = amount_start;

    for _ in 0..max_iterations {
        let mut did_update = false;
        for edge_ref in graph.edge_references() {
            let edge_idx = edge_ref.id();
            let from = edge_ref.source();
            let to = edge_ref.target();
            if to==source {
                continue; // we handle the edges that point back to the source node in the end
            }
            let from_idx = graph.to_index(from);
            let to_idx = graph.to_index(to);
            let from_node_data = graph.node_weight(from).unwrap();
            let to_node_data = graph.node_weight(to).unwrap();

            if let Some(edge) = graph.edge_weight(edge_idx) {
                if distance[from_idx].is_zero() {
                    continue;
                }
                let t0 = Instant::now();
                let result = edge.borrow().quoter_amount_out(
                        &from_node_data.token,
                        &to_node_data.token,
                        &distance[from_idx].amount_out,
                        stats,
                    );
                *quoter_time += t0.elapsed();
                if let Some(mut price_data) = result {
                    // Compare (amount_out + gas*price) without extra clones
                    let lhs_amt = &price_data.amount_out;
                    let lhs_gas_cost = &price_data.gas * &gas_price_bui;
                    let rhs_amt = &distance[to_idx].amount_out;
                    let rhs_gas_cost = &distance[to_idx].gas * &gas_price_bui;
                    if lhs_amt + lhs_gas_cost > rhs_amt + rhs_gas_cost {
                        if !has_node_in_path(to_idx, from_idx, graph, &predecessor) {
                            price_data.gas += &distance[from_idx].gas;
                            distance[to_idx] = price_data;
                            predecessor[to_idx] = Some(edge_idx);
                            did_update = true;
                        } else {
                        }
                    }
                }
                if distance_with_loop[from_idx].is_zero() {
                    continue;
                }
                let t0 = Instant::now();
                let result = edge.borrow().quoter_amount_out(
                        &from_node_data.token,
                        &to_node_data.token,
                        &distance_with_loop[from_idx].amount_out,
                        stats,
                    );
                *quoter_time += t0.elapsed();
                if let Some(mut price_data) = result {
                    let lhs_amt = &price_data.amount_out;
                    let lhs_gas_cost = &price_data.gas * &gas_price_bui;
                    let rhs_amt = &distance_with_loop[to_idx].amount_out;
                    let rhs_gas_cost = &distance_with_loop[to_idx].gas * &gas_price_bui;
                    if lhs_amt + lhs_gas_cost > rhs_amt + rhs_gas_cost {
                        if !has_node_in_path(to_idx, from_idx, graph, &predecessor_with_loop) {
                            price_data.gas += &distance_with_loop[from_idx].gas;
                            distance_with_loop[to_idx] = price_data;
                            predecessor_with_loop[to_idx] = Some(edge_idx);
                            did_update = true;
                        } else {
                        }
                    }
                }
            }
        }
        if !did_update {
            break;
        }
    }

    (distance, predecessor, distance_with_loop, predecessor_with_loop)
}

// Improved has_node_in_path function
pub fn has_node_in_path(
    start: usize, 
    from_idx: usize, 
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>, 
    predecessor: &Vec<Option<EdgeIndex>>,
) -> bool
{
    let mut counter = 50; // Nagyobb counter
    let mut idx_node = from_idx;
    let mut visited = std::collections::HashSet::new();
    
    loop {
    // Cycle detection — if we've already visited here
        if visited.contains(&idx_node) {
            return true;
        }
        visited.insert(idx_node);
        
    // If we reached the target
        if idx_node == start {
            return true;
        }
        
    // Find predecessor
        let edge_index = match predecessor[idx_node] {
            Some(e) => e,
            None => {
                return false; // No path
            }
        };
        
    // Next node
        let (ancestor, to_node) = graph.edge_endpoints(edge_index).unwrap();
        let ancestor_idx = graph.to_index(ancestor);
        
    // Verify that the edge indeed points to the current node
        let to_idx = graph.to_index(to_node);
        if to_idx != idx_node {
            return false;
        }
        
        idx_node = ancestor_idx;
        
    // Timeout protection
        counter -= 1;
        if counter <= 0 {
            return true; // Conservative: treat as a cycle
        }
    }
} 

// Completely rewritten get_path_with_loop function
pub fn get_path_with_loop(
    edge_idx_: EdgeIndex,
    source: NodeIndex,
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>, 
    predecessor: &Vec<Option<EdgeIndex>>,
    predecessor_with_loop: &Vec<Option<EdgeIndex>>,
) -> Vec<EdgeIndex>
{
    let mut path = Vec::new();
    let mut current_edge = edge_idx_;
    let mut visited_nodes = std::collections::HashSet::new();
    let mut phase = 1;
    let max_steps = 100; // Safe upper bound
    
    loop {
        path.push(current_edge);
        
        let (from_node, _to_node) = match graph.edge_endpoints(current_edge) {
            Some(endpoints) => endpoints,
            None => {
                break;
            }
        };
        
    // If we reached the source, we're done
        if from_node == source {
            break;
        }
        
        let from_idx = graph.to_index(from_node);
        
    // Phase switch: if this node was already seen, switch to predecessor
        if visited_nodes.contains(&from_node) && phase == 1 {
            phase = 2;
        }
        
        visited_nodes.insert(from_node);
        
    // Choose the next edge based on the current phase
        let next_edge = if phase == 1 {
            predecessor_with_loop[from_idx]
        } else {
            predecessor[from_idx]
        };
        
        match next_edge {
            Some(edge) => {
                current_edge = edge;
            }
            None => {
                // If in phase 1 there is no predecessor_with_loop, try the regular predecessor
                if phase == 1 {
                    if let Some(fallback_edge) = predecessor[from_idx] {
                        current_edge = fallback_edge;
                        phase = 2;
                        continue;
                    }
                }
                break;
            }
        }
    }
    
    if path.len() >= max_steps {
        println!("WARNING: Path building reached max steps ({}), possible infinite loop", max_steps);
    }
    
    path.reverse();
    return path;
}

// Improved get_path function as well
pub fn get_path(
    edge_idx_: EdgeIndex,
    till: NodeIndex,
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>, 
    predecessor: &Vec<Option<EdgeIndex>>,
) -> Vec<EdgeIndex>
{
    let mut path = Vec::new();
    let mut current_edge = edge_idx_;
    let mut visited_nodes = std::collections::HashSet::new();
    let max_steps = 100;
    
    loop {
        path.push(current_edge);
        
        let (from_node, _to_node) = match graph.edge_endpoints(current_edge) {
            Some(endpoints) => endpoints,
            None => {
                break;
            }
        };
        
    // Target reached
        if from_node == till {
            break;
        }
        
    // Cycle detection
        if visited_nodes.contains(&from_node) {
            break;
        }
        visited_nodes.insert(from_node);
        
    // Next edge
        let from_idx = graph.to_index(from_node);
        match predecessor[from_idx] {
            Some(edge) => {
                current_edge = edge;
            }
            None => {
                break;
            }
        }
    }
    
    if path.len() >= max_steps {
        println!("WARNING: Path building reached max steps, possible infinite loop");
    }
    
    path.reverse();
    return path;
}

pub fn describe_path(
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    path: &[EdgeIndex],
) -> String {
    let mut result = String::new();

    for &edge_idx in path {
        if let Some(edge_ref) = graph.edge_references().find(|e| e.id() == edge_idx) {
            let source = edge_ref.source();
            let target = edge_ref.target();

            let source_name = graph
                .node_weight(source)
                .map(|n| n.token.symbol.clone())
                .unwrap_or_else(|| "[missing src]".to_string());

            let target_name = graph
                .node_weight(target)
                .map(|n| n.token.symbol.clone())
                .unwrap_or_else(|| "[missing tgt]".to_string());

            let edge_data = graph
                .edge_weight(edge_idx)
                //.map(|e| e.borrow().pool.id.clone())
                .map(|e| format!("0x{}", hex::encode(&e.borrow().pool.id)))
                .unwrap_or_else(|| "[no pool]".to_string());

            let pool_str = &edge_data.chars().take(5).collect::<String>();

            result.push_str(&format!(
                "{} ={}=> {} | ",
                source_name, pool_str, target_name
            ));
        } else {
            result.push_str(&format!("[invalid edge index {:?}]", edge_idx));
        }
    }
    result
}


// Evaluate profit for a given cycle and input amount. Returns (is_profitable, amount_in, amount_out, total_gas_units, net_profit_in_start_token)
