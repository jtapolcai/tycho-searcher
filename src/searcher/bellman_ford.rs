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
) -> Vec<(f64, f64, f64, Rc<Vec<EdgeIndex>>)> {
    let mut tabu_cycles: HashSet<Rc<Vec<EdgeIndex>>> = HashSet::new();
    let mut best_cycles = Vec::new();
    let mut _amount_in = amount_in_min;
    // Counters for Golden-section search usage
    let mut gss_calls: usize = 0;
    let mut gss_profitable: usize = 0;
    // Total time spent in quoter_amount_out across BF + finalize + GSS
    let mut total_quoter_time: Duration = Duration::from_secs(0);

    loop {
        let gas_price_search = if _amount_in < 0.01 {0.0} else {gas_price};
    let (_profit_wo_gas, _profit_w_gas, cycles) = find_all_negative_cycles_amount(
            graph,
            stats,
            source,
            target,
            max_iterations,
            _amount_in,
            gas_price_search,
            &mut total_quoter_time,
        );

        let mut _last_best_x = None;
        let mut _last_cycle: Option<Rc<Vec<EdgeIndex>>> = None;

        // Collect cycles to avoid borrow checker issues
        let new_cycles: Vec<_> = cycles
            .iter()
            .filter(|(_, _, _, c)| !tabu_cycles.iter().any(|rc| rc.as_ref() == &**c))
            .map(|(a, b, c, cycle_rc)| (*a, *b, *c, Rc::clone(cycle_rc)))
            .collect();

    for (cycle_amount_in, cycle_amount_out, cycle_total_gas, cycle_rc) in new_cycles {
            let cycle: &Vec<EdgeIndex> = &cycle_rc;
            // Check if the cycle starts with WETH
            if let Some(&first_edge) = cycle.first() {
                if let Some((start_node, _)) = graph.edge_endpoints(first_edge) {
                    let start_token = graph.node_weight(start_node).unwrap();
                    let weth_symbol = "WETH";
                    if start_token.token.symbol != weth_symbol {
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
            let profit = cycle_amount_out - cycle_amount_in - gas_price * cycle_total_gas; 
            if profit > 0.0 {
                // ARB log for profitable cycle before optimization (compute path only if logging is enabled)
                if crate::searcher::logging::is_arb_enabled() {
                    let path_str = describe_path(graph, &cycle);
                    let gas_cost_eth = cycle_total_gas * gas_price * 1e-9;
                    let net_eth = (cycle_amount_out - cycle_amount_in) - gas_cost_eth;
                    log_arb_info!(
                        "profitable cycle pre-opt: in={:.6}, out={:.6}, gas_units={:.0}, gas_cost={:.6} WETH, net={:.6} WETH | {}",
                        cycle_amount_in, cycle_amount_out, cycle_total_gas, gas_cost_eth, net_eth, path_str
                    );
                }
                best_cycles.push((cycle_amount_in, cycle_amount_out, cycle_total_gas, Rc::clone(&cycle_rc)));
            }
            tabu_cycles.insert(Rc::clone(&cycle_rc));
            _last_best_x = Some(cycle_amount_in);
            _last_cycle = Some(Rc::clone(&cycle_rc));
            // Use GSS with gas fee to find optimal input amount
        // Count a GSS attempt for this cycle
        gss_calls += 1;
        let gss_result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                golden_section_search_with_gas(
                    &cycle,
                    graph,
            stats,
                    source,
                    amount_in_min,
                    amount_in_max,
                    gss_tolerance,
                    gss_max_iter,
                    gas_price,
                    &mut total_quoter_time,
                )
            })) {
                Ok(result) => result,
                Err(_) => {
                    tabu_cycles.insert(Rc::clone(&cycle_rc));
                    continue;
                }
            };

        if let Some((best_x, amount_out, total_gas_units, best_profit)) = gss_result {
                if best_profit > 0.0 {
            gss_profitable += 1;
                    // Keep gas as units to be consistent with non-optimized entries
                    if crate::searcher::logging::is_arb_enabled() {
                        let path_str = describe_path(graph, &cycle);
                        log_arb_info!(
                            "optimized cycle: in={:.6}, out={:.6}, gas_units={:.0}, net_profit={:.6} WETH | {}",
                            best_x, amount_out, total_gas_units, best_profit, path_str
                        );
                    }
                    best_cycles.push((best_x, amount_out, total_gas_units, Rc::clone(&cycle_rc)));
                }
                tabu_cycles.insert(Rc::clone(&cycle_rc));
                _last_best_x = Some(best_x);
                _last_cycle = Some(Rc::clone(&cycle_rc));
            } else {
                tabu_cycles.insert(Rc::clone(&cycle_rc));
                continue;
            }
        }
    if cycles.iter().all(|(_, _, _, c)| tabu_cycles.iter().any(|rc| rc.as_ref() == &**c)) {
            break;
        }
        break;
    }
    // Report how many cycles triggered the Golden-section search
    if gss_calls > 0 {
        log_arb_info!("Golden-section search called for {} cycle(s), profitable: {}", gss_calls, gss_profitable);
        log_arb_info!("Total quoter time: {:?}", total_quoter_time);
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
fn evaluate_cycle_profit(
    cycle: &[EdgeIndex],
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in: f64,
    gas_price_in_start_token: f64,
    quoter_time: &mut Duration,
) -> (bool, f64, f64, f64, f64) {
    if cycle.is_empty() {
        return (false, 0.0, 0.0, 0.0, 0.0);
    }

    let from_node_data = graph.node_weight(source).unwrap();
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);

    let mut current_amount = (amount_in * multiplier).to_biguint().unwrap_or_default();
    let mut total_gas = BigUint::zero();

    let mut current_node = source;
    for &edge_idx in cycle {
        let (from, to) = graph.edge_endpoints(edge_idx).unwrap();
        if from != current_node { return (false, 0.0, 0.0, 0.0, 0.0); }
        let token_in = &graph.node_weight(from).unwrap().token;
        let token_out = &graph.node_weight(to).unwrap().token;
        let edge = graph.edge_weight(edge_idx).unwrap();
        let t0 = Instant::now();
        let result = edge.borrow().quoter_amount_out(token_in, token_out, &current_amount, stats);
        *quoter_time += t0.elapsed();
        if let Some(price_data) = result {
            current_amount = price_data.amount_out;
            total_gas += price_data.gas;
            current_node = to;
        } else {
            return (false, 0.0, 0.0, 0.0, 0.0);
        }
    }

    let amount_out = current_amount.to_f64().unwrap_or(0.0) / multiplier;
    let profit = amount_out - amount_in;
    let total_gas_f64 = total_gas.to_f64().unwrap_or(0.0);
    let gas_cost_in_start = total_gas_f64 * gas_price_in_start_token * 1e-9;
    let net_profit = profit - gas_cost_in_start;
    (net_profit > 0.0, amount_in, amount_out, total_gas_f64, net_profit)
}

fn golden_section_search_with_gas(
    //&self,
    //stats: &mut Statistics,
    //cycle: &[EdgeIndex],
    //gas_price_in_start_node_token: f64,
    cycle: &[EdgeIndex],
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in_min: f64,
    amount_in_max: f64,
    gss_tolerance: f64,
    gss_max_iter: usize,
    gas_price_in_start_node_token: f64,
    quoter_time: &mut Duration,
) -> Option<(f64, f64, f64, f64)> {
    // 0. lépés: keresd meg a minimális amount_in-t, amivel a kör legalább a gas költséget kitermeli
    let mut min_in = amount_in_min.max(1e-12); // ne legyen nulla
    let mut last_min_in = 0.0;
    let tolerance = 1e-9;
    let max_iter = 20;

    for _ in 0..max_iter {
        let (_profitable, _ain, aout, total_gas, _profit) =
            evaluate_cycle_profit(cycle, graph, stats, source, min_in, gas_price_in_start_node_token, quoter_time);

        let gas_cost_eth = total_gas * gas_price_in_start_node_token * 1e-9;
        let flashloan_fee = min_in * 0.0005;
        let total_cost = gas_cost_eth + flashloan_fee;

        // Mekkora input kell, hogy legalább total_cost legyen a profit?
        // profit = (aout - min_in) - total_cost >= 0
        // aout - min_in = total_cost
        // aout = min_in + total_cost
        // Ha aout < min_in + total_cost, akkor növelni kell min_in-t
        if aout < min_in + total_cost {
            // Lineárisan becsüljük meg, mennyivel kell növelni
            let diff = (min_in + total_cost) - aout;
            last_min_in = min_in;
            min_in += diff.max(min_in * 0.1); // legalább 10%-kal növeljük, hogy gyorsabban konvergáljon
        } else {
            break;
        }
        if (min_in - last_min_in).abs() < tolerance {
            break;
        }
    }

    // Ha a min_in túl nagy lett, akkor nincs értelmes megoldás
    if min_in > amount_in_max {
        return None;
    }

    // Golden section search a megtalált min_in-től indul
    let mut a = min_in;
    let mut b = amount_in_max;
    let gr = (5.0f64.sqrt() - 1.0) / 2.0; // Golden ratio conjugate

    let mut x1 = a + (1.0 - gr) * (b - a);
    let mut x2 = a + gr * (b - a);

    let mut f1 = evaluate_cycle_profit(cycle, graph, stats, source, x1, gas_price_in_start_node_token, quoter_time).4;
    let mut f2 = evaluate_cycle_profit(cycle, graph, stats, source, x2, gas_price_in_start_node_token, quoter_time).4;

    for _ in 0..gss_max_iter {
        if f1 > f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (1.0 - gr) * (b - a);
            f1 = evaluate_cycle_profit(cycle, graph, stats, source, x1, gas_price_in_start_node_token, quoter_time).4;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = evaluate_cycle_profit(cycle, graph, stats, source, x2, gas_price_in_start_node_token, quoter_time).4;        
        }

        if (b - a).abs() < gss_tolerance {
            break;
        }
    }

    let best_amount = (a + b) / 2.0;
    let (is_profitable, final_amount_in, final_amount_out, final_gas, final_profit) =
         evaluate_cycle_profit(cycle, graph, stats, source, best_amount, gas_price_in_start_node_token, quoter_time);

    if is_profitable {
        Some((final_amount_in, final_amount_out, final_gas, final_profit))
    } else {
        None
    }
}
// Golden-section search over input amount including gas fee. Returns Some(best_amount_in, amount_out, total_gas_units, net_profit) if profitable
fn golden_section_search_with_gas_old(
    cycle: &[EdgeIndex],
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in_min: f64,
    amount_in_max: f64,
    gss_tolerance: f64,
    gss_max_iter: usize,
    gas_price_in_start_node_token: f64,
    quoter_time: &mut Duration,
) -> Option<(f64, f64, f64, f64)> {
    // 0) Find a minimal input that at least covers gas cost
    let mut min_in = amount_in_min.max(1e-12); // avoid zero
    let tolerance = 1e-9;
    let max_iter = 20;
    let mut prev_min_in = min_in;

    for _ in 0..max_iter {
        let (_profitable, _ain, aout, total_gas, _profit) =
            evaluate_cycle_profit(cycle, graph, stats, source, min_in, gas_price_in_start_node_token, quoter_time);
        let gas_cost_in_start = total_gas * gas_price_in_start_node_token * 1e-9;
        let total_cost = gas_cost_in_start;

        if aout < min_in + total_cost {
            // Linearly estimate how much to increase
            let diff = (min_in + total_cost) - aout;
            min_in += diff.max(min_in * 0.1); // increase at least 10% to converge faster
        } else {
            break;
        }
        let step = (min_in - prev_min_in).abs();
        prev_min_in = min_in;
        if step < tolerance { break; }
    }

    if min_in > amount_in_max { return None; }

    // Golden section search starting from min_in
    let mut a = min_in;
    let mut b = amount_in_max;
    let gr = (5.0f64.sqrt() - 1.0) / 2.0; // conjugate golden ratio

    let mut x1 = a + (1.0 - gr) * (b - a);
    let mut x2 = a + gr * (b - a);

    let mut f1 = evaluate_cycle_profit(cycle, graph, stats, source, x1, gas_price_in_start_node_token, quoter_time).4;
    let mut f2 = evaluate_cycle_profit(cycle, graph, stats, source, x2, gas_price_in_start_node_token, quoter_time).4;

    for _ in 0..gss_max_iter {
        if f1 > f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (1.0 - gr) * (b - a);
            f1 = evaluate_cycle_profit(cycle, graph, stats, source, x1, gas_price_in_start_node_token, quoter_time).4;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + gr * (b - a);
            f2 = evaluate_cycle_profit(cycle, graph, stats, source, x2, gas_price_in_start_node_token, quoter_time).4;
        }
        if (b - a).abs() < gss_tolerance { break; }
    }

    let best_amount = (a + b) / 2.0;
    let (is_profitable, final_amount_in, final_amount_out, final_gas, final_profit) =
        evaluate_cycle_profit(cycle, graph, stats, source, best_amount, gas_price_in_start_node_token, quoter_time);
    if is_profitable {
        Some((final_amount_in, final_amount_out, final_gas, final_profit))
    } else {
        None
    }
}

