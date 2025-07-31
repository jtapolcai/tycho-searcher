// bellman_ford.rs
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::rc::Rc;

lazy_static::lazy_static! {
    static ref LOG_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);
    static ref LOGGING_ENABLED: AtomicBool = AtomicBool::new(false);
}

#[allow(dead_code)]
fn get_log_file() -> Result<std::sync::MutexGuard<'static, Option<std::fs::File>>, std::sync::PoisonError<std::sync::MutexGuard<'static, Option<std::fs::File>>>> {
    let mut guard = LOG_FILE.lock()?;
    if guard.is_none() {
        *guard = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("log.txt")
            .ok();
    }
    Ok(guard)
}

pub fn enable_logging() {
    LOGGING_ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable_logging() {
    LOGGING_ENABLED.store(false, Ordering::Relaxed);
}

pub fn is_logging_enabled() -> bool {
    LOGGING_ENABLED.load(Ordering::Relaxed)
}

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
use tycho_simulation::protocol::state::ProtocolSim;

// pub struct CycleSearch {
//     graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
//     stats: &mut Statistics,
//     source: NodeIndex,
//     target: NodeIndex,
//     max_iterations: usize,
//     amount_in_min: f64,
//     amount_in_max: f64,
//     max_outer_iterations: usize,
//     gss_tolerance: f64,
//     gss_max_iter: usize,
//     gas_price: f64,
//     // the reselts are tuples: amount_in, gas, cycle, the path from start node, and the path to target node 
//     cycles: mut Vec<(f64, f64, Vec<EdgeIndex>, Option<Vec<EdgeIndex>>, Option<Vec<EdgeIndex>>)>, 
// }


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

    loop {
        let gas_price_search = if _amount_in < 0.01 {0.0} else {gas_price};
        let (_profit_wo_gas, _profit_w_gas, cycles) = find_all_negative_cycles_amount(
            graph,
            stats,
            source,
            target,
            max_iterations,
            _amount_in,
            gas_price_search
        );

        let mut _last_best_x = None;
        let mut _last_cycle: Option<Rc<Vec<EdgeIndex>>> = None;

        // Collect cycles to avoid borrow checker issues
        let new_cycles: Vec<_> = cycles
            .iter()
            .filter(|(_, _, _, c)| !tabu_cycles.iter().any(|rc| rc.as_ref() == c))
            .map(|(a, b, c, cycle)| (*a, *b, *c, Rc::new(cycle.clone())))
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
                best_cycles.push((cycle_amount_in, cycle_amount_out, cycle_total_gas, Rc::clone(&cycle_rc)));
            }
            tabu_cycles.insert(Rc::clone(&cycle_rc));
            _last_best_x = Some(cycle_amount_in);
            _last_cycle = Some(Rc::clone(&cycle_rc));
            // WETH check before GSS optimization
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
            let (best_x, best_profit, best_gas) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                optimize_cycle_gss(
                    &cycle,
                    graph,
                    source,
                    amount_in_min,
                    amount_in_max,
                    gss_tolerance,
                    gss_max_iter,
                    gas_price
                )
            })) {
                Ok(result) => result,
                Err(_) => {
                    tabu_cycles.insert(Rc::clone(&cycle_rc));
                    continue;
                }
            };
            if best_profit > 0.0 {
                let amount_out = best_x + best_profit + best_gas*gas_price;
                best_cycles.push((best_x, amount_out, best_gas, Rc::clone(&cycle_rc)));
            }
            tabu_cycles.insert(Rc::clone(&cycle_rc));
            _last_best_x = Some(best_x);
            _last_cycle = Some(Rc::clone(&cycle_rc));
        }
        if cycles.iter().all(|(_, _, _, c)| tabu_cycles.iter().any(|rc| rc.as_ref() == c)) {
            break;
        }
        break;
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
) -> (f64,f64,Vec<(f64, f64, f64, Vec<EdgeIndex>)>) {
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
        gas_price_bui
    );

    if distance.is_empty() || predecessor.is_empty() {
        return (0.0, 0.0, Vec::new());
    }
    if source != target {
        println!("Not implemented yet! Source and target are not the same.");
        return (0.0, 0.0, Vec::new());
    }
    let mut cycles = Vec::new();
    let mut max_profit_without_gas= 0.0; 
    let mut max_profit_with_gas= 0.0; 

    for edge_ref in graph.edge_references() {
        let edge_idx = edge_ref.id();
        let (from, to) = graph.edge_endpoints(edge_idx).unwrap();
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
            let result = edge.borrow().quoter_amount_out(
                &from_node_data.token,
                &to_node_data.token,
                &distance[from_idx].amount_out,
                stats,
            );
            if let Some(price_data) = result {
                if price_data.amount_out > amount_start {
                    let profit = price_data.amount_out - amount_start.clone();
                    let profit_f64= profit.to_f64().unwrap_or(0.0)/multiplier;
                    let total_gas=price_data.gas + distance[from_idx].gas.clone();
                    let total_gas_f64= total_gas.to_f64().unwrap_or(0.0); // Keep as gas units
                    let total_gas_cost_eth = total_gas_f64 * gas_price * 1e-9; // Convert to eth for profit calculation
                    // Gas költség kiszámítása: gas_units * gas_price (gwei-ben)
                    if profit_f64 > max_profit_without_gas{
                        max_profit_without_gas=profit_f64;
                    }
                    if profit_f64 - total_gas_cost_eth > max_profit_with_gas {
                        max_profit_with_gas = profit_f64 - total_gas_cost_eth;
                    }
                    // Profitábilis csak akkor, ha a profit nagyobb mint a gas költség
                    if profit_f64 > total_gas_cost_eth {
                        let path = get_path(edge_idx, source, graph, &predecessor);
                        
                        // Kiszámítjuk az amount_out értékét a ciklus végén
                        let amount_out_f64 = profit_f64 + start_token_amount;
                        
                        cycles.push((start_token_amount, amount_out_f64, total_gas_f64, path));
                    }
                } 
            }
        }
        if distance_with_loop[from_idx].is_zero() {
            continue;
        }
        if let Some(edge) = graph.edge_weight(edge_idx) {
            let result = edge.borrow().quoter_amount_out(
                &from_node_data.token,
                &to_node_data.token,
                &distance_with_loop[from_idx].amount_out,
                stats,
            );
            if let Some(price_data) = result {
                if price_data.amount_out > amount_start {
                    let profit = price_data.amount_out - amount_start.clone();
                    let profit_f64= profit.to_f64().unwrap_or(0.0)/multiplier;
                    let total_gas=price_data.gas + distance_with_loop[from_idx].gas.clone(); // Fix: use distance_with_loop
                    let total_gas_f64= total_gas.to_f64().unwrap_or(0.0); // Keep as gas units
                    let total_gas_cost_eth = total_gas_f64 * gas_price * 1e-9; // Convert to eth for profit calculation
                    // Gas költség kiszámítása: gas_units * gas_price (gwei-ben)
                    if profit_f64 > max_profit_without_gas{
                        max_profit_without_gas=profit_f64;
                    }
                    if profit_f64 - total_gas_cost_eth > max_profit_with_gas {
                        max_profit_with_gas = profit_f64 - total_gas_cost_eth;
                    }
                    // Profitábilis csak akkor, ha a profit nagyobb mint a gas költség
                    if profit_f64 > total_gas_cost_eth {
                        let path = get_path_with_loop(edge_idx, source, graph, &predecessor, &predecessor_with_loop);
                        
                        // Kiszámítjuk az amount_out értékét a ciklus végén
                        let amount_out_f64 = profit_f64 + start_token_amount;
                        
                        cycles.push((start_token_amount, amount_out_f64, total_gas_f64, path.clone()));
                    }
                } 
            }
        }
    }
    (max_profit_without_gas,max_profit_with_gas,cycles)
}


pub fn bellman_ford_initialize_relax(
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    max_iterations: usize,
    amount_start: BigUint,
    gas_price_bui: BigUint
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
            let (from, to) = graph.edge_endpoints(edge_idx).unwrap();
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
                let result = edge.borrow().quoter_amount_out(
                        &from_node_data.token,
                        &to_node_data.token,
                        &distance[from_idx].amount_out,
                        stats,
                    );
                if let Some(mut price_data) = result {
                    if price_data.amount_out.clone() + price_data.gas.clone() * gas_price_bui.clone() > 
                        distance[to_idx].amount_out.clone() + distance[to_idx].gas.clone() * gas_price_bui.clone() {
                        if !has_node_in_path(to_idx, from_idx, graph, &predecessor) {
                            price_data.gas += distance[from_idx].gas.clone();
                            distance[to_idx] = price_data.clone();
                            predecessor[to_idx] = Some(edge_idx);
                            did_update = true;
                        } else {
                        }
                    }
                }
                if distance_with_loop[from_idx].is_zero() {
                    continue;
                }
                let result = edge.borrow().quoter_amount_out(
                        &from_node_data.token,
                        &to_node_data.token,
                        &distance_with_loop[from_idx].amount_out,
                        stats,
                    );
                if let Some(mut price_data) = result {
                    if price_data.amount_out.clone() + price_data.gas.clone() * gas_price_bui.clone() > 
                        distance_with_loop[to_idx].amount_out.clone() + distance_with_loop[to_idx].gas.clone() * gas_price_bui.clone() {
                        if !has_node_in_path(to_idx, from_idx, graph, &predecessor_with_loop) {
                            price_data.gas += distance_with_loop[from_idx].gas.clone();
                            distance_with_loop[to_idx] = price_data.clone();
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

// Javított has_node_in_path függvény
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
        // Ciklus detektálás - ha már jártunk itt
        if visited.contains(&idx_node) {
            return true;
        }
        visited.insert(idx_node);
        
        // Ha elértük a célpontot
        if idx_node == start {
            return true;
        }
        
        // Predecessor keresése
        let edge_index = match predecessor[idx_node] {
            Some(e) => e,
            None => {
                return false; // Nincs út
            }
        };
        
        // Következő node
        let (ancestor, to_node) = graph.edge_endpoints(edge_index).unwrap();
        let ancestor_idx = graph.to_index(ancestor);
        
        // Ellenőrizzük, hogy az él valóban a current node-ba mutat
        let to_idx = graph.to_index(to_node);
        if to_idx != idx_node {
            return false;
        }
        
        idx_node = ancestor_idx;
        
        // Timeout védelem
        counter -= 1;
        if counter <= 0 {
            return true; // Konzervatív: ciklusnak tekintjük
        }
    }
} 

// Teljesen újraírt get_path_with_loop függvény
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
    let max_steps = 100; // Biztonságos felső határ
    
    loop {
        path.push(current_edge);
        
        let (from_node, _to_node) = match graph.edge_endpoints(current_edge) {
            Some(endpoints) => endpoints,
            None => {
                break;
            }
        };
        
        // Ha elértük a source-t, kész vagyunk
        if from_node == source {
            break;
        }
        
        let from_idx = graph.to_index(from_node);
        
        // Fázis váltás: ha már láttuk ezt a node-ot, váltunk predecessor-ra
        if visited_nodes.contains(&from_node) && phase == 1 {
            phase = 2;
        }
        
        visited_nodes.insert(from_node);
        
        // Következő él kiválasztása fázis szerint
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
                // Ha phase 1-ben nincs predecessor_with_loop, próbáljuk a regular-t
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

// Javított get_path függvény is
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
        
        // Célpont elérése
        if from_node == till {
            break;
        }
        
        // Ciklus detektálás
        if visited_nodes.contains(&from_node) {
            break;
        }
        visited_nodes.insert(from_node);
        
        // Következő él
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

fn golden_section_search<F>(
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
    mut f: F,
) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    // Ellenőrizzük, hogy a és b értékek validak-e
    let fa = f(a);
    let fb = f(b);
    
    // Ha mindkét végpont hibás, visszatérünk a középértékkel
    if fa == f64::MIN && fb == f64::MIN {
        println!("Both endpoints failed, trying midpoint");
        let mid = (a + b) / 2.0;
        let fmid = f(mid);
        if fmid != f64::MIN {
            return (mid, fmid);
        } else {
            println!("All evaluation points failed, returning zero profit");
            return (a, 0.0);
        }
    }
    
    // Ha csak az egyik végpont hibás, szűkítsük a tartományt
    if fa == f64::MIN {
        println!("Lower bound failed, adjusting range");
        a = a + (b - a) * 0.1; // 10%-kal beljebb
        let fa_new = f(a);
        if fa_new == f64::MIN {
            // Ha még mindig hibás, próbáljuk a felső tartományt
            return golden_section_search(b * 0.5, b, tol, max_iter, f);
        }
    }
    
    if fb == f64::MIN {
        println!("Upper bound failed, adjusting range");
        b = b - (b - a) * 0.1; // 10%-kal beljebb
        let fb_new = f(b);
        if fb_new == f64::MIN {
            // Ha még mindig hibás, próbáljuk az alsó tartományt
            return golden_section_search(a, a + (b - a) * 0.5, tol, max_iter, f);
        }
    }

    let gr = (5f64.sqrt() + 1.0) / 2.0;
    let mut c = b - (b - a) / gr;
    let mut d = a + (b - a) / gr;
    let mut fc = f(c);
    let mut fd = f(d);

    // Ha a belső pontok is hibásak, próbáljunk kisebb tartományt
    if fc == f64::MIN && fd == f64::MIN {
        println!("Inner points failed, trying smaller range");
        let mid = (a + b) / 2.0;
        let quarter = (b - a) / 4.0;
        return golden_section_search(mid - quarter, mid + quarter, tol, max_iter / 2, f);
    }

    for i in 0..max_iter {
        if (b - a).abs() < tol {
            break;
        }
        
        if fc == f64::MIN {
            // Ha c pont hibás, újra számoljuk
            c = b - (b - a) / gr;
            fc = f(c);
            if fc == f64::MIN {
                println!("Point c failed at iteration {}, stopping", i);
                break;
            }
        }
        
        if fd == f64::MIN {
            // Ha d pont hibás, újra számoljuk
            d = a + (b - a) / gr;
            fd = f(d);
            if fd == f64::MIN {
                println!("Point d failed at iteration {}, stopping", i);
                break;
            }
        }
        
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / gr;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / gr;
            fd = f(d);
        }
    }
    
    // Visszatérünk a jobb értékkel
    if fc != f64::MIN && fd != f64::MIN {
        if fc > fd {
            (c, fc)
        } else {
            (d, fd)
        }
    } else if fc != f64::MIN {
        (c, fc)
    } else if fd != f64::MIN {
        (d, fd)
    } else {
        println!("All points failed, returning midpoint with zero profit");
        ((a + b) / 2.0, 0.0)
    }
}

fn optimize_cycle_gss(
    cycle: &Vec<EdgeIndex>,
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    source: NodeIndex,
    amount_in_min: f64,
    amount_in_max: f64,
    gss_tolerance: f64,
    gss_max_iter: usize,
    gas_price: f64
) -> (f64, f64, f64) { // (best_x, best_profit, total_gas)
    if cycle.is_empty() {
        println!("Cycle is empty, returning zero profit");
        return (amount_in_min, 0.0, 0.0);
    }
    
    let (start_node, _) = match graph.edge_endpoints(cycle[0]) {
        Some(endpoints) => endpoints,
        None => {
            println!("Invalid edge in cycle, returning zero profit");
            return (amount_in_min, 0.0, 0.0);
        }
    };
    
    if start_node != source {
        println!("Cycle does not start at source node, returning zero profit");
        return (amount_in_min, 0.0, 0.0);
    }
    
    let from_node_data = graph.node_weight(source).unwrap();
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);

    // Adaptív tartomány keresés - ha a teljes tartomány hibás, szűkítsük
    let mut current_min = amount_in_min;
    let mut current_max = amount_in_max;
    
    // Először teszteljük néhány pontot a tartományban
    let test_points = vec![
        current_min,
        current_min + (current_max - current_min) * 0.1,
        current_min + (current_max - current_min) * 0.5,
        current_min + (current_max - current_min) * 0.9,
        current_max
    ];
    
    let mut valid_points = Vec::new();
    
    for &test_point in &test_points {
        let amount_start = (test_point * multiplier).to_biguint().unwrap_or(BigUint::zero());
        match evaluate_cycle(cycle.clone(), amount_start, graph, source) {
            Ok((amount, total_gas)) => {
                let y = amount.to_f64().unwrap_or(0.0) / multiplier;
                let gas = total_gas.to_f64().unwrap_or(0.0) * gas_price * 1e-9;
                let profit = y - test_point - gas;
                valid_points.push((test_point, profit));
                println!("Test point {} -> profit: {}", test_point, profit);
            }
            Err(_e) => {
                println!("Test point {} failed: {}", test_point, _e);
            }
        }
    }
    
    if valid_points.is_empty() {
        println!("No valid points found in range [{}, {}], returning zero", current_min, current_max);
        return (current_min, 0.0, 0.0);
    }
    
    // Szűkítsük a tartományt a valid pontok alapján
    let min_valid = valid_points.iter().map(|(x, _)| *x).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_valid = valid_points.iter().map(|(x, _)| *x).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    current_min = min_valid;
    current_max = max_valid;
    
    println!("Adjusted range for GSS: [{}, {}] with {} valid points", current_min, current_max, valid_points.len());

    let (best_x, best_profit) = golden_section_search(
        current_min,
        current_max,
        gss_tolerance,
        gss_max_iter,
        |x| {
            let amount_start = (x * multiplier).to_biguint().unwrap_or(BigUint::zero());
            match evaluate_cycle(cycle.clone(), amount_start, graph, source) {
                Ok((amount, total_gas)) => {
                    let y = amount.to_f64().unwrap_or(0.0) / multiplier;
                    let gas = total_gas.to_f64().unwrap_or(0.0) * gas_price * 1e-9;
                    let profit = y - x - gas;
                    println!(
                        "Evaluating cycle with amount_start: {} -> {} (-gas {}) profit: {}",
                        x, y, gas, profit
                    );
                    profit
                }
                Err(_e) => {
                    println!("Cycle evaluation failed for amount {}: {}", x, _e);
                    f64::MIN // Hibás értékek esetén MIN értéket visszaadunk
                }
            }
        },
    );
    
    // Kiszámítjuk a gas költséget a legjobb x értékre
    let best_gas = {
        let amount_start = (best_x * multiplier).to_biguint().unwrap_or(BigUint::zero());
        match evaluate_cycle(cycle.clone(), amount_start, graph, source) {
            Ok((_amount, total_gas)) => {
                total_gas.to_f64().unwrap_or(0.0) * gas_price * 1e-9
            }
            Err(_) => 0.0
        }
    };
    
    (best_x, best_profit, best_gas)
}