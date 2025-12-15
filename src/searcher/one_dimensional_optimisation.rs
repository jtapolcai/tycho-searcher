// one_dimensional_optimisation.rs
// Golden-section search (1D optimization) for single-cycle profit maximization

use std::time::{Duration, Instant};
use num_bigint::BigUint;
use num_bigint::ToBigUint;
use num_traits::{ToPrimitive, Zero};
use petgraph::{
    graph::{EdgeIndex, Graph, NodeIndex},
    Directed,
};
use std::cell::RefCell;

use crate::searcher::{EdgeData, NodeData, Statistics};
use crate::log_arb_info;

/// Find bottleneck edges that would benefit from sharing load (having a partner cycle)
/// For each edge, checks if doubling the swap amount causes significant slippage
/// and if halving gas_price would be a worthwhile trade-off
pub fn find_bottleneck_links(
    cycle: &[EdgeIndex],
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in: f64,
    gas_price: f64,
    quoter_time: &mut Duration,
) -> Option<Vec<EdgeIndex>> {
    log_arb_info!("[BOTTLENECK] Analyzing edges for bottleneck candidates at amount_in={:.6}", amount_in);
    
    let from_node_data = graph.node_weight(source).unwrap();
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);
    
    let mut shareable_edge: Vec<EdgeIndex> = Vec::new(); // Unused variable, can be removed
    
    // First, evaluate the cycle at baseline
    let (_, _, _, baseline_gas, baseline_profit, baseline_amounts) = 
        evaluate_cycle_profit(cycle, graph, stats, source, amount_in, gas_price, quoter_time);
    
    log_arb_info!("[BOTTLENECK] Baseline: profit={:.6}, gas={:.0}", baseline_profit, baseline_gas);
    
    // For each edge, check if doubling the input causes significant slippage
    for (i, &edge_idx) in cycle.iter().enumerate() {
        let (from, to) = match graph.edge_endpoints(edge_idx) {
            Some((f, t)) => (f, t),
            None => continue,
        };
        
        let token_in = match graph.node_weight(from) {
            Some(n) => &n.token,
            None => continue,
        };
        let token_out = match graph.node_weight(to) {
            Some(n) => &n.token,
            None => continue,
        };
        let edge = match graph.edge_weight(edge_idx) {
            Some(e) => e,
            None => continue,
        };
        
        // Get the input amount at this edge from the baseline evaluation
        let baseline_input_at_edge = if i == 0 {
            amount_in * multiplier
        } else {
            baseline_amounts.get(i - 1).copied().unwrap_or(0.0) * 10f64.powi(token_in.decimals as i32)
        };
        
        // Query baseline output at this edge
        let baseline_input_biguint = (baseline_input_at_edge).to_biguint().unwrap_or_default();
        let t0 = Instant::now();
        let baseline_result = edge.borrow().quoter_amount_out(token_in, token_out, &baseline_input_biguint, stats);
        *quoter_time += t0.elapsed();
        
        if let Some(baseline_data) = baseline_result {
            let baseline_output_f = baseline_data.amount_out.to_f64().unwrap_or(0.0) / 10f64.powi(token_out.decimals as i32);
            let baseline_gas_units = baseline_data.gas.to_f64().unwrap_or(0.0);
            
            // Try doubling the input to this edge
            let doubled_input_biguint = baseline_input_biguint * 2u32;
            let t0 = Instant::now();
            let doubled_result = edge.borrow().quoter_amount_out(token_in, token_out, &doubled_input_biguint, stats);
            *quoter_time += t0.elapsed();
            
            if let Some(doubled_data) = doubled_result {
                let doubled_output_f = doubled_data.amount_out.to_f64().unwrap_or(0.0) / 10f64.powi(token_out.decimals as i32);
                
                // Calculate slippage: how much worse is the rate when we double input
                let expected_doubled_output = baseline_output_f * 2.0;
                let slippage = expected_doubled_output - doubled_output_f;
                let slippage_pct = if expected_doubled_output > 0.0 { slippage / expected_doubled_output * 100.0 } else { 0.0 };
                
                // Calculate gas saving when running 2 cycles in parallel sharing this edge
                // When 2 cycles share an edge: gas paid once instead of twice
                // Gas saving per cycle: edge_gas / 2
                // Gas saving in start token: (edge_gas / 2) * gas_price
                let edge_gas_saving_in_start_token = (baseline_gas_units / 2.0) * gas_price * 1e-9;
                
                // Convert gas saving to token_out using exchange rate from baseline_amounts
                // Exchange rate: baseline_amounts[i] token_out per amount_in start_token
                // Both amounts are normalized (divided by their respective decimals)
                let token_out_amount = baseline_amounts.get(i).copied().unwrap_or(0.0);
                let exchange_rate = if amount_in > 0.0 { token_out_amount / amount_in } else { 0.0 };
                let gas_saving_in_token_out = edge_gas_saving_in_start_token * exchange_rate;
                
                log_arb_info!("[BOTTLENECK] Edge {} ({} → {}): slippage={:.6} ({:.2}%), edge_gas={:.0}, gas_saving={:.6} {} (rate={:.6})",
                    edge_idx.index(), token_in.symbol, token_out.symbol, slippage, slippage_pct, 
                    baseline_gas_units, gas_saving_in_token_out, token_out.symbol, exchange_rate);
                
                // If slippage loss is less than gas saving, then sharing is beneficial
                if slippage < gas_saving_in_token_out {
                    log_arb_info!("[BOTTLENECK]   -> SHAREABLE: slippage {:.6} < gas_saving {:.6} {} - worth sharing!", 
                        slippage, gas_saving_in_token_out, token_out.symbol);
                   shareable_edge.push(edge_idx);
                }
            }
        }
    }
    
    log_arb_info!("[BOTTLENECK] Found {} bottleneck edges",shareable_edge.len());
    Some(shareable_edge)
}

/// Evaluates the profit of a single cycle at a given input amount
/// Returns: (is_profitable, amount_in, amount_out, total_gas, net_profit, per_node_amounts)
/// per_node_amounts contains the output amount after each edge in the cycle
pub fn evaluate_cycle_profit(
    cycle: &[EdgeIndex],
    graph: &mut Graph<NodeData, RefCell<EdgeData>, Directed>,
    stats: &mut Statistics,
    source: NodeIndex,
    amount_in: f64,
    gas_price_in_start_token: f64,
    quoter_time: &mut Duration,
) -> (bool, f64, f64, f64, f64, Vec<f64>) {
    if cycle.is_empty() {
        return (false, 0.0, 0.0, 0.0, 0.0, Vec::new());
    }

    let from_node_data = graph.node_weight(source).unwrap();
    let decimals = from_node_data.token.decimals as u32;
    let multiplier = 10f64.powi(decimals as i32);

    let mut current_amount = (amount_in * multiplier).to_biguint().unwrap_or_default();
    let mut total_gas = BigUint::zero();
    let mut per_node_amounts = Vec::new();  // Track output at each step

    let mut current_node = source;
    for &edge_idx in cycle {
        let (from, to) = graph.edge_endpoints(edge_idx).unwrap();
        if from != current_node { return (false, 0.0, 0.0, 0.0, 0.0, Vec::new()); }
        let token_in = &graph.node_weight(from).unwrap().token;
        let token_out = &graph.node_weight(to).unwrap().token;
        let edge = graph.edge_weight(edge_idx).unwrap();
        let t0 = Instant::now();
        let result = edge.borrow().quoter_amount_out(token_in, token_out, &current_amount, stats);
        *quoter_time += t0.elapsed();
        if let Some(price_data) = result {
            current_amount = price_data.amount_out.clone();
            total_gas += price_data.gas;
            let amount_out_f = current_amount.to_f64().unwrap_or(0.0) / 10f64.powi(token_out.decimals as i32);
            per_node_amounts.push(amount_out_f);
            current_node = to;
        } else {
            return (false, 0.0, 0.0, 0.0, 0.0, Vec::new());
        }
    }

    let amount_out = current_amount.to_f64().unwrap_or(0.0) / multiplier;
    let profit = amount_out - amount_in;
    let total_gas_f64 = total_gas.to_f64().unwrap_or(0.0);
    let gas_cost_in_start = total_gas_f64 * gas_price_in_start_token * 1e-9;
    let net_profit = profit - gas_cost_in_start;

    log_arb_info!("[GSS-EVAL] amount_in={:.6}, amount_out={:.6}, gas={:.0} ({:.9} ETH), profit={}, net_profit={:.6}",
        amount_in, amount_out, total_gas_f64, gas_cost_in_start, profit, net_profit);
    
    (net_profit > 0.0, amount_in, amount_out, total_gas_f64, net_profit, per_node_amounts)
}

/// Golden-section search for single-cycle optimization including gas costs
/// Returns: (amount_in, amount_out, total_gas, net_profit) if profitable, None otherwise
pub fn golden_section_search_with_gas(
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
    // Step 0: Find the minimum amount_in such that the cycle at least covers the gas cost
    let mut min_in = amount_in_min.max(1e-12); // do not be zero
    let mut last_min_in;
    let tolerance = 1e-9;
    let max_iter = 20;
    let mut b;
    let mut last_profit= 0.0;

    if amount_in_min < amount_in_max {
        for i in 0..max_iter {
            let (_profitable, ain, aout, total_gas, profit, _) =
                evaluate_cycle_profit(cycle, graph, stats, source, min_in, gas_price_in_start_node_token, quoter_time);
            if last_profit > 0.0 && aout - ain < last_profit {
                log_arb_info!("[GSS-MIN-IN] Profit decreased from {:.6} to {:.6}, stopping search for min_in", last_profit, profit);
                if profit<0.0 {
                    return Some((ain, aout, total_gas, profit))
                }
                break;
            }
            last_profit = aout - ain;

            if i == 0 && aout < 0.0 {
                // If even a small input produces no output, abort early
                log_arb_info!("[GSS-MIN-IN] Cycle produces no output even for small input, aborting");
                return None;
            }
            let gas_cost_eth = total_gas * gas_price_in_start_node_token * 1e-9;

            if aout < ain {
                log_arb_info!("[GSS-MIN-IN] Cycle not profitable at min_in={:.6}, aout={:.6} < ain={:.6}, aborting", min_in, aout, ain);
                if i == 0 {
                    return None;
                }
                break;
            }

            // How much input is needed to have at least total_cost as profit?
            // profit = (aout - min_in) - total_cost >= 0
            // aout - min_in = total_cost
            // aout = min_in + total_cost
            // If amount_out < min_in + total_cost, then min_in needs to be increased
            if aout < min_in + gas_cost_eth {
                // Linearly estimate how much to increase
                let diff = (min_in + gas_cost_eth) - aout;
                if diff < amount_in_min*0.001 {
                    break;
                }
                last_min_in = min_in;
                min_in += diff; //.max(min_in * 0.1); // increase by at least 10% for faster convergence
            } else {
                break;
            }
            if (min_in - last_min_in).abs() < tolerance {
                break;
            }
        }

        // If min_in became too large, then there is no meaningful solution
        if min_in > amount_in_max {
            return None;
        }

        // Binary search: double amount_in until profit starts decreasing
        // This finds the right boundary for the golden section search
        b = min_in;
        //let min_in_profit = evaluate_cycle_profit(cycle, graph, stats, source, min_in, gas_price_in_start_node_token, quoter_time).4;
        //let mut last_profit = min_in_profit;
        
        log_arb_info!("[GSS-DOUBLING] Starting binary search from min_in={:.6}, initial_profit={:.6}", min_in, last_profit);
        
        loop {
            let candidate = b * 2.0;
            if candidate > 2.0 * amount_in_max {
                log_arb_info!("[GSS-DOUBLING] Candidate {:.6} exceeds amount_in_max {:.6}, stopping", candidate, amount_in_max);
                break;
            }
            let (_profitable, ain, aout, _total_gas, _profit, _) = evaluate_cycle_profit(cycle, graph, stats, source, candidate, gas_price_in_start_node_token, quoter_time);
            let candidate_profit = aout - ain ;
            log_arb_info!("[GSS-DOUBLING] candidate={:.6}, profit={:.6}", candidate, candidate_profit);
            
            // Stop if profit becomes non-profitable (no point in increasing further)
            if candidate_profit <= 0.0 {
                log_arb_info!("[GSS-DOUBLING] Profit not profitable ({:.6}), stopping. Upper bound b={:.6}", candidate_profit, b);
                break;
            }
            
            // Stop if profit decreased
            if candidate_profit < last_profit {
                log_arb_info!("[GSS-DOUBLING] Profit decreased from {:.6} to {:.6}, stopping. Upper bound b={:.6}", last_profit, candidate_profit, b);
                break;
            }
            b = candidate;
            last_profit = candidate_profit;
        }
    } else {
        // weonly identify the bottleneck links
        b = amount_in_min*1.01;
    }

    // Golden section search starts from the found min_in
    let mut a = min_in;
    let gr = (5.0f64.sqrt() - 1.0) / 2.0; // Golden ratio conjugate

    let mut x1 = a + (1.0 - gr) * (b - a);
    let mut x2 = a + gr * (b - a);

    let (_, _, _, _, f1, amounts_x1) = evaluate_cycle_profit(cycle, graph, stats, source, x1, gas_price_in_start_node_token, quoter_time);
    let (_, _, _, _, f2, amounts_x2) = evaluate_cycle_profit(cycle, graph, stats, source, x2, gas_price_in_start_node_token, quoter_time);
    let mut last_f1 = f1;
    let mut last_f2 = f2;
    let mut last_amounts_x1 = amounts_x1.clone();
    let mut last_amounts_x2 = amounts_x2.clone();

    for _ in 0..gss_max_iter {
        if last_f1 > last_f2 {
            b = x2;
            x2 = x1;
            last_f2 = last_f1;
            last_amounts_x2 = last_amounts_x1.clone();
            x1 = a + (1.0 - gr) * (b - a);
            let (_, _, _, _, f, amounts) = evaluate_cycle_profit(cycle, graph, stats, source, x1, gas_price_in_start_node_token, quoter_time);
            last_f1 = f;
            last_amounts_x1 = amounts;
        } else {
            a = x1;
            x1 = x2;
            last_f1 = last_f2;
            last_amounts_x1 = last_amounts_x2.clone();
            x2 = a + gr * (b - a);
            let (_, _, _, _, f, amounts) = evaluate_cycle_profit(cycle, graph, stats, source, x2, gas_price_in_start_node_token, quoter_time);
            last_f2 = f;
            last_amounts_x2 = amounts;
        }

        if (b - a).abs() < gss_tolerance {
            break;
        }
    }

    let best_amount = (a + b) / 2.0;
    let (is_profitable, final_amount_in, final_amount_out, final_gas, final_profit, _) =
         evaluate_cycle_profit(cycle, graph, stats, source, best_amount, gas_price_in_start_node_token, quoter_time);

    if is_profitable {
        log_arb_info!("[GSS-RESULT] Converged: amount_in={:.6}, amount_out={:.6}, gas={:.0}, net_profit={:.6}",
            final_amount_in, final_amount_out, final_gas, final_profit);
        Some((final_amount_in, final_amount_out, final_gas, final_profit))
    } else {
        log_arb_info!("[GSS-RESULT] No profitable solution found");
        None
    }
}
