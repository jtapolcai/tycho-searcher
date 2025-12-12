
use serde::Deserialize;
use std::sync::OnceLock;
#[derive(Debug, Clone, Deserialize)]
pub struct QuoterLogEntry {
    pub input: QuoterLogInput,
    pub output: Option<QuoterLogOutput>,
    pub error: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuoterLogInput {
    pub token_in: String,
    pub token_out: String,
    pub amount_in: String,
    pub pool_address: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuoterLogOutput {
    pub amount_out: String,
    pub gas: String,
}

static QUOTER_CACHE: OnceLock<std::collections::HashMap<(String, String, String), (BigUint, BigUint)>> = OnceLock::new();
static PLAYBACK_MODE: OnceLock<bool> = OnceLock::new();

pub fn load_quoter_cache_once() {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    if QUOTER_CACHE.get().is_some() { return; }
    let playback = std::env::args().any(|a| a == "--playback");
    PLAYBACK_MODE.set(playback).ok();
    let mut cache = std::collections::HashMap::new();
    if playback {
        if let Ok(file) = File::open("quoter_log.json") {
            let reader = BufReader::new(file);
            let mut loaded = 0;
            for line in reader.lines().flatten() {
                if let Ok(entry) = serde_json::from_str::<QuoterLogEntry>(&line) {
                    if entry.status == "ok" {
                        if let (Some(output), input) = (entry.output, entry.input) {
                            let key = (input.token_in.clone(), input.token_out.clone(), input.amount_in.clone());
                            let amount_out = BigUint::parse_bytes(output.amount_out.as_bytes(), 10).unwrap_or(BigUint::zero());
                            let gas = BigUint::parse_bytes(output.gas.as_bytes(), 10).unwrap_or(BigUint::zero());
                            cache.insert(key, (amount_out, gas));
                            loaded += 1;
                        }
                    }
                }
            }
            println!("[PLAYBACK] Loaded {} cache entries from quoter_log.json", loaded);
        }
    }
    QUOTER_CACHE.set(cache).ok();
}

fn log_quoter_json_zero_input(token_in: &str, token_out: &str, amount_in: &BigUint, pool_address: &str) -> String {
    format!(
        "{{\"input\":{{\"token_in\":\"{}\",\"token_out\":\"{}\",\"amount_in\":\"{}\",\"pool_address\":\"{}\"}},\"output\":{{\"amount_out\":\"0\",\"gas\":\"0\"}},\"status\":\"zero_input\"}}\n",
        token_in, token_out, amount_in, pool_address
    )
}

fn log_quoter_json_ok(token_in: &str, token_out: &str, amount_in: &BigUint, pool_address: &str, amount_out: &BigUint, gas: &BigUint) -> String {
    format!(
        "{{\"input\":{{\"token_in\":\"{}\",\"token_out\":\"{}\",\"amount_in\":\"{}\",\"pool_address\":\"{}\"}},\"output\":{{\"amount_out\":\"{}\",\"gas\":\"{}\"}},\"status\":\"ok\"}}\n",
        token_in, token_out, amount_in, pool_address, amount_out, gas
    )
}

fn log_quoter_json_error(token_in: &str, token_out: &str, amount_in: &BigUint, pool_address: &str, error: &str) -> String {
    format!(
        "{{\"input\":{{\"token_in\":\"{}\",\"token_out\":\"{}\",\"amount_in\":\"{}\",\"pool_address\":\"{}\"}},\"error\":\"{}\",\"status\":\"error\"}}\n",
        token_in, token_out, amount_in, pool_address, error
    )
}
fn log_quoter_json(log: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;
    let mut file = OpenOptions::new().create(true).append(true).open("quoter_log.json").unwrap();
    file.write_all(log.as_bytes()).unwrap();
}

// price_quoter.rs
use crate::log_quoter_info;

use std::{
    collections::BTreeMap,
    // collections::{HashMap, HashSet},
    //sync::Arc,
};

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};

use ordered_float::OrderedFloat;

use tycho_common::{
    models::{token::Token},
};

use hex::ToHex;

#[derive(Debug, Clone, PartialEq)]
pub struct PriceDataRaw {
    pub amount_out: BigUint,
    pub gas: BigUint
}

impl PriceDataRaw {
    pub fn is_zero(&self) -> bool {
        self.amount_out.is_zero()
    }
}

//pub mod graph_types; // Contains NodeData, EdgeData, GraphError, etc.

// Use types from graph_types module
//use graph_types::{NodeData, EdgeData, GraphError, graph_to_json, Statistics, ArbitrageExport, ArbitragePool};

use crate::searcher::graph_types::{NodeData, EdgeData, PriceData, Statistics};



/// Calculates the interpolated y-value for a given new_x,
/// if new_x lies on a straight line segment defined by its neighbors.
///
/// Returns `Some(y_value)` if interpolation is possible and the point is collinear
/// with its neighbors.
/// Returns `None` otherwise (e.g., not enough points, new_x outside range,
/// or existing point not collinear with its neighbors).
pub fn get_interpolated_y(
    points: &BTreeMap<OrderedFloat<f64>, PriceData>,
    new_x: f64,
) -> Option<f64> {
    let ordered_new_x = OrderedFloat(new_x);

    // Need at least two points to define a line segment.
    if points.len() < 2 {
        return None;
    }

    // Find the point just less than or equal to new_x
    let lower_bound = points.range(..=ordered_new_x).next_back();

    // Find the point just greater than or equal to new_x
    let upper_bound = points.range(ordered_new_x..).next();

    match (lower_bound, upper_bound) {
        (Some((x1, y1)), Some((x2, y2))) => {
            if *x1 == ordered_new_x || *x2 == ordered_new_x {
                // `new_x` is an existing point. We need to check if this existing point
                // is collinear with its *actual* distinct neighbors.
                // We'll return its `y` value if it is.

                // Determine the actual point (x_target, y_target)
                let (x_target, y_target) = if *x1 == ordered_new_x { (x1, y1) } else { (x2, y2) };

                // Find the distinct point before ordered_new_x
                let prev_point_iter = points.range(..ordered_new_x).next_back();
                // Find the distinct point after ordered_new_x
                let next_point_iter = points.range(ordered_new_x..).skip(1).next();

                if let (Some((px, py)), Some((nx, ny))) = (prev_point_iter, next_point_iter) {
                    // Check collinearity: (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1)
                    // Here, (px, py) is (x1, y1), (x_target, y_target) is (x2, y2), (nx, ny) is (x3, y3)

                    // Ensure no division by zero for slope calculation in theoretical interpolation
                    let px_f64 = px.into_inner();
                    let nx_f64 = nx.into_inner();
                    let x_target_f64 = x_target.into_inner();

                    let interpolated_y_from_neighbors = if (nx_f64 - px_f64).abs() < f64::EPSILON {
                        // This implies px and nx have essentially the same x-coordinate.
                        // For points to be collinear on a "vertical" line, they must also have the same y.
                        // If they are not the same, it's not a single line segment for interpolation.
                        if (py.amount_out - ny.amount_out).abs() < f64::EPSILON { py.amount_out } else { f64::NAN }
                    } else {
                        py.amount_out + (ny.amount_out - py.amount_out) * (x_target_f64 - px_f64) / (nx_f64 - px_f64)
                    };

                    // Check if the actual y-value is close enough to the interpolated value
                    if (y_target.amount_out - interpolated_y_from_neighbors).abs() < f64::EPSILON {
                        Some(y_target.amount_out) // Return the existing y-value as it's on a straight segment
                    } else {
                        None // Existing point is not collinear with its neighbors
                    }
                } else {
                    // Not enough distinct neighbors to check collinearity for an existing point.
                    None
                }
            } else {
                // `new_x` is strictly between `x1` and `x2`, and `x1` != `x2`.
                // This is the classic linear interpolation case.
                // The points (x1, y1) and (x2, y2) form a straight line segment.
                // We can always interpolate `y` for `new_x` on this segment.
                let x1_f64 = x1.into_inner();
                let x2_f64 = x2.into_inner();
                let new_x_f64 = ordered_new_x.into_inner();

                let x_diff = x2_f64 - x1_f64;
                if x_diff.abs() < f64::EPSILON {
                    // This implies x1 and x2 have essentially the same x-coordinate.
                    // This is not a valid segment for linear interpolation in a function x->y.
                    None
                } else {
                    let slope = (y2.amount_out - y1.amount_out) / x_diff;
                    let interpolated_y = y1.amount_out + slope * (new_x_f64 - x1_f64);
                    Some(interpolated_y)
                }
            }
        }
        _ => {
            // `new_x` is outside the range of existing points,
            // or there's only one bounding point.
            None
        }
        // (None, None) => todo!(),
        // (None, Some(_)) => todo!(),
        // (Some(_), None) => todo!(),
    }
}

impl EdgeData {
    // These methods are kept with EdgeData as they directly operate on its fields.
    pub fn address(&self) -> String {
        self.pool.id.encode_hex::<String>()
    }



    pub fn quoter_amount_out(
        &self,
        token_in: &Token,
        token_out: &Token,
        amount_in: &BigUint,
        stats: &mut Statistics
    ) -> Option<PriceDataRaw> {
        // Playback mode: use quoter_log.json cache ONLY, do not fallback to live quoter
        let playback = PLAYBACK_MODE.get().copied().unwrap_or(false);
        if playback {
            // Log all playback queries for later diff
            //use std::fs::OpenOptions;
            //use std::io::Write;
            //let mut file = OpenOptions::new().create(true).append(true).open("playback_queries.log").unwrap();
            //let log = format!("{}|{}|{}\n", token_in.symbol, token_out.symbol, amount_in.to_string());
            //file.write_all(log.as_bytes()).unwrap();

            let key = (token_in.symbol.clone(), token_out.symbol.clone(), amount_in.to_string());
            match QUOTER_CACHE.get().and_then(|c| c.get(&key)) {
                Some((amount_out, gas)) => {
                    log_quoter_info!(
                        "[PLAYBACK] Cached quoter: {} → {} @ {}: amount_in={} amount_out={} gas={}",
                        token_in.symbol, token_out.symbol, self.address(), amount_in, amount_out, gas
                    );
                    return Some(PriceDataRaw { amount_out: amount_out.clone(), gas: gas.clone() });
                }
                None => {
                    log_quoter_info!(
                        "[PLAYBACK] ERROR: No cached quoter for {} → {}: amount_in={}",
                        token_in.symbol, token_out.symbol,  amount_in // self.address(),
                    );
                    return None;
                }
            }
        }
        // if not playback
        use std::fs::OpenOptions;
        use std::io::Write;
        // Debug mód: ha a programot --debug kapcsolóval futtatják
        let debug_mode = std::env::args().any(|a| a == "--debug");
        if *amount_in == BigUint::zero() {
            if debug_mode {
                let log = log_quoter_json_zero_input(&token_in.symbol, &token_out.symbol, amount_in, &self.address());
                log_quoter_json(&log);
            }
            return Some(PriceDataRaw { amount_out: BigUint::zero(), gas: BigUint::zero() });
        }
        let amount_out = self.state.get_amount_out(amount_in.clone(), token_in, token_out);
        match amount_out {
            Ok(out) => {
                stats.quoter += 1;
                let amount_out_f64 = out.amount.to_f64().unwrap_or(0.0) / 10f64.powi(token_out.decimals as i32);
                let amount_in_f64 = amount_in.to_f64().unwrap_or(0.0) / 10f64.powi(token_in.decimals as i32);
                let price = if amount_out_f64 != 0.0 { amount_in_f64 / amount_out_f64 } else { 0.0 };
                let gas_f64 = out.gas.to_f64().unwrap_or(0.0) / 10f64.powi(9);
                log_quoter_info!(
                    "Computed amount_out for {} → {} @ {}: {} (price: {}, gas: {})",
                    token_in.symbol, token_out.symbol, self.address(),
                    amount_out_f64, price, gas_f64
                 );
                if debug_mode {
                    let log = log_quoter_json_ok(&token_in.symbol, &token_out.symbol, amount_in, &self.address(), &out.amount, &out.gas);
                    log_quoter_json(&log);
                }
                Some(PriceDataRaw { amount_out: out.amount, gas: out.gas } )
            }
            Err(e) => {
                self.handle_compute_error(e.to_string(), token_in, token_out, stats);
                log_quoter_info!("No amount out for quoter_amount_out({})", amount_in);
                if debug_mode {
                    let log = log_quoter_json_error(&token_in.symbol, &token_out.symbol, amount_in, &self.address(), &e.to_string());
                    log_quoter_json(&log);
                }
                None
            }
        }
    }  


    pub fn compute_weight(
        &mut self,
        amount_in: &BigUint,
        stats: &mut Statistics,
        source: &NodeData,
        target: &NodeData,
    ) -> Option<(BigUint, f64)> {
        if *amount_in == BigUint::zero() {
            return Some((BigUint::zero(), 0.0));
        }

        let amount_in_f64 = amount_in.to_f64().unwrap_or(0.0) / 10f64.powi(source.token.decimals as i32);
        let point_key = OrderedFloat(amount_in_f64);
        if self.points.contains_key(&point_key) {
            return Some((
                BigUint::from((self.points[&point_key].output_with_gas(target.price) * 10f64.powi(target.token.decimals as i32)) as u128),
                0.0,
            ));
        }
        if let Some(y) = get_interpolated_y(&self.points, amount_in_f64) {
                //println!("Interpolated y-value: {}", y);
                stats.price_is_interpolated+=1;
                return Some((BigUint::from((y * 10f64.powi(target.token.decimals as i32)) as u128), 0.0));
        } 
        
        let amount_out = self.state.get_amount_out(amount_in.clone(), &*source.token, &*target.token);
        match amount_out {
            Ok(out) => {
                stats.quoter += 1;
                let amount_out_f64 = out.amount.to_f64().unwrap_or(0.0) / 10f64.powi(target.token.decimals as i32);
                let price = if amount_out_f64 != 0.0 { amount_in_f64 / amount_out_f64 } else { 0.0 };
                let gas_f64 = out.gas.to_f64().unwrap_or(0.0) / 10f64.powi(9);
                // let amount_out_without_gas = amount_out_f64 * 0.0f64.max((swap_amount - gas_f64) / swap_amount);
                // let amount_out_without_gas_big_uint = BigUint::from(
                //     (amount_out_without_gas * 10f64.powi(self.target.token.decimals as i32)) as u128
                // );
                // log_quoter_info!(
                //     "Computed amount_out for {} → {} @ {}: {} → {} (price: {}, gas: {})",
                //     self.source.token.symbol, self.target.token.symbol, self.address(),
                //     amount_in_f64, amount_out_without_gas, price, gas_f64
                // );
                log_quoter_info!(
                    "Computed amount_out for {} → {} @ {}: {} → {} (price: {} gas: {})",
                    source.token.symbol, target.token.symbol, self.address(),
                    amount_in_f64, amount_out_f64, price, gas_f64
                );
                self.points.insert(
                    OrderedFloat(amount_in_f64),
                    PriceData {
                        amount_out: amount_out_f64,
                        gas: gas_f64,
                    },
                );
                Some((out.amount, gas_f64))
            }
            Err(_e) => {
                log_quoter_info!(
                    "No amount out for compute_weight({}) ",amount_in);
                None
            }
        }
    }

    pub fn compute_weight_minus_gas(
        &mut self,
        amount_in: &BigUint,
        stats: &mut Statistics,
        _swap_amount: f64,
        source: &NodeData,
        target: &NodeData,
    ) -> Option<f64> {
        match self.compute_weight(amount_in, stats, source, target) {
            Some((amount_out, gas_f64)) => {
                let amount_out_f64 = amount_out.to_f64().unwrap_or(0.0) / 10f64.powi(target.token.decimals as i32);
                let amount_out_without_gas = amount_out_f64  - gas_f64*target.price / 10f64.powi(6);
                Some(amount_out_without_gas)
            }
            None => {
                None
            }
        }
    }


    fn handle_compute_error(&self, err: impl std::fmt::Display, token_in: &Token, token_out: &Token, stats: &mut Statistics) {
        let error_text = err.to_string();
        if error_text.contains("No liquidity") {
            stats.quoter_failed_no_liquidity += 1;
        } else if error_text.contains("Simulation reverted") {
            stats.quoter_failed_reverted += 1;
        } else if error_text.contains("Ticks exceeded") {
            stats.quoter_failed_tick_exceeded += 1;
        } else if error_text.contains("Sell amount exceeds limit") {
            stats.quoter_failed_sell_amount_exceeds_limit += 1;
        } else {
            log_quoter_info!(
                "Warning! Failed to compute amount_out for {} → {} @ {}",
                token_in.symbol, token_out.symbol, self.address()
            );
            stats.quoter_failed += 1;
        }
    }

}
