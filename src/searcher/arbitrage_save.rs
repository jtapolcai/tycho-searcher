//use crate::log_arb_info;

// STD
use std::fs::File;
use std::sync::Arc;
use std::cell::RefCell;

use num_traits::{Zero, ToPrimitive}; 

// Regex
use regex::Regex;

// Serde
use serde::{Serialize, ser::SerializeSeq};

// Petgraph
use petgraph::{
    graph::{EdgeIndex, Graph},
    Directed,
};

// Num
use num_bigint::BigUint;

// Collections
use std::collections::{HashMap};

// Filesystem and Path
use std::path::Path;
use std::fs;
use std::io::Write;

// Tycho
use tycho_common::{
    models::{token::Token},
    simulation::protocol_sim::ProtocolSim,
};

use tycho_simulation::{
    protocol::models::ProtocolComponent,
};

#[derive(Serialize)]
pub struct ArbitrageExport {
    profit: f64,
    #[serde(serialize_with = "as_string")]
    amount_in: BigUint,
    #[serde(serialize_with = "vec_as_string")]
    amounts_out: Vec<BigUint>,
    tokens: Vec<Token>,
    pools: Vec<ProtocolComponent>,
}

fn as_string<S>(x: &BigUint, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_str(&x.to_string())
}

fn vec_as_string<S>(v: &Vec<BigUint>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let as_strings: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    let mut seq = s.serialize_seq(Some(as_strings.len()))?;
    for s in as_strings {
        seq.serialize_element(&s)?;
    }
    seq.end()
}

use crate::searcher::{EdgeData,NodeData};

/// Verifies an arbitrage opportunity in a given cycle and returns profit and related data if profitable.
pub fn verify_arbitrage_opportunity(
    cycle: Vec<EdgeIndex>,
    amount_: f64,
    start_token_name: String,
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    gas_price: f64,
) -> Option<(Vec<Token>, Vec<tycho_simulation::protocol::models::ProtocolComponent>, BigUint, Vec<BigUint>,f64,u64,f64)> {
    if cycle.is_empty() {
        // Cycle too short to be meaningful
        return None;
    }
    let mut tokens: Vec<Token> = Vec::new();
    let mut pools: Vec<tycho_simulation::protocol::models::ProtocolComponent> = Vec::new();
    let mut amounts_out: Vec<BigUint> = Vec::new();
    let mut pool_ids: HashMap<String, Arc<dyn ProtocolSim>> = HashMap::new();

    let (start, _) = graph.edge_endpoints(cycle[0]).unwrap();
    let start_token_node = graph.node_weight(start).unwrap();
    let start_token = (*start_token_node.token).clone();
    if start_token.symbol != start_token_name {
        return None;
    }
    tokens.push(start_token.clone());

    let amount_start =
        BigUint::from((amount_ * 10f64.powi(start_token.decimals as i32)) as u128);
    let mut amount = amount_start.clone();
    let mut total_gas = BigUint::zero();

    for &edge_index in &cycle {
        let (from, to) = graph.edge_endpoints(edge_index).unwrap();
        let token_in = graph.node_weight(from).unwrap();
        let token_out = graph.node_weight(to).unwrap();
        tokens.push((*token_out.token).clone());

        let edge_data = graph.edge_weight(edge_index).unwrap();
        let pool = edge_data.borrow().pool.clone();
        pools.push((*pool).clone());
        let _amount_in = amount.clone();
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
            let out_amount = new_amount.amount.clone();
            amount = out_amount.clone();
            amounts_out.push(out_amount.clone());
            total_gas += new_amount.gas.clone();
        } else {
            return None;
        }
    }

    if amount >= amount_start.clone() - total_gas.clone() {
        let profit = &amount - &amount_start;
        let profit_f64 = profit.to_f64().unwrap_or(0.0);
        let profit_without_gas = profit_f64 / 1e12 - &total_gas .to_f64().unwrap_or(0.0) * gas_price / 1000.0;
        Some((tokens, pools, amount_start.clone(), amounts_out, profit_f64 / 1e12, total_gas.to_u64().unwrap_or(0)/ 1000, profit_without_gas))
    } else {
        None
    }
}

/// Saves an arbitrage result to a JSON file in the specified directory.
pub fn save_arbitrage_result(
    amount_start: BigUint,
    tokens: Vec<Token>,
    pools: Vec<ProtocolComponent>,
    amounts_out: Vec<BigUint>,
    dir: &str,
    profit: f64,
) -> std::io::Result<()> {
    let result = ArbitrageExport {
        profit,
        amount_in: amount_start,
        amounts_out,
        tokens,
        pools,
    };

    let re = Regex::new(r"arbitrage_(\\d+)\\.json$").unwrap();

    let mut max_index = 0;
    if Path::new(dir).exists() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let file_name = entry.file_name().into_string().unwrap_or_default();
            if let Some(caps) = re.captures(&file_name) {
                if let Ok(num) = caps[1].parse::<u32>() {
                    max_index = max_index.max(num);
                }
            }
        }
    }

    let new_index = max_index + 1;
    let filename = format!("{}/arbitrage_{:03}.json", dir, new_index);

    let json = serde_json::to_string_pretty(&result).unwrap();
    match File::create(&filename) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(json.as_bytes()) {
                eprintln!("[WARNING] Failed to write to {}: {}", filename, e);
            } else {
                println!("Arbitrage opportunity is saved to {}", filename);
            }
        }
        Err(e) => {
            eprintln!("[WARNING] Failed to create {}: {}", filename, e);
        }
    }
    Ok(())
}
