// src/solver/graph_types.rs

// STD
use std::{
    sync::Arc,
    error::Error,
    collections::{BTreeMap, HashMap},
};
use ordered_float::OrderedFloat;

use petgraph::{
    graph::Graph, Directed 
};

use serde::Serialize;

// Tycho
use tycho_simulation::{
    models::Token,
    protocol::models::ProtocolComponent,
    protocol::state::ProtocolSim, 
};

use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq)]
pub struct PriceData {
    pub amount_out: f64,
    pub gas: f64
}

impl PriceData {
    pub fn output_with_gas(&self, token_price: f64) -> f64 {
        self.amount_out + self.gas * token_price
    }
}

#[derive(Debug,Clone)]
pub struct EdgeData {
    pub pool: Arc<ProtocolComponent>, 
    pub state: Arc<dyn ProtocolSim>, 
    pub points: BTreeMap<OrderedFloat<f64>, PriceData>
}

//impl EdgeData {
    //pub fn update(&mut self) {
    //    self.points.clear();  
    //}
//}

#[derive(Debug, Clone, Serialize)]
pub struct NodeData {
    pub token: Arc<Token>,
    pub price : f64, // to convert gas cost
}

// Custom error type
#[derive(Debug)]
pub enum GraphError {
    IoError(std::io::Error),
    SerializationError(serde_json::Error),
    StateNotFound(String),
    InvalidEdge(String),
    InvalidNode(String),
    NegativeCycleDetected,
    MissingPoolAddress(String),
    DuplicatePoolError(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::IoError(e) => write!(f, "IO error: {}", e),
            GraphError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            GraphError::StateNotFound(msg) => write!(f, "State not found: {}", msg),
            GraphError::InvalidEdge(msg) => write!(f, "Invalid edge: {}", msg),
            GraphError::InvalidNode(msg) => write!(f, "Invalid node: {}", msg),
            GraphError::NegativeCycleDetected => write!(f, "Negative cycle detected"),
            GraphError::MissingPoolAddress(msg) => write!(f, "Missing pool address: {}", msg),
            GraphError::DuplicatePoolError(msg) => write!(f, "Duplicate pool error: {}", msg),
        }
    }
}

impl Error for GraphError {}


pub fn graph_to_json(graph: &Graph<NodeData, RefCell<EdgeData>, Directed>, comment: &str) -> Result<String, GraphError> {
    let mut nodes_json = Vec::new();

    for (node_idx, node_data) in graph.node_weights().enumerate() {
        nodes_json.push(serde_json::json!({
            "index": node_idx, // Use node_idx as index
            "token": {
                "address": format!("{:?}", node_data.token.address), // Assuming Bytes can be debug-formatted or convert to hex string
                "decimals": node_data.token.decimals,
                "symbol": node_data.token.symbol,
                "gas": node_data.token.gas, 
            },
            "price": node_data.price,
        }));
    }

    let mut edges_json = Vec::new();

    for edge_ref in graph.raw_edges() {
        let edge_data = edge_ref.weight.borrow();
        edges_json.push(serde_json::json!({
            "source": edge_ref.source().index(),
            "target": edge_ref.target().index(),   
            "pool": {
                "address": edge_data.pool.id.clone(),
                "id": edge_data.pool.id.clone(),
                "protocol_system": edge_data.pool.protocol_system.clone(),
                "protocol_type_name": edge_data.pool.protocol_type_name.clone(),
                "chain": format!("{:?}", edge_data.pool.chain),
                "created_at": edge_data.pool.created_at,
            },
            "points": edge_data.points.iter()
                .map(|(k, v)| serde_json::json!({
                    "x": k.into_inner(),
                    "amount_out": v.amount_out,
                    "gas": v.gas,
                }))
                .collect::<Vec<serde_json::Value>>(),
        }));
    }

    let json_output = serde_json::json!({
        "comment": comment,
        "nodes": nodes_json,
        "edges": edges_json,
    });

    serde_json::to_string_pretty(&json_output)
        .map_err(GraphError::SerializationError)
}


#[derive(Default, Debug)]
pub struct Statistics {
    pub quoter: usize,
    pub quoter_failed: usize,
    pub quoter_failed_reverted: usize,
    pub quoter_failed_tick_exceeded: usize,
    pub quoter_failed_no_liquidity: usize,
    pub quoter_failed_sell_amount_exceeds_limit: usize,
    pub price_is_interpolated: usize,
    pub update_pool: usize,
    pub state_not_found_for_pool: usize,
    pub pool_type_counts: HashMap<String, usize>,
}

impl Statistics {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn print(&self) {
        let total_failed = self.quoter_failed +
                           self.quoter_failed_reverted +
                           self.quoter_failed_tick_exceeded +
                           self.quoter_failed_no_liquidity +
                           self.quoter_failed_sell_amount_exceeds_limit;


        println!(
            "Tycho requiest stats: Total: {} | Failed: {} (Reverted: {}, Tick Exceeded: {}, No Liquidity: {}, Sell Amount Exceeded: {}) | Pool Updates: {} | Interpolated: {} | State Not Found: {}",
            self.quoter,
            total_failed,
            self.quoter_failed_reverted,
            self.quoter_failed_tick_exceeded,
            self.quoter_failed_no_liquidity,
            self.quoter_failed_sell_amount_exceeds_limit,
            self.update_pool,
            self.price_is_interpolated,
            self.state_not_found_for_pool
        );
        if !self.pool_type_counts.is_empty() {
            println!("Pool type distribution: {}", {
                self.pool_type_counts
                    .iter()
                    .map(|(proto, count)| format!("{}={}", proto, count))
                    .collect::<Vec<_>>()
                    .join(", ")
            });
        }
    }
}

