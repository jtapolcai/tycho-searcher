// Dummy ProtocolSim for playback mode
#[derive(Debug)]
pub struct DummyProtocolSim;
use tycho_common::simulation::protocol_sim::{ProtocolSim, GetAmountOutResult, Balances};
use tycho_common::dto::ProtocolStateDelta;
use tycho_common::simulation::errors::TransitionError;
use tycho_common::Bytes;
use tycho_common::simulation::errors::SimulationError;
use std::collections::HashMap;
impl ProtocolSim for DummyProtocolSim {
    fn fee(&self) -> f64 { 0.0 }
    fn spot_price(&self, _a: &tycho_common::models::token::Token, _b: &tycho_common::models::token::Token) -> Result<f64, SimulationError> { Err(SimulationError::FatalError("Not implemented".to_string())) }
    fn get_amount_out(&self, _amount_in: num_bigint::BigUint, _token_in: &tycho_common::models::token::Token, _token_out: &tycho_common::models::token::Token) -> Result<GetAmountOutResult, SimulationError> { Err(SimulationError::FatalError("Not implemented".to_string())) }
    fn get_limits(&self, _a: Bytes, _b: Bytes) -> Result<(num_bigint::BigUint, num_bigint::BigUint), SimulationError> { Err(SimulationError::FatalError("Not implemented".to_string())) }
    fn delta_transition(&mut self, _delta: ProtocolStateDelta, _tokens: &HashMap<Bytes, tycho_common::models::token::Token>, _balances: &Balances) -> Result<(), TransitionError<String>> { Err(TransitionError::DecodeError("Dummy".into())) }
    fn clone_box(&self) -> Box<dyn ProtocolSim> { Box::new(DummyProtocolSim) }
    fn as_any(&self) -> &(dyn std::any::Any + 'static) { self }
    fn as_any_mut(&mut self) -> &mut (dyn std::any::Any + 'static) { self }
    fn eq(&self, _other: &(dyn ProtocolSim + 'static)) -> bool { false }
}
// src/solver/graph_types.rs

// STD
use std::{
    sync::Arc,
    error::Error,
    collections::{BTreeMap},
};
use serde::Deserialize;
 
use ordered_float::OrderedFloat;

use petgraph::{
    graph::Graph, Directed 
};

use serde::Serialize;

// Tycho
use tycho_common::{
    models::{token::Token},
    // ...existing code...
};
use tycho_simulation::{
    protocol::models::ProtocolComponent,
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

#[derive(Debug, Clone, Deserialize)]
pub struct NodeDataPlayback {
    pub token: Token,
    pub price: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EdgeDataPlayback {
    pub source: usize,
    pub target: usize,
    pub pool: ProtocolComponent,
    pub points: Vec<PricePointPlayback>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PricePointPlayback {
    pub x: f64,
    pub amount_out: f64,
    pub gas: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphJsonPlayback {
    pub nodes: Vec<NodeDataPlayback>,
    pub edges: Vec<EdgeDataPlayback>,
    #[serde(default)]
    pub gas_price: Option<f64>,
    #[serde(default)]
    pub source: Option<usize>,
    #[serde(default)]
    pub sink: Option<usize>,
}

pub fn graph_from_json_str(json_str: &str) -> Result<Graph<NodeData, RefCell<EdgeData>, Directed>, GraphError> {
    let parsed: GraphJsonPlayback = serde_json::from_str(json_str).map_err(GraphError::SerializationError)?;
    let mut graph = Graph::<NodeData, RefCell<EdgeData>, Directed>::new();
    let mut node_indices = Vec::new();
    for node in &parsed.nodes {
        let idx = graph.add_node(NodeData {
            token: Arc::new(node.token.clone()),
            price: node.price,
        });
        node_indices.push(idx);
    }
    for edge in &parsed.edges {
        let source = node_indices[edge.source];
        let target = node_indices[edge.target];
        let mut points = std::collections::BTreeMap::new();
        for p in &edge.points {
            points.insert(OrderedFloat(p.x), PriceData { amount_out: p.amount_out, gas: p.gas });
        }
        let edge_data = EdgeData {
            pool: Arc::new(edge.pool.clone()),
            state: Arc::new(DummyProtocolSim), // Use DummyProtocolSim for playback mode
            points,
        };
        graph.add_edge(source, target, RefCell::new(edge_data));
    }
    // Optionally: parsed.gas_price, parsed.source, parsed.sink can be returned or handled here if needed
    Ok(graph)
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


pub fn graph_to_json(
    graph: &Graph<NodeData, RefCell<EdgeData>, Directed>,
    comment: &str,
    gas_price: Option<f64>,
    source: Option<usize>,
    sink: Option<usize>,
) -> Result<String, GraphError> {
    let mut nodes_json = Vec::new();
    for (node_idx, node_data) in graph.node_weights().enumerate() {
        let token_value = serde_json::to_value(&*node_data.token).map_err(GraphError::SerializationError)?;
        nodes_json.push(serde_json::json!({
            "index": node_idx,
            "token": token_value,
            "price": node_data.price,
        }));
    }
    let mut edges_json = Vec::new();
    for edge_ref in graph.raw_edges() {
        let edge_data = edge_ref.weight.borrow();
        let pool_value = serde_json::to_value(&*edge_data.pool).map_err(GraphError::SerializationError)?;
        edges_json.push(serde_json::json!({
            "source": edge_ref.source().index(),
            "target": edge_ref.target().index(),
            "pool": pool_value,
            "points": edge_data.points.iter()
                .map(|(k, v)| serde_json::json!({
                    "x": k.into_inner(),
                    "amount_out": v.amount_out,
                    "gas": v.gas,
                }))
                .collect::<Vec<serde_json::Value>>(),
        }));
    }
    let mut json_output = serde_json::json!({
        "comment": comment,
        "nodes": nodes_json,
        "edges": edges_json,
    });
    if let Some(gas_price) = gas_price {
        json_output["gas_price"] = serde_json::json!(gas_price);
    }
    if let Some(source) = source {
        json_output["source"] = serde_json::json!(source);
    }
    if let Some(sink) = sink {
        json_output["sink"] = serde_json::json!(sink);
    }
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

