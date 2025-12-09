// src/solver/mod.rs
use crate::log_arb_info;

// STD
use std::sync::Arc;

use tokio::sync::mpsc::Receiver;

// Alloy for gas price queries
use alloy::providers::{ProviderBuilder, Provider};

// Hex
use hex;

// Petgraph
use petgraph::{
    graph::{EdgeIndex, Graph, NodeIndex},
    Directed,
};

// Collections
use std::collections::{HashSet, HashMap, BTreeMap};

use num_bigint::BigUint;

// Tycho
use tycho_simulation::{
    protocol::models::{Update, ProtocolComponent},
};

use tycho_common::{
    models::{token::Token},
    simulation::protocol_sim::ProtocolSim,
};

use std::cell::RefCell;

use std::time::Instant;

pub mod graph_types; // Contains NodeData, EdgeData, GraphError, etc.
pub mod graph_components; // Contains GraphComponents and BCC logic
pub mod logging; // Console logging toggle
pub mod price_quoter;
pub mod bellman_ford;
pub mod arbitrage_save;
// pub mod crate::execution::arbitrage_save_worker;

use bellman_ford::{find_all_negative_cycles,describe_path};
// Use types from graph_types module
use graph_types::{NodeData, EdgeData, GraphError, graph_to_json, Statistics, GraphJsonPlayback};
// Import GraphComponents from graph_components module
use crate::searcher::graph_components::GraphComponents;
use crate::searcher::arbitrage_save::verify_arbitrage_opportunity;

use crate::log_quoter_info;

// single path
#[derive(Debug, Clone)]
pub struct SwapRequest {
    pub tokens: Vec<Token>,
    pub pools: Vec<ProtocolComponent>,
    pub amount: BigUint,
    pub amounts_out: Vec<BigUint>,
}

#[derive(Debug)]
pub struct Searcher {
    graph: Graph::<NodeData, RefCell<EdgeData>, Directed>,
    components: GraphComponents,
    node_indices: HashMap<String, NodeIndex>,
    pub edge_index_by_pool: HashMap<String, Vec<EdgeIndex>>,
    rx: Receiver<Update>,
    start_token_index: Option<NodeIndex>,
    start_token_name: Option<String>,
    blacklist_tokens: HashSet<Vec<u8>>,
    stats: Statistics,
    pub cli: crate::command_line_parameters::Cli, 
    rpc_url: String, // Ethereum node URL for fetching gas price
    pub did_export_and_exit: bool,
    pub last_gas_price: Option<f64>,
    pub last_source: Option<usize>,
    pub last_sink: Option<usize>,
}

impl Searcher {
    pub fn new(
        block_update_rx: Receiver<Update>,
        cli: crate::command_line_parameters::Cli,
        blacklist_tokens: HashSet<Vec<u8>>,
        rpc_url: String, // Added parameter
    ) -> Self {
        Self {
            rx: block_update_rx,
            graph: Graph::<NodeData, RefCell<EdgeData>, petgraph::Directed>::new(),
            components: GraphComponents::default(), // Initialize with default empty components
            node_indices: HashMap::new(),
            edge_index_by_pool: HashMap::new(),
            start_token_index: None, // Will be set later when the token is found in the graph
            start_token_name: None,
            blacklist_tokens,
            stats: Statistics::default(),
            cli,
            rpc_url, // Initialize new field
            did_export_and_exit: false,
            last_gas_price: None,
            last_source: None,
            last_sink: None,
        }
    }

    pub fn get_or_insert_token_node(&mut self, token: &Token) -> NodeIndex {
        let address = format!("{:#042x}", token.address);
        *self.node_indices.entry(address.clone()).or_insert_with(|| {
            let node_data = NodeData {
                token: Arc::new(token.clone()),
                price: {
                    if token.symbol=="WETH" {
                        1.0 
                    } else {
                        0.0
                    }
                }
            };
            let index = self.graph.add_node(node_data);
            index
        })
    }

    pub fn is_finished(&self) -> bool {
        self.did_export_and_exit
    }

    /// Adds a pool as an edge (or to an existing edge's pool list) in the graph.
    ///
    /// Checks if an edge already exists between token0 and token1.
    /// If it exists, the new pool is added to the existing edge's pools vector.
    /// If no such edge exists, a new edge is created with the new pool.
    pub fn add_pool(
        &mut self,
        token0: &Token,
        token1: &Token,
        new_pool: &ProtocolComponent,
        state: Box<dyn ProtocolSim>,  
        pool_address: String
    ) -> Result<EdgeIndex, GraphError> {
        let node_index_token_0 = self.get_or_insert_token_node(token0);
        let node_index_token_1 = self.get_or_insert_token_node(token1);

        // Check for existing edge 
        for edge_ref in self.graph.edges_connecting(node_index_token_0, node_index_token_1) {
            let edge_data = edge_ref.weight().borrow();
            if edge_data.pool.id == new_pool.id {
                return Err(GraphError::DuplicatePoolError(
                    format!(
                        "Pool with ID '{}' already exists on edge between {} and {}.",
                        new_pool.id, token0.symbol, token1.symbol
                    )
                ));
            }
        }
        // Edge does not exist, create a new one
        let edge_data = EdgeData {
            pool: Arc::new(new_pool.clone()),
            state: Arc::from(state.clone()),
            points: BTreeMap::new()
        };
        let idx = self.graph.add_edge(node_index_token_0, node_index_token_1, RefCell::new(edge_data));
        self.edge_index_by_pool
            .entry(pool_address)
            .or_default()
            .push(idx);

        let _price_token_0 = self.graph.node_weight(node_index_token_0).map(|n| n.price).unwrap_or(0.0);
        let _price_token_1 = self.graph.node_weight(node_index_token_1).map(|n| n.price).unwrap_or(0.0);

        Ok(idx)
    }


    fn remove_pool_edges(&mut self, pool_address: &str) -> Result<(), GraphError> {
        if let Some(edges) = self.edge_index_by_pool.remove(pool_address) {
            for &idx in &edges {
                self.graph.remove_edge(idx);
            }
        }
        Ok(())
    }

    pub fn update_graph(&mut self, update: &Update) -> Result< HashSet<usize>, GraphError> {
        let mut bool_update_graph = false;
        let first_run=self.start_token_index.is_none();
        let block_number= format!("block:{}",update.block_number_or_timestamp);
        // Top-level debug: always print counts so we know why no edges are added
        println!("[DEBUG] update summary: block={} new_pairs={} states={}", update.block_number_or_timestamp, update.new_pairs.len(), update.states.len());
        for (_id, comp) in update.new_pairs.iter() {
            let pool_address = format!("0x{}", hex::encode(&comp.id));
            // Debug helpers to diagnose missing states / skipped pools
            //println!("[DEBUG] update.new_pairs.len={} update.states.len={}", update.new_pairs.len(), update.states.len());
            //println!("[DEBUG] processing pair key='{}' pool_address='{}' protocol_system='{}' tokens_count={}", _id, pool_address, comp.protocol_system, comp.tokens.len());
            for t in &comp.tokens {
                log_quoter_info!("[DEBUG]   token symbol='{}' address='{}' decimals={}", t.symbol, format!("{:#042x}", t.address), t.decimals);
            }
            if update.states.contains_key(_id) {
                log_quoter_info!("[DEBUG] state found by new_pairs key: '{}'", _id);
            } else if update.states.contains_key(&pool_address) {
                log_quoter_info!("[DEBUG] state found by computed pool_address: '{}'", pool_address);
            } else {
                log_quoter_info!("[DEBUG] state MISSING for pool {} (key: {})", pool_address, _id);
            }
            if comp.protocol_system == "uniswap_v4" {
                //println!("Uniswap V4 pool detected: {} {}", pool_address, pool_address.len());
            };
            *self.stats.pool_type_counts
                .entry(comp.protocol_system.clone())
                .or_insert(0) += 1;
            for i in 0..comp.tokens.len() {
                if !self.blacklist_tokens.contains(&comp.tokens[i].address.to_vec()) {
                    for j in (i + 1)..comp.tokens.len() {
                        if !self.blacklist_tokens.contains(&comp.tokens[j].address.to_vec()) {
                            // Try to find a matching state using multiple key formats (map key, 0x-prefixed pool address, or raw hex)
                            let raw_hex = hex::encode(&comp.id);
                            let keys_to_try = vec![_id.to_string(), pool_address.clone(), raw_hex.clone()];
                            let mut found_state: Option<Box<dyn tycho_common::simulation::protocol_sim::ProtocolSim>> = None;
                            for key in keys_to_try.iter() {
                                if let Some(s) = update.states.get(key) {
                                    found_state = Some(s.clone());
                                    break;
                                }
                            }
                            if let Some(state) = found_state {
                                let state_clone = state.clone();
                                self.add_pool(
                                    &comp.tokens[i],
                                    &comp.tokens[j],
                                    comp,
                                    state_clone.clone(),
                                    pool_address.clone(),
                                )?;
                                self.add_pool(
                                    &comp.tokens[j],
                                    &comp.tokens[i],
                                    comp,
                                    state_clone,
                                    pool_address.clone(),
                                )?;
                                bool_update_graph = true;
                            } else {
                                self.stats.state_not_found_for_pool += 1;
                            }
                        } else {
                            //println!("Skipping blacklisted token: {:?} (j:{})", comp.tokens[j],j);
                        }
                    }
                } else {
                    //println!("Skipping blacklisted token: {:?} (i:{})", comp.tokens[i],i);
                }
            }
        }

        for comp in update.removed_pairs.values() {
            let pool_address = format!("0x{}", hex::encode(&comp.id)); // format!("{:#042x}", comp.id);
            //println!("Removing pool edges for: {}", pool_address);
            self.remove_pool_edges(&pool_address)?;
            let entry = self.stats.pool_type_counts
                .entry(comp.protocol_system.clone())
                .or_insert(0);
            if *entry > 0 {
                *entry -= 1;
            }
            bool_update_graph = true;
        }

        if bool_update_graph {
            //println!("Graph updated with {} nodes and {} edges", self.graph.node_count(), self.graph.edge_count());
            //let json_string = graph_to_json(&self.graph)?;
            //std::fs::write("graph.json", json_string).map_err(GraphError::IoError)?;
            //println!("Graph is exported into graph.json with {} nodes and {} edges",
            if self.start_token_index.is_none() {
                // Find the start token index if it exists in the graph
                if let Some(index) = self.node_indices.get(&self.cli.start_token) {
                    self.start_token_index = Some(*index);
                    self.start_token_name = self.graph.node_weight(*index).map(|n| n.token.symbol.clone());
                    //println!("Start token {:?} found", self.start_token_name);
                    self.stats.print();
                } else {
                    //println!("Start token '{}' not found in the graph.", self.cli.start_token);
                }
            }
            println!("Graph now has {} nodes and {} edges", self.graph.node_count(), self.graph.edge_count());
            self.components = GraphComponents::build(self.graph.clone(), self.start_token_index); 
            println!("Graph components built with {} components.", self.components.graph_comps.len());
            println!("Largest component has {} nodes and {} edges",
                self.components.graph_comps.iter().map(|g| g.node_count()).max().unwrap_or(0),
                self.components.graph_comps.iter().map(|g| g.edge_count()).max().unwrap_or(0));
            // Find the largest component by node count
            if let Some((idx, largest)) = self.components.graph_comps
                .iter()
                .enumerate()
                .max_by_key(|(_, g)| g.node_count())
            {
                //println!("Largest component has {} nodes and {} edges.",
                //    largest.node_count(),
                //    largest.edge_count()
                //);
                
                // Only export to JSON if the CLI flag is enabled
                if self.cli.export_graph {
                    // Export full graph &self.graph
                    let full_graph_json = graph_to_json(
                        largest,
                        &block_number,
                        self.last_gas_price,
                        self.last_source,
                        self.last_sink,
                    )?;
                    if let Err(e) = std::fs::write("graph.json", full_graph_json) {
                        eprintln!("[WARNING] Failed to write graph.json: {}", e);
                    } else {
                        println!("Full graph written to graph.json ({} nodes, {} edges)", 
                                self.graph.node_count(), self.graph.edge_count());
                    }
                    // Export largest component
                    let largest_json = graph_to_json(
                        largest,
                        &block_number,
                        self.last_gas_price,
                        self.last_source,
                        self.last_sink,
                    )?;
                    if let Err(e) = std::fs::write("largest_component.json", largest_json) {
                        eprintln!("[WARNING] Failed to write largest_component.json: {}", e);
                    } else {
                        println!("Largest component (id:{}) written to largest_component.json ({} nodes, {} edges)", 
                                idx, largest.node_count(), largest.edge_count());
                    }
                }
            }
        } else {
            println!("No changes in the graph, skipping update.");
        }
 
        let mut components_to_update = HashSet::new();
        for (_address, _state) in update.states.iter() {
            if let Some(edge_indices) = self.edge_index_by_pool.get(_address) {
                let mut updated = false;
                for &edge_idx in edge_indices {
                    // First, get the endpoints and node data in a separate scope
                    let (_node_data_0, _node_data_1) = if let Some((from_idx, to_idx)) = self.graph.edge_endpoints(edge_idx) {
                        let node_data_0 = self.graph.node_weight(from_idx).cloned();
                        let node_data_1 = self.graph.node_weight(to_idx).cloned();
                        (node_data_0, node_data_1)
                    } else {
                        (None, None)
                    };
                    updated = true;
                }
                if updated {
                    // Find all component IDs containing any of these edge indices
                    for (comp_id, comp) in self.components.graph_comps.iter().enumerate() {
                        if edge_indices.iter().any(|idx| comp.edge_indices().any(|e| e == *idx)) {
                            components_to_update.insert(comp_id);
                        }
                    }
                }
            } else {
                //println!("Failed to update edge states for pool {} {}: MISSING POOL ADDRESS", address, address.len());
            }
        }
        if first_run {
            for (i, _comp) in self.components.graph_comps.iter().enumerate() {
                components_to_update.insert(i);
            }
        }
        Ok(components_to_update)
    }


    pub async fn run(mut self) -> anyhow::Result<()> {
        if self.did_export_and_exit {
            println!("Debug mode: exiting without running the solver loop again.");
            return Err(anyhow::anyhow!("Debug mode: exited without running the solver loop again."));
        }
        //println!("Solver loop started...");
        while let Some(data) = self.rx.recv().await {
            let mut gas_price = match self.rpc_url.parse() {
                Ok(url) => {
                    let provider = ProviderBuilder::new().connect_http(url);
                    match provider.get_gas_price().await {
                        Ok(price) => price as f64 / 1_000_000_000.0, // Convert wei to gwei
                        Err(e) => {
                            eprintln!("Failed to get gas price: {}, using default 25 gwei", e);
                            25.0 // Default 25 gwei
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Invalid RPC URL: {}, using default 25 gwei", e);
                    25.0 // Default 25 gwei
                }
            };
            if self.cli.debug {
                gas_price *= 0.01;
                println!("=== Processing new block {} (gas price is scaled down to: {:.2} gwei)===", data.block_number_or_timestamp, gas_price);
            } else {
                println!("=== Processing new block {} (gas price is: {:.2} gwei)===", data.block_number_or_timestamp, gas_price);
            }
            self.last_gas_price = Some(gas_price);
            
            // Periodic channel health check (every 10 blocks)
            // if data.block_number % 10 == 0 {
            //     println!("Statistics of the channel between the searcher and executor thread: {}", self.get_channel_health());
            // }
            
            let block_number = format!("block:{}", data.block_number_or_timestamp);
            let first_run=self.start_token_index.is_none();
            match self.update_graph(&data) {
                Ok(components_to_update) => {
                    if components_to_update.is_empty() {
                        println!("No components to update, skipping cycle detection.");
                        continue;
                    }
                    //println!("Components to update: {:?}", components_to_update);
                    for comp_id in components_to_update {
                        // Do something with each component id
                        if !first_run {
                            println!("Component {} is being updated", comp_id);
                        }
                        let orig_start_index = self.start_token_index.unwrap_or_else(|| {
                            panic!("Start token index is not set, cannot find cycles.");
                        });
                        let start_index = self.components.original_node_to_component_map.get(&orig_start_index).cloned().unwrap_or_else(|| {
                            panic!(
                                "Start token index {:?} not found in component, cannot find cycles.",
                                orig_start_index
                            );
                        }).iter().find_map(|(cid, idx)| if *cid == comp_id { Some(*idx) } else { None }).unwrap_or_else(|| {
                            panic!(
                                "Component id {} not found in original_node_to_component_map for start token index {:?}.",
                                comp_id, orig_start_index
                            );
                        });

                        self.last_source = Some(start_index.index());
                        self.last_sink = Some(start_index.index());
                        // Extract the data needed from self before mutable borrow
                        let start_token_name = self.start_token_name.clone();
                        // If verify_arbitrage_opportunity needs more from self, extract here

                        let graph_component = &mut self.components.graph_comps[comp_id];
                        if graph_component.node_count() >= self.cli.min_size_export {
                            let file_name = format!("graphs/graph_{}.json", comp_id);
                            let json_string = graph_to_json(
                                graph_component,
                                &block_number,
                                self.last_gas_price,
                                self.last_source,
                                self.last_sink,
                            )?;
                            if let Err(e) = std::fs::write(&file_name, json_string) {
                                eprintln!("[WARNING] Failed to write {}: {}", file_name, e);
                            } else {
                                println!("Graph component {} has {} nodes and {} edges (start node:{:?} {})", 
                                    file_name, graph_component.node_count(), graph_component.edge_count(),start_index, 
                                    graph_component.node_weight(start_index).map(|n| n.token.symbol.clone()).unwrap_or_else(|| "Unknown".to_string()));
                                // If debug+export_graph: export and exit after first component
                                if self.cli.debug && self.cli.export_graph && !self.did_export_and_exit {
                                    println!("[SNAPSHOT] To run playback: cp {} graph.json && cargo run --bin tycho-searcher -- --playback --log-pool", file_name);
                                    self.did_export_and_exit = true;
                                }
                            }
                        }
                        let start_timer = Instant::now();
                        let cycles = find_all_negative_cycles(
                            graph_component,
                            &mut self.stats,
                            start_index,
                            start_index,
                            self.cli.bf_max_iterations,
                            self.cli.bf_amount_in_min,
                            self.cli.bf_amount_in_max,
                            self.cli.bf_max_outer_iterations,
                            self.cli.bf_gss_tolerance,
                            self.cli.bf_gss_max_iter,
                            gas_price, 
                        );
                        if graph_component.node_count() >= 5 || !first_run {
                            println!("Runtime of the cycle search: {:?} in component {} with nodes {}", start_timer.elapsed(), comp_id, graph_component.node_count());
                            if self.stats.quoter>100{
                                self.stats.print();
                                self.stats.reset();
                            }
                        }
                        if cycles.is_empty() {
                            if graph_component.node_count() >= 10 || !first_run {
                                 //println!("Component {}: No negative cycle from: {}", comp_id, start_token_name.unwrap_or("N/A".to_string()));
                            }
                        } else {
                            for (cycle_amount_in, _cycle_amount_out, _cycle_total_gas, cycle) in cycles {
                                log_arb_info!("---- Negative cycle detected  -----");
                                let cycle_clone = cycle.clone();
                                // After calling verify_arbitrage_opportunity, where you submit the job
                                if let Some((_tokens, _pools, _amount_in, _amounts_out, profit_in_microeth, _total_gas, profit_without_gas_in_microeth)) =
                                    verify_arbitrage_opportunity(
                                        cycle_clone.as_ref().clone(),
                                        cycle_amount_in,
                                        start_token_name.clone().unwrap_or_else(|| "Unknown".to_string()),
                                        &graph_component,
                                        gas_price
                                    ) {
                                    log_arb_info!("Try token {} {:.6} ({}) profit: {:.0} mikreETH (minus gas: {:.0} mikroETH) - TOO_NEGATIVE - SKIPPING", 
                                        start_token_name.as_deref().unwrap_or("Unknown"), 
                                        cycle_amount_in, 
                                        describe_path(&graph_component, &cycle_clone), 
                                        profit_in_microeth,
                                        profit_without_gas_in_microeth
                                    );
                                }
                            };
                        } 
                        if self.did_export_and_exit {
                            return Err(anyhow::anyhow!("Debug mode: exited without running the solver loop again."))
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error updating graph: {}", e);
                }
            }
            //println!("Token graph updating finished.");
        }
        println!("❌ Solver run() loop exited!");
        Ok(())
    }

    /// Playback: run cycle search on loaded snapshot and print results
    pub fn playback_cycle_search(&mut self) {
        // Feltételezzük, hogy a graph.json már egyetlen komponens (a teljes vagy legnagyobb részgráf)
        println!("[PLAYBACK] Loaded graph: nodes={}, edges={}", self.graph.node_count(), self.graph.edge_count());
        let gas_price = self.last_gas_price.unwrap_or(25.0);
        let start_index = self.last_source
            .and_then(|idx| self.graph.node_indices().find(|i| i.index() == idx))
            .or_else(|| self.start_token_index)
            .unwrap_or_else(|| self.graph.node_indices().next().unwrap());
        if self.graph.node_count() >= 5 {
            let mut stats = crate::searcher::graph_types::Statistics::default();
            let mut graph_mut = self.graph.clone();
            let cycles = crate::searcher::bellman_ford::find_all_negative_cycles(
                &mut graph_mut,
                &mut stats,
                start_index,
                start_index,
                self.cli.bf_max_iterations,
                self.cli.bf_amount_in_min,
                self.cli.bf_amount_in_max,
                self.cli.bf_max_outer_iterations,
                self.cli.bf_gss_tolerance,
                self.cli.bf_gss_max_iter,
                gas_price,
            );
            println!("[PLAYBACK] Found {} negative cycles", cycles.len());
        } else {
            println!("[PLAYBACK] Graph too small for cycle search ({} nodes)", self.graph.node_count());
        }
        println!("[PLAYBACK] Cycle search finished.");
    }

    pub fn from_graph_json(
        cli: crate::command_line_parameters::Cli,
        blacklist_tokens: HashSet<Vec<u8>>,
        rpc_url: String,
        graph_json_path: &str,
    ) -> Result<Self, crate::searcher::graph_types::GraphError> {
        use std::fs;
        use crate::searcher::graph_types::{graph_from_json_str, DummyProtocolSim, GraphJsonPlayback};
        use std::sync::Arc;
        let json_str = fs::read_to_string(graph_json_path).map_err(crate::searcher::graph_types::GraphError::IoError)?;
        let parsed: GraphJsonPlayback = serde_json::from_str(&json_str).map_err(crate::searcher::graph_types::GraphError::SerializationError)?;
        let mut graph = graph_from_json_str(&json_str)?;
        // Patch all edge states to DummyProtocolSim
        for edge in graph.edge_weights_mut() {
            edge.get_mut().state = Arc::new(DummyProtocolSim);
        }
        Ok(Self {
            rx: tokio::sync::mpsc::channel(1).1, // dummy receiver, no feed in playback mode
            graph,
            components: GraphComponents::default(),
            node_indices: HashMap::new(),
            edge_index_by_pool: HashMap::new(),
            start_token_index: parsed.source.map(|idx| petgraph::graph::NodeIndex::new(idx)),
            start_token_name: None,
            blacklist_tokens,
            stats: Statistics::default(),
            cli,
            rpc_url,
            did_export_and_exit: false,
            last_gas_price: parsed.gas_price,
            last_source: parsed.source,
            last_sink: parsed.sink,
        })
    }
}