#[cfg(test)]
mod tests {
    //use super::*;
    use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
    use std::cell::RefCell;
    
     // Needed for to_index
    use petgraph::visit::NodeIndexable; // Import for to_index method
    use tycho_searcher::searcher::graph_types::{NodeData, EdgeData, PriceData,Statistics};
    use tycho_searcher::searcher::bellman_ford::{bellman_ford_initialize_relax, find_all_negative_cycles,evaluate_cycle};
    use tycho_searcher::searcher::price_quoter::PriceDataRaw;
    use std::collections::{BTreeMap, HashMap};
    use ordered_float::OrderedFloat;
    use std::sync::Arc;
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use tycho_simulation::{
        models::{Balances, Token},
        protocol::{
            errors::{SimulationError, TransitionError},
            models::GetAmountOutResult,
        },
        protocol::models::ProtocolComponent,
        protocol::state::ProtocolSim,
    };
    use tycho_common::{dto::ProtocolStateDelta, Bytes};
    use tycho_common::models::Chain;
    use chrono::NaiveDateTime;
    use std::any::Any;

    
    use num_traits::Zero;

    #[derive(Debug, Clone)]
    pub struct UniswapV2Sim {
        price: f64,
        liquidity: f64, // L^2 = x * y
        gas: f64,
    }

    impl UniswapV2Sim {
        pub fn new(price: f64, liquidity: f64, gas: f64) -> Self {
            Self { price, liquidity: liquidity*liquidity, gas }
        }

        fn reserves(&self) -> (f64, f64) {
            // From price and L = x * y → x = sqrt(L / price), y = sqrt(L * price)
            let x = (self.liquidity / self.price).sqrt();
            let y = (self.liquidity * self.price).sqrt();
            (x, y)
        }
    }

    impl ProtocolSim for UniswapV2Sim {
        fn fee(&self) -> f64 {
            0.003 // Uniswap v2 default fee: 0.3%
        }

        fn spot_price(&self, _base: &Token, _quote: &Token) -> Result<f64, SimulationError> {
            let (x, y) = self.reserves();
            Ok(y / x)
        }

        fn get_amount_out(
            &self,
            amount_in: BigUint,
            _token_in: &Token,
            _token_out: &Token,
        ) -> Result<GetAmountOutResult, SimulationError> {
            let ( x,  y) = self.reserves();

            let amount_in_f64 = amount_in.to_f64().unwrap_or(0.0);
            let amount_in_with_fee = amount_in_f64 * (1.0 - self.fee());

            let new_x = x + amount_in_with_fee;
            let new_y = self.liquidity / new_x;
            let amount_out_f64 = y - new_y;

            Ok(GetAmountOutResult {
                amount: BigUint::from(amount_out_f64.max(0.0) as u64),
                gas: BigUint::from(self.gas.max(0.0) as u64),
                new_state: Box::new(self.clone()),
            })
        }

        fn get_limits(
            &self,
            _sell_token: Bytes,
            _buy_token: Bytes,
        ) -> Result<(BigUint, BigUint), SimulationError> {
            // Just return a simple test range
            Ok((BigUint::from(1u64), BigUint::from(10_000u64)))
        }

        fn delta_transition(
            &mut self,
            _delta: ProtocolStateDelta,
            _tokens: &HashMap<Bytes, Token>,
            _balances: &Balances,
        ) -> Result<(), TransitionError<String>> {
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn ProtocolSim> {
            Box::new(self.clone())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn eq(&self, other: &dyn ProtocolSim) -> bool {
            other
                .as_any()
                .downcast_ref::<Self>()
                .map_or(false, |o| o.price == self.price && o.liquidity == self.liquidity)
        }
    }

    // Helper to create NodeData based on index (symbol = "n{index}")
    fn create_node_data(index: usize) -> NodeData {
        NodeData {
            token: Arc::new(Token {
                address: Bytes::from(hex::decode(format!("{:0>40x}", index)).unwrap()),
                decimals: 2,
                symbol: format!("n{}", index), // Symbol is now "n{index}"
                gas: BigUint::from(30000u64 + index as u64),
            }),
            price: 0.0,
        }
    }

    fn create_edge_data(
        pool_id: &str,
        protocol: &str,
        points: BTreeMap<OrderedFloat<f64>, PriceData>,
        price: f64,
        liquidity: f64,
        gas: f64
    ) -> RefCell<EdgeData> {
        RefCell::new(EdgeData {
            pool: 
                Arc::new(ProtocolComponent::new(
                    Bytes::from(pool_id.as_bytes()),
                    protocol.to_string(),
                    "mock_type".to_string(),
                    Chain::Ethereum,
                    vec![], // tokens
                    vec![], // contract_ids
                    HashMap::new(), // static_attributes
                    Bytes::new(),   // creation_tx
                    NaiveDateTime::parse_from_str("2023-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S").unwrap(),
                )),
            points,
            state: Arc::new(UniswapV2Sim::new(price, liquidity, gas)) as Arc<dyn ProtocolSim>,
        })
    }
    
    fn make_simple_graph() -> (Graph<NodeData, RefCell<EdgeData>>, NodeIndex) {
        let mut graph = Graph::<NodeData, RefCell<EdgeData>>::new();
        let n0 = graph.add_node(create_node_data(0));
        let n1 = graph.add_node(create_node_data(1));
        graph.add_edge(n0, n1, create_edge_data("0", "test_proto_rev",BTreeMap::new(), 0.15,10000.0,5.0));
        graph.add_edge(n1, n0, create_edge_data("1", "test_proto_rev",BTreeMap::new(), 11.9,100000.0,5.0));
        (graph, n0)
    }

    fn make_simple_graph_single_pool() -> (Graph<NodeData, RefCell<EdgeData>>, NodeIndex) {
        let mut graph = Graph::<NodeData, RefCell<EdgeData>>::new();
        let n0 = graph.add_node(create_node_data(0));
        let n1 = graph.add_node(create_node_data(1)); 
        graph.add_edge(n0, n1, create_edge_data("0", "test_proto_rev",BTreeMap::new(), 0.15,10000.0,5.0));
        graph.add_edge(n1, n0, create_edge_data("0", "test_proto_rev",BTreeMap::new(), 11.9,100000.0,5.0));
        (graph, n0)
    }

    fn make_simple_graph_no_solution() -> (Graph<NodeData, RefCell<EdgeData>>, NodeIndex) {
        let mut graph = Graph::<NodeData, RefCell<EdgeData>>::new();
        let n0 = graph.add_node(create_node_data(0));
        let n1 = graph.add_node(create_node_data(1));
        graph.add_edge(n0, n1, create_edge_data("0", "test_proto_rev",BTreeMap::new(), 0.11,10000.0,5.0));
        graph.add_edge(n1, n0, create_edge_data("0", "test_proto_rev",BTreeMap::new(), 10.9,100000.0,5.0));
        (graph, n0)
    }

    fn print_result(result: &(Vec<PriceDataRaw>, Vec<Option<EdgeIndex>>)) {
        let (price_data_vec, edge_index_vec) = result;
        println!("PriceDataRaw:");
        for (i, price_data) in price_data_vec.iter().enumerate() {
            println!("  [{}]: {:?}", i, price_data);
        }
        println!("EdgeIndex Option:");
        for (i, edge_index) in edge_index_vec.iter().enumerate() {
            println!("  [{}]: {:?}", i, edge_index);
        }
    }

    #[test]
    fn test_bellman_ford_initialize_relax_adv_basic() {
        let (mut graph, source) = make_simple_graph();
        let mut stats = Statistics::default();
        let result = bellman_ford_initialize_relax(
            &mut graph,
            &mut stats,
            source,
            3,
            BigUint::from(1u64),
            BigUint::from(1_000_000_000u64)
        );
        // Should have a non-empty result for each node
        print_result(&(result.0.clone(), result.1.clone()));
        assert_eq!(result.0.len(), 2);
        // The source node should have a valid PriceDataRaw (e.g., amount_out or gas not zero)
        let price_data = &result.0[graph.to_index(source)];
        assert!(!price_data.amount_out.is_zero() || price_data.gas != BigUint::zero());
    }

    #[test]
    fn test_find_all_negative_cycles_newton_iteration_with_very_small_start_value() {
        let (mut graph, source) = make_simple_graph();
        let mut stats = Statistics::default();
        let cycles = find_all_negative_cycles(&mut graph, &mut stats, source, source, 2,0.5,1000.0,20,1e-4,40, 1.0);
        // Should find at least one cycle in this simple 2-node, 2-edge graph
        print!("Cycles found: {:?}", cycles);
        assert!(cycles.len()== 1);
        match evaluate_cycle(
            cycles[0].3.as_ref().clone(), // Use the 4th element (Vec<EdgeIndex>)
            BigUint::from((cycles[0].0 * 100.0) as u64),
            &graph,
            source,
        ) {
            Ok((amount_out,total_gas)) => {
                let amount_out_f64 = amount_out.to_f64().unwrap_or(0.0);
                let total_gas_f64 = total_gas.to_f64().unwrap_or(0.0);
                let profit = amount_out_f64 * 0.01 - cycles[0].0 - total_gas_f64 * 0.01;
                println!("profit {:?}", profit);
                assert!(profit > 25.0 );
            },
            Err(e) => panic!("Error evaluating cycle: {}", e),
        }
    }
    #[test]
    fn test_find_all_negative_cycles_newton_iteration_with_small_start_value() {
        let (mut graph, source) = make_simple_graph();
        let mut stats = Statistics::default();
        let cycles = find_all_negative_cycles(&mut graph, &mut stats, source, source, 2,1.0,1000.0,20,1e-4,40, 1.0);
        // Should find at least one cycle in this simple 2-node, 2-edge graph
        print!("Cycles found: {:?}", cycles);
        assert!(cycles.len()== 1);
        match evaluate_cycle(
            cycles[0].3.as_ref().clone(), // Use the 4th element (Vec<EdgeIndex>)
            BigUint::from((cycles[0].0 * 100.0) as u64),
            &graph,
            source,
        ) {
            Ok((amount_out,total_gas)) => {
                let amount_out_f64 = amount_out.to_f64().unwrap_or(0.0);
                let total_gas_f64 = total_gas.to_f64().unwrap_or(0.0);
                let profit = amount_out_f64 * 0.01 - cycles[0].0 - total_gas_f64 * 0.01;
                println!(" profit {:?} ({},{})", profit, amount_out_f64, total_gas_f64);
                assert!(profit > 25.0 );
            },
            Err(e) => panic!("Error evaluating cycle: {}", e),
        }
    }
     #[test]
    fn test_find_all_negative_cycles_newton_iteration_with_large_start_value() {
        let (mut graph, source) = make_simple_graph();
        let mut stats = Statistics::default();
        let cycles = find_all_negative_cycles(&mut graph, &mut stats, source, source, 2,75.0,1000.0,20,1e-4,40, 1.0);
        // Should find at least one cycle in this simple 2-node, 2-edge graph
        print!("Cycles found: {:?}", cycles);
        assert!(cycles.len()== 1);
        match evaluate_cycle(
            cycles[0].3.as_ref().clone(), // Use the 4th element (Vec<EdgeIndex>)
            BigUint::from((cycles[0].0 * 100.0) as u64),
            &graph,
            source,
        ) {
            Ok((amount_out,total_gas)) => {
                let amount_out_f64 = amount_out.to_f64().unwrap_or(0.0);
                let total_gas_f64 = total_gas.to_f64().unwrap_or(0.0);
                let profit = amount_out_f64 * 0.01 - cycles[0].0 - total_gas_f64 * 0.01;
                println!("profit {:?}", profit);
                assert!(profit > 25.0 );
            },
            Err(e) => panic!("Error evaluating cycle: {}", e),
        }
    }
        #[test]
    fn test_find_all_negative_cycles_infeasible_as_single_pool() {
        let (mut graph, source) = make_simple_graph_single_pool();
        let mut stats = Statistics::default();
        let cycles = find_all_negative_cycles(&mut graph, &mut stats, source, source, 2,1.0,1000.0,20,1e-4,40, 1.0);
        // Should find at least one cycle in this simple 2-node, 2-edge graph
        print!("Cycles found: {:?}", cycles);
        assert!(cycles.len()== 0);
    }
    #[test]
    fn test_find_all_negative_cycles_infeasible_because_of_prices() {
        let (mut graph, source) = make_simple_graph_no_solution();
        let mut stats = Statistics::default();
        let cycles = find_all_negative_cycles(&mut graph, &mut stats, source, source, 2,100.0,1000.0,10,1e-4,20, 1.0);
        // Should find at least one cycle in this simple 2-node, 2-edge graph
        assert!(cycles.is_empty());
        print!("Cycles found: {:?}", cycles);
        // Each cycle should be a tuple (f64, Vec<EdgeIndex>)
        //for (val, path) in cycles {
        //    assert!(val >= 0.0 || val < 0.0); // val is a float
        //    assert!(!path.is_empty());
        //}
    }
}