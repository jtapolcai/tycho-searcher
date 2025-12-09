// === Tycho related ===
//extern crate tycho_simulation;
use tycho_simulation::{tycho_client::feed::component_tracker::ComponentFilter};
use tycho_common::{models::Chain};
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
         protocol::{
            ekubo::state::EkuboState,
            filters::{
                balancer_v2_pool_filter, curve_pool_filter
            },
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            pancakeswap_v2::state::PancakeswapV2State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
};

use std::env;
use clap::Parser;
// Add ethers imports
//use ethers::providers::{Provider, Http, Middleware};
//use ethers::types::U256;
use std::collections::HashSet;
use hex::decode;

#[derive(Parser,Debug,Clone)]
pub struct Cli {
    /// The tvl threshold to filter the graph by
    //#[arg(short, long, default_value_t = 10000.0)]
    #[arg(short, long, default_value_t = 100.0)]
    //#[arg(short, long, default_value_t = 10.0)]
    pub tvl_threshold: f64,
    /// The target blockchain
    #[arg(long, default_value = "ethereum")]
    pub chain: String,

    #[arg(long, default_value = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")] // WETH
    //#[arg(long, default_value = "0xdac17f958d2ee523a2206206994597c13d831ec7")] // USDT
    pub start_token: String,

    #[arg(long, default_value = "false")]
    pub uniswap_pools: bool,

    #[arg(long, default_value = "4")]
    pub bf_max_iterations: usize,
    #[arg(long, default_value = "0.001")] // should be approximatelly the gass price in eth
    pub bf_amount_in_min: f64,
    #[arg(long, default_value = "1000.0")]
    pub bf_amount_in_max: f64,
    #[arg(long, default_value = "20")]
    pub bf_max_outer_iterations: usize,
    #[arg(long, default_value = "1e-4")]
    pub bf_gss_tolerance: f64,
    #[arg(long, default_value = "40")]
    pub bf_gss_max_iter: usize,
    
    /// Enable all console logs (both POOL and ARB)
    #[arg(long, default_value = "false")]
    pub log: bool,

    /// Enable POOL (quoter) logs printed by log_quoter_info!
    #[arg(long, default_value = "false")]
    pub log_pool: bool,

    /// Enable ARB logs printed by log_arb_info!
    #[arg(long, default_value = "false")]
    pub log_arb: bool,

    /// Export graph to JSON file after processing
    #[arg(long, default_value = "false")]
    pub export_graph: bool,

    /// Debug mód: graph.json és quoter log mentése, egy blokk után leáll
    #[arg(long, default_value = "false")]
    pub debug: bool,

    /// Playback mode: load graph.json and quoter_log.json, use these instead of the Tycho feed
    #[arg(long, default_value = "false")]
    pub playback: bool,
}

pub fn get_default_url(chain: &Chain) -> Option<String> {
    match chain {
        Chain::Ethereum => Some("tycho-beta.propellerheads.xyz".to_string()),
        Chain::Base => Some("tycho-base-beta.propellerheads.xyz".to_string()),
        Chain::Unichain => Some("tycho-unichain-beta.propellerheads.xyz".to_string()),
        _ => None,
    }
}

pub fn get_rpc_url(chain: &Chain) -> Option<String> {
    match chain {
        Chain::Ethereum => {
            dotenv::dotenv().ok();
            env::var("RPC_URL").ok().or_else(|| {
                eprintln!("RPC_URL not set, using default for Ethereum");
                Some("https://mainnet.infura.io/v3/YOUR_INFURA_KEY".to_string())
            })
        }
        Chain::Base => Some("https://mainnet.base.org".to_string()),
        Chain::Unichain => Some("https://unichain.example.org".to_string()),
        _ => None,
    }
}

//pub async fn get_gas_price(chain: &Chain) -> anyhow::Result<BigUint> {
//    let url = get_rpc_url(chain).ok_or_else(|| anyhow::anyhow!("Missing RPC URL for chain"))?;
//    let provider = Provider::<Http>::try_from(url.as_str())?
//        .interval(std::time::Duration::from_millis(200)); // optional rate limit
//    let gas_price = provider.get_gas_price().await?;
//    Ok(gas_price)
//}

pub fn register_exchanges(
    builder: ProtocolStreamBuilder,
    chain: &Chain,
    tvl_filter: ComponentFilter,
    uniswap_pools: bool
) -> ProtocolStreamBuilder {
    if !uniswap_pools {
        register_all_exchanges(builder, chain, tvl_filter)
    } else {
        register_uniswap_exchanges(builder, chain, tvl_filter)
    }
}

pub fn register_uniswap_exchanges(
    mut builder: ProtocolStreamBuilder,
    chain: &Chain,
    tvl_filter: ComponentFilter
) -> ProtocolStreamBuilder {
    match chain {
        Chain::Ethereum => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None);
        }
        Chain::Base => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None);
        }
        Chain::Unichain => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None);
        }
        _ => {}
    }
    builder
}

fn register_all_exchanges(
    mut builder: ProtocolStreamBuilder,
    chain: &Chain,
    tvl_filter: ComponentFilter,
) -> ProtocolStreamBuilder {
    match chain {
        Chain::Ethereum => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV2State>("sushiswap_v2", tvl_filter.clone(), None)
                .exchange::<PancakeswapV2State>("pancakeswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("pancakeswap_v3", tvl_filter.clone(), None)
                .exchange::<EVMPoolState<PreCachedDB>>(
                    "vm:balancer_v2",
                    tvl_filter.clone(),
                    Some(balancer_v2_pool_filter),
                )
                .exchange::<UniswapV4State>("uniswap_v4", tvl_filter.clone(), None)
                .exchange::<EkuboState>("ekubo_v2", tvl_filter.clone(), None)
                .exchange::<EVMPoolState<PreCachedDB>>(
                    "vm:curve",
                    tvl_filter.clone(),
                    Some(curve_pool_filter),
                );
        }
        Chain::Base => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
        }
        Chain::Unichain => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
                .exchange::<UniswapV4State>("uniswap_v4", tvl_filter.clone(), None)
        }
        _ => {}
    }
    builder
}

pub fn load_blacklist_from_file(path: &str) -> std::io::Result<HashSet<Vec<u8>>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut set = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(bytes) = decode(line.trim()) {
            set.insert(bytes);
            println!("Loaded token from blacklist: {}", line);
        }
    }

    Ok(set)
}

