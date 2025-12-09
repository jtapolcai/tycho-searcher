// source .env && cargo run 
// === Modules ===
pub mod searcher;

use std::{
    env,
    str::FromStr,
};
use clap::Parser;
use std::collections::HashSet;

use tycho_client::feed::component_tracker::ComponentFilter;
use tycho_common::models::Chain;

use tycho_simulation::{
    evm::{
        stream::ProtocolStreamBuilder,
    },
    protocol::models::Update,
    utils::load_all_tokens,
};

use crate::searcher::{Searcher};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use crate::command_line_parameters::register_exchanges;

pub mod command_line_parameters;
use crate::command_line_parameters::{Cli, get_default_url, load_blacklist_from_file};
use crate::searcher::price_quoter::load_quoter_cache_once;


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let cli = Cli::parse();

    // Törli a quoter_log.json tartalmát debug módban
    if cli.debug {
        use std::fs::File;
        File::create("quoter_log.json").ok();
    }

    let chain = Chain::from_str(&cli.chain)
        .unwrap_or_else(|_| panic!("Unknown chain {}", cli.chain));

    let tycho_url = env::var("TYCHO_URL").unwrap_or_else(|_| {
        get_default_url(&chain).unwrap_or_else(|| panic!("Unknown URL for chain {}", cli.chain))
    });

    let tycho_api_key: String =
        env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    // Only check Ethereum node if not in playback mode
    let args: Vec<String> = std::env::args().collect();
    let is_playback = args.iter().any(|a| a == "--playback");
    let rpc_url = std::env::var("RPC_URL").unwrap_or_else(|_| "".to_string());
    if !is_playback {
        if rpc_url.is_empty() {
            eprintln!("[ERROR] RPC_URL env variable is not set. Exiting.");
            std::process::exit(1);
        }
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        });
        let resp = client.post(&rpc_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await;

        match resp {
            Ok(r) => {
                if !r.status().is_success() {
                    eprintln!("[ERROR] Ethereum node unreachable (HTTP {}). Exiting.", r.status());
                    std::process::exit(1);
                }
                let body = r.text().await.unwrap_or_default();
                if !body.contains("result") {
                    eprintln!("[ERROR] Ethereum node did not return block number. Response: {}", body);
                    std::process::exit(1);
                }
                println!("[INFO] Ethereum node is reachable. Block number response: {}", body);
            }
            Err(e) => {
                eprintln!("[ERROR] Failed to reach Ethereum node: {}. Exiting.", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("[PLAYBACK] Playback mode enabled, skipping Ethereum node check.");
        //init_playback_mode();
        load_quoter_cache_once();
    }
    println!("Starting searcher...");
    let blacklist = load_blacklist_from_file("blacklist_tokens.txt")
    .unwrap_or_else(|_| {
        eprintln!("Failed to load blacklist_tokens.txt, continuing with empty blacklist.");
        HashSet::new()
    });

    let (block_update_tx, block_update_rx) = mpsc::channel::<Update>(100);

    let searcher = Searcher::new(
        block_update_rx,
        cli.clone(),
        blacklist.clone(),
        rpc_url.clone(),
    );

    // Spawn the searcher task (same behaviour as tycho-searcher-cpp)
    let searcher_task = tokio::spawn(async move {
        if let Err(e) = searcher.run().await {
            eprintln!("Searcher error: {:?}", e);
        } else {
            println!("Searcher finished successfully.");
        }
    });
    println!("Searcher started successfully");

    // Console logging switches: --log is master; --log-pool and --log-arb control categories
    let pool_on = cli.log || cli.log_pool;
    let arb_on = cli.log || cli.log_arb;
    if pool_on { println!("[LOG] POOL logs enabled"); crate::searcher::logging::enable_pool(); } else { crate::searcher::logging::disable_pool(); }
    if arb_on  { println!("[LOG] ARB  logs enabled"); crate::searcher::logging::enable_arb(); } else { crate::searcher::logging::disable_arb(); }

    let playback_mode = cli.playback;
    if playback_mode {
        // Playback mode: only run the searcher, do not start the Tycho feed
        let searcher = match crate::searcher::Searcher::from_graph_json(
            cli.clone(),
            blacklist.clone(),
            rpc_url.clone(),
            "graph.json",
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[PLAYBACK] Failed to load graph.json: {}", e);
                return Ok(());
            }
        };
        let mut searcher = searcher;
        searcher.playback_cycle_search();
        return Ok(());
    }
    let all_tokens = load_all_tokens(
        tycho_url.as_str(),
        false,
        Some(tycho_api_key.as_str()),
        true, // compression (set to true to match C++ logic)
        chain,
        Some(100i32),
        None,
    ).await?;
    println!("Loaded {} tokens for chain {} from {}", all_tokens.len(), chain, tycho_url);
    let tvl_filter = ComponentFilter::with_tvl_range(cli.tvl_threshold, cli.tvl_threshold);
    println!("Building protocol stream for TVL>={}ETH",cli.tvl_threshold);
    let mut protocol_stream =
        register_exchanges(ProtocolStreamBuilder::new(&tycho_url, chain), &chain, tvl_filter, cli.uniswap_pools)
            .auth_key(Some(tycho_api_key.clone()))
            .skip_state_decode_failures(true)
            .set_tokens(all_tokens)
            .await
            .build()
            .await
            .expect("Failed building protocol stream");
    println!("Protocol stream built successfully");

    use futures::StreamExt;
    println!("Waiting for block updates from tycho feed...");
    let tx_clone = block_update_tx.clone();
    while let Some(msg) = protocol_stream.next().await {
        match msg {
            Ok(block_update) => {
                println!("Received block update from tycho: {} ({})", block_update.block_number_or_timestamp, searcher_task.is_finished() );
                if let Err(e) = tx_clone.send(block_update).await {
                    eprintln!("Failed to send block update to searcher: {}", e);
                    break;
                }
                // If debug mode is enabled, stop after the first successfu {l block
                if searcher_task.is_finished() {
                    println!("Stopping as searcher is finished");
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error receiving block update: {}", e);
            }
        }
    }

    // Close the sender so the searcher run() loop receives None and exits
    drop(block_update_tx);

    // Await the spawned searcher task to finish cleanly
    match searcher_task.await {
        Ok(_) => println!("Searcher task finished."),
        Err(e) => eprintln!("Searcher task failed: {:?}", e),
    }

    println!("Tycho feed thread finished");
    Ok(())
}
