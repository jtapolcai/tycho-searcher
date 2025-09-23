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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let cli = Cli::parse();
    let chain = Chain::from_str(&cli.chain)
        .unwrap_or_else(|_| panic!("Unknown chain {}", cli.chain));

    let tycho_url = env::var("TYCHO_URL").unwrap_or_else(|_| {
        get_default_url(&chain).unwrap_or_else(|| panic!("Unknown URL for chain {}", cli.chain))
    });

    let tycho_api_key: String =
        env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    let rpc_url = env::var("RPC_URL").expect("RPC_URL env variable should be set");

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
        blacklist,
        rpc_url,
    );

    // Console logging switches: --log is master; --log-pool and --log-arb control categories
    let pool_on = cli.log || cli.log_pool;
    let arb_on = cli.log || cli.log_arb;
    if pool_on { println!("[LOG] POOL logs enabled"); crate::searcher::logging::enable_pool(); } else { crate::searcher::logging::disable_pool(); }
    if arb_on  { println!("[LOG] ARB  logs enabled"); crate::searcher::logging::enable_arb(); } else { crate::searcher::logging::disable_arb(); }

    let searcher_task = tokio::spawn(async move {
        if let Err(e) = searcher.run().await {
            eprintln!("Searcher error: {:?}", e);
        } else {
            println!("Searcher finished successfully.");
        }
    });
    println!("Searcher started successfully");

    let tycho_message_processor: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        println!("Starting Tycho feed thread...");
        let all_tokens = load_all_tokens(
            tycho_url.as_str(),
            false,
            Some(tycho_api_key.as_str()),
            chain,
            Some(100 as i32),
            None,
        ).await;
        println!("Loaded {} tokens", all_tokens.len());

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

        println!("Waiting for block updates from tycho feed...");
        let tx_clone = block_update_tx.clone();
        use futures::StreamExt;
        while let Some(msg) = protocol_stream.next().await {
            match msg {
                Ok(block_update) => {
                    println!("Received block update from tycho: {}", block_update.block_number_or_timestamp);
                    if let Err(e) = tx_clone.send(block_update).await {
                        eprintln!("Failed to send block update to searcher: {}", e);
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error receiving block update: {}", e);
                }
            }
        }

        println!("Tycho feed thread finished");
        Ok(())
    });

    
    let (tycho_result, searcher_result) = tokio::join!(
        tycho_message_processor,
        searcher_task
    );
    if let Err(e) = tycho_result {
        eprintln!("Tycho task failed: {:?}", e);
    }

    if let Err(e) = searcher_result {
        eprintln!("Searcher task failed: {:?}", e);
    }

    Ok(())
}
