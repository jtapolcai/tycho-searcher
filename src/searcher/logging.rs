use std::sync::atomic::{AtomicBool, Ordering};

// Separate toggles for POOL (quoter) and ARB logs
static LOG_POOL_ENABLED: AtomicBool = AtomicBool::new(false);
static LOG_ARB_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable_pool() { LOG_POOL_ENABLED.store(true, Ordering::Relaxed); }
pub fn disable_pool() { LOG_POOL_ENABLED.store(false, Ordering::Relaxed); }
pub fn is_pool_enabled() -> bool { LOG_POOL_ENABLED.load(Ordering::Relaxed) }

pub fn enable_arb() { LOG_ARB_ENABLED.store(true, Ordering::Relaxed); }
pub fn disable_arb() { LOG_ARB_ENABLED.store(false, Ordering::Relaxed); }
pub fn is_arb_enabled() -> bool { LOG_ARB_ENABLED.load(Ordering::Relaxed) }

// Centralized logging macros
#[macro_export]
macro_rules! log_quoter_info {
    ($($arg:tt)*) => {
    if $crate::searcher::logging::is_pool_enabled() {
            println!("[POOL] {}", format_args!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_arb_info {
    ($($arg:tt)*) => {
    if $crate::searcher::logging::is_arb_enabled() {
            println!("[ARB] {}", format_args!($($arg)*));
        }
    };
}
