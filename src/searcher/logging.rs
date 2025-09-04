use std::sync::atomic::{AtomicBool, Ordering};

// Global toggle for console logging macros like log_quoter_info!, log_arb_info!, etc.
static LOG_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable() {
    LOG_ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable() {
    LOG_ENABLED.store(false, Ordering::Relaxed);
}

pub fn is_enabled() -> bool {
    LOG_ENABLED.load(Ordering::Relaxed)
}

// Centralized logging macros
#[macro_export]
macro_rules! log_quoter_info {
    ($($arg:tt)*) => {
        if $crate::searcher::logging::is_enabled() {
            println!("[POOL] {}", format_args!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_arb_info {
    ($($arg:tt)*) => {
        if $crate::searcher::logging::is_enabled() {
            println!("[ARB] {}", format_args!($($arg)*));
        }
    };
}
