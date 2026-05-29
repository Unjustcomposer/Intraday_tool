use pyo3::prelude::*;
use log::{info, error, debug};
use std::collections::HashMap;

pub mod websocket;

#[pyclass]
pub struct FastOrderManager {
    orders: HashMap<String, String>,
}

#[pymethods]
impl FastOrderManager {
    #[new]
    pub fn new() -> Self {
        FastOrderManager {
            orders: HashMap::new(),
        }
    }

    pub fn route_order(&mut self, symbol: &str, quantity: f64, side: &str) -> PyResult<String> {
        let order_id = format!("{}-{}-{}", symbol, quantity, side);
        debug!("Routing order: {} {} {}", side, quantity, symbol);
        self.orders.insert(order_id.clone(), symbol.to_string());
        info!("Order routed with ID: {}", order_id);
        Ok(order_id)
    }

    pub fn cancel_order(&mut self, order_id: &str) -> PyResult<bool> {
        if self.orders.remove(order_id).is_some() {
            info!("Order cancelled: {}", order_id);
            Ok(true)
        } else {
            error!("Failed to cancel order: {}", order_id);
            Ok(false)
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn core_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    
    m.add_class::<FastOrderManager>()?;
    Ok(())
}
