use log::{info, error};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{StreamExt, SinkExt};
use url::Url;

pub async fn run_tick_client(url: &str) {
    let url = match Url::parse(url) {
        Ok(u) => u,
        Err(e) => {
            error!("Failed to parse URL: {}", e);
            return;
        }
    };

    let (ws_stream, _) = match connect_async(url).await {
        Ok(res) => res,
        Err(e) => {
            error!("Failed to connect: {}", e);
            return;
        }
    };

    info!("WebSocket connected");

    let (mut write, mut read) = ws_stream.split();

    while let Some(message) = read.next().await {
        match message {
            Ok(msg) => {
                match msg {
                    Message::Text(text) => {
                        info!("Received text tick: {}", text);
                    },
                    Message::Binary(bin) => {
                        info!("Received binary tick of len: {}", bin.len());
                    },
                    Message::Ping(ping) => {
                        let _ = write.send(Message::Pong(ping)).await;
                    },
                    _ => {}
                }
            }
            Err(e) => {
                error!("Error receiving message: {}", e);
                break;
            }
        }
    }
}
