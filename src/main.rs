use actix_web::{get, post, web::{self, Json}, App, HttpRequest, HttpResponse, HttpServer, Responder,middleware};
use serde::{Deserialize, Serialize};
use std::{env, fs, io};
use std::sync::RwLock;
use rand::rngs::ThreadRng;

mod graph;
use crate::graph::{Graph,fill_rand_dfs_rect};




/// create a tweet `/tweets`


#[actix_web::main]

async fn main() -> io::Result<()> {
    env::set_var("RUST_LOG", "actix_web=debug,actix_server=info");
    env_logger::init();

    HttpServer::new(|| {
        App::new()
        .wrap(middleware::Logger::default())
            .service(graph::generate_graph)
            .service(graph::get_graph)
            .service(graph::check_automata)
            .service(graph::check_membership)
            .service(graph::check_memberships)
            .service(graph::check_table)
            .service(graph::get_path)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
