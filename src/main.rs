use rand::Rng;
struct Node {
    value: i32,
    neighbors: Vec<(usize, char)>,
}

// Define a struct to represent the graph
struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    // Create a new graph
    fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    // Add a new node to the graph
    fn add_node(&mut self, value: i32) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node {
            value,
            neighbors: Vec::new(),
        });
        index
    }

    // Add an edge between two nodes in the graph
    fn add_edge(&mut self, from: usize, to: usize, symbol: char) {
        if from < self.nodes.len() && to < self.nodes.len() {
            self.nodes[from].neighbors.push((to, symbol));
        }
    }

    // Print out the adjacency list representation of the graph
    fn print_adjacency_list(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            println!("Node {}: {:?}", i, node.value);
            println!("Neighbors: {:?}", node.neighbors);
        }
    }

    // Print the graph in the DOT language
    fn print_dot(&self) {
        println!("digraph G {{");
        for (i, node) in self.nodes.iter().enumerate() {
            println!("  {} [label=\"{}\"];", i, node.value);
            for &(neighbor, symbol) in &node.neighbors {
                println!("  {} -> {} [label=\"{}\"];", i, neighbor, symbol);
            }
        }
        println!("}}");
    }
    fn print_svg(&self, size: usize) {
        let mut svg = String::new();
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?><svg width=\"");
        svg.push_str(&size.to_string());
        svg.push_str("\" height=\"");
        svg.push_str(&size.to_string());
        svg.push_str("\">\n");
    
        let mut table = vec![vec![false; size * 2 - 1]; size * 2 - 1];
    
        for node in &self.nodes {
            let index = node.value as usize - 1; // Предполагаем, что узлы нумеруются с 1
            let row = index / size * 2;
            let col = index % size * 2;
    
            // Добавляем узел
            svg.push_str("  <rect x=\"");
            svg.push_str(&col.to_string());
            svg.push_str("00");
            svg.push_str("\" y=\"");
            svg.push_str(&row.to_string());
            svg.push_str("00");
            svg.push_str("\" width=\"200\" height=\"2\" fill=\"black\"/>\n");
    
            // Добавляем ребра
            for &(neighbor_index, symbol) in &node.neighbors {
                let neighbor_index = neighbor_index as usize;
                let neighbor_row = neighbor_index / size * 2;
                let neighbor_col = neighbor_index % size * 2;
    
                let dx = (neighbor_col as i32 - col as i32) / 2;
                let dy = (neighbor_row as i32 - row as i32) / 2;
    
                if dx == 1 {
                    // Вправо
                    table[row][col + 1] = true;
                } else if dx == -1 {
                    // Влево
                    table[row][col - 1] = true;
                } else if dy == 1 {
                    // Вниз
                    table[row + 1][col] = true;
                } else if dy == -1 {
                    // Вверх
                    table[row - 1][col] = true;
                }
            }
        }
    
        // Добавляем стены
        for i in 0..size * 2 - 1 {
            for j in 0..size * 2 - 1 {
                if table[i][j] == false {
                    svg.push_str("  <rect x=\"");
                    svg.push_str(&j.to_string());
                    svg.push_str("00");
                    svg.push_str("\" y=\"");
                    svg.push_str(&i.to_string());
                    svg.push_str("00");
                    svg.push_str("\" width=\"200\" height=\"200\" fill=\"white\" stroke=\"black\"/>\n");
                }
            }
        }
    
        svg.push_str("</svg>");
        println!("{}", svg);
    }
}
fn pseudo_random(seed: u32, a: u32, b: u32, m: u32) -> u32 {
    (a * (seed >> m) + b) % m
}
// Function to fill graph with a square
fn fill_graph_with_square(graph: &mut Graph, size: usize) {
    for i in 0..size {
        for j in 0..size {
            let value = i * size + j + 1;
            graph.add_node(value as i32);
        }
    }

    for i in 0..size {
        for j in 0..size {
            let index = i * size + j;
            if i > 0 {
                graph.add_edge(index, index - size, 'N');
            }
            if i < size - 1 {
                graph.add_edge(index, index + size, 'S');
            }
            if j > 0 {
                graph.add_edge(index, index - 1, 'W');
            }
            if j < size - 1 {
                graph.add_edge(index, index + 1, 'E');
            }
        }
    }
}
enum Side {
    North,
    South,
    East,
    West,
}

impl Side {
    fn abbreviation(&self) ->  char {
        match self {
            Side::North => 'N',
            Side::South => 'S',
            Side::East => 'E',
            Side::West => 'W',
        }
    }

    fn full_name(&self) -> &'static str {
        match self {
            Side::North => "North",
            Side::South => "South",
            Side::East => "East",
            Side::West => "West",
        }
    }

    fn opposite(&self) -> Side {
        match self {
            Side::North => Side::South,
            Side::South => Side::North,
            Side::East => Side::West,
            Side::West => Side::East,
        }
    }
}
fn rand_dfs(graph: &mut Graph, node_index: usize, size: usize) {

    // Получаем соседей текущего узла
    let mut neighbors = Vec::new();

    loop {
        let north = node_index as i32 + size as i32;
        let south = node_index as i32 - size as i32;
        let east = node_index as i32 + 1;
        let west = node_index as i32 - 1;
        let sizei32 = size as i32;
        if north > 0
            && north < sizei32 * sizei32
            && (graph.nodes[north as usize].neighbors.len() == 0)
        {
            neighbors.push((north as usize,Side::North));
        }
        if south > 0 && south <  sizei32 * sizei32 && (graph.nodes[south as usize].neighbors.len() == 0) {
            neighbors.push((south as usize,Side::South));
        }
        if (east > 0 && (east % sizei32) != 0) && (graph.nodes[east as usize].neighbors.len() == 0) {
            neighbors.push((east as usize, Side::East));
        }
        if (west > 0 && (west % sizei32) != (sizei32 - 1) ) && (graph.nodes[west as usize].neighbors.len() == 0) {
            neighbors.push((west as usize, Side::West));
        }

        if neighbors.len() == 0 {
            return;
        }
        let i = generate_random_number(neighbors.len() as u32);
        graph.add_edge(node_index, neighbors[i as usize].0, neighbors[i as usize].1.abbreviation());
        graph.add_edge( neighbors[i as usize].0,node_index, neighbors[i as usize].1.opposite().abbreviation());
        rand_dfs(graph, neighbors[i as usize].0, size);
        neighbors.clear();
    }

    // Перемешиваем соседей для случайного выбора
}
static mut GLOBAL_RNG:Option<rand::prelude::ThreadRng>  = None; 
fn generate_random_number(n: u32) -> u32 {
    unsafe{
        GLOBAL_RNG.as_mut().unwrap().gen_range(0..=n-1)
    }
}
fn fill_rand_dfs(graph: &mut Graph, size: usize) {
    for i in 0..size {
        for j in 0..size {
            let value = i * size + j + 1;
            graph.add_node(value as i32);
        }
    }
    rand_dfs(graph, 0, size)
}


fn main() {
    let mut graph = Graph::new();
    unsafe{
    GLOBAL_RNG = Some(rand::thread_rng()); 
    }
    fill_rand_dfs(&mut graph,10);
    // Create a Hilbert curve of level 3 in a grid of size 8x8
    
    
   
    graph.print_svg(10);

}
